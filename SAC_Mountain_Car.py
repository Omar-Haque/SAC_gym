import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import time
import gymnasium as gym

buffer_size = 50_000
batch_size = 64

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def np2torch(a, dtype=torch.float32, device=device):
  return torch.as_tensor(a, dtype=dtype, device=device)

class ReplayMemoryBuffer:
    def __init__(self, obs_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act[0]
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])

class QValue(nn.Module):
  def __init__(self, obs_dim, actions_dim):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(obs_dim + actions_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 1)
    )
  
  def forward(self, obs, action):
    if isinstance(obs, np.ndarray):
      obs = np2torch(obs)
    if isinstance(action, np.ndarray):
      action = np2torch(action)

    x = torch.cat([obs, action], dim=-1)

    return self.network(x)

class PolicyPi(nn.Module):
  def __init__(self, obs_dim):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(obs_dim, 256),
      nn.ReLU(),
    )
    
    self.mu = nn.Linear(256, 1)
    self.log_std = nn.Linear(256, 1)
  
  def forward(self, obs: np.ndarray | torch.Tensor):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2
    if isinstance(obs, np.ndarray):
      obs = np2torch(obs)
    x = self.network(obs)

    mu = self.mu(x)
    log_std = self.log_std(x)
    log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
    std = torch.exp(log_std)
    
    return mu, std
  
  def sample_action(self, obs: np.ndarray | torch.Tensor) -> tuple[torch.tensor, torch.tensor]:
    mu, std = self.forward(obs)

    pi_distribution = Normal(mu, std)

    a_bar = pi_distribution.rsample()
    action = torch.tanh(a_bar)

    log_prob = pi_distribution.log_prob(a_bar)

    log_prob -= torch.log(1 - action.pow(2) + 1e-6)
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return action, log_prob

class SAC:
  def __init__(self, observation_shape, action_space_dim, gamma=0.99, alpha = 0.2, polyak=0.995, Q_lr=3e-4, pi_lr=3e-4):
    self.replayMemoryBuffer = ReplayMemoryBuffer(observation_shape, buffer_size)
    
    self.gamma  = gamma
    self.polyak = polyak
    self.alpha  = alpha
    self.Q_lr   = Q_lr
    self.pi_lr  = pi_lr

    self.Q1        = QValue(obs_dim, actions_dim).to(device)
    self.Q2        = QValue(obs_dim, actions_dim).to(device)
    self.Q1_target = QValue(obs_dim, actions_dim).to(device)
    self.Q2_target = QValue(obs_dim, actions_dim).to(device)
    self.pi        = PolicyPi(obs_dim).to(device)
    self.Q1_target.load_state_dict(self.Q1.state_dict())
    self.Q2_target.load_state_dict(self.Q2.state_dict())
    
    self.Q1_optim = optim.Adam(self.Q1.parameters(), lr=self.Q_lr)
    self.Q2_optim = optim.Adam(self.Q2.parameters(), lr=self.Q_lr)
    self.pi_optim = optim.Adam(self.pi.parameters(), lr=self.pi_lr)
  
  def update(self, batch_size=batch_size):
    batch = self.replayMemoryBuffer.sample_batch(batch_size)
    s, a, r, s2, d = batch['s'], batch['a'], batch['r'], batch['s2'], batch['d']

    # Convert to tensors
    s  = np2torch(s)                # Shape: [batch_size, 32]
    a  = np2torch(a).unsqueeze(-1)  # Shape:[batch_size, 1]
    r  = np2torch(r)
    s2 = np2torch(s2)
    d  = np2torch(d)
    
    # ----------------- 1. Critic (Q) Update -----------------
    with torch.no_grad():
      a2, lpi2     = self.pi.sample_action(s2) # a2 is the action taken at s2(next_obs)
      q_1_target_v = self.Q1_target(s2, a2)
      q_2_target_v = self.Q2_target(s2, a2)
      compare_q_target = torch.min(q_1_target_v, q_2_target_v)
      y = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * (compare_q_target - self.alpha * lpi2)
    q_1_v = self.Q1(s, a)
    q_2_v = self.Q2(s, a)
    # Two Q-functions to mirigate positive bias in the policy improvement step
    q_1_loss = F.mse_loss(q_1_v, y) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    q_2_loss = F.mse_loss(q_2_v, y) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    
    
    # Update Q-network
    self.Q1_optim.zero_grad()
    q_1_loss.backward()
    self.Q1_optim.step()
    
    self.Q2_optim.zero_grad()
    q_2_loss.backward()
    self.Q2_optim.step()

    
    # ----------------- 2. Actor and Alpha Update ----------------- #
    # Freeze Q-networks to reduce computation and improve stability
    for p in self.Q1.parameters(): p.requires_grad = False
    for p in self.Q2.parameters(): p.requires_grad = False
    
    a1, lpi1 = self.pi.sample_action(s)
    q_1_a1_v = self.Q1(s, a1)
    q_2_a2_v = self.Q2(s, a1)
    compare_q_a1_v = torch.min(q_1_a1_v, q_2_a2_v)
    
    # gradient ascent
    # CHANGE: pi_loss = -torch.sum(compare_q_a1_v - self.alpha * lpi1)
    pi_loss = (self.alpha * lpi1 - compare_q_a1_v).mean()
    
    # Update policy network
    self.pi_optim.zero_grad()
    pi_loss.backward()
    self.pi_optim.step()
    
    # Unfreeze Q-networks
    for p in self.Q1.parameters(): p.requires_grad = True
    for p in self.Q2.parameters(): p.requires_grad = True
    
    
    # ----------------- 3. Target Network Update ----------------- #
    with torch.no_grad():
      for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
        target_param.data.mul_(self.polyak)
        target_param.data.add_((1 - self.polyak) * param.data)

      for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
        target_param.data.mul_(self.polyak)
        target_param.data.add_((1 - self.polyak) * param.data)

    return q_1_loss, q_2_loss, pi_loss

# env = gym.make('MountainCarContinuous-v0', render_mode="human")
env = gym.make('MountainCarContinuous-v0')


# dimension variables
obs_dim = env.observation_space.shape[0] # 2
actions_dim = env.action_space.shape[0]  # 1

agent = SAC(obs_dim, actions_dim)
init_steps = 20_000

init_steps_counter = 0
while init_steps > init_steps_counter:
  obs, _ = env.reset()
  loop_done = False
  while not loop_done:
    # print(f"init_steps_counter: {init_steps_counter}")
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)
    agent.replayMemoryBuffer.store(obs, action, reward, next_obs, terminated)
    
    obs = next_obs
    init_steps_counter += 1
    
    loop_done = terminated or truncated
    if terminated:
        print("WIN")
    if init_steps <= init_steps_counter:
        break

print(f"Initialization complete. Buffer size: {agent.replayMemoryBuffer.size}")

num_episodes = 2000
total_time_steps = 0
current_episode_time_steps = 0
update_period_timestep = 1
rewards = np.zeros(shape=num_episodes)

for episode in range(num_episodes):
    obs, _ = env.reset()
    ep_reward = 0
    env_reward = 0
    done = False
    while not done:
        action_tensor, _ = agent.pi.sample_action(obs)
        action_numpy = action_tensor.cpu().detach().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_numpy)
        
        position_reward = 10 * next_obs[0] 
        velocity_reward = 5 * abs(next_obs[1])
        shaped_reward = reward + position_reward + velocity_reward
        
        agent.replayMemoryBuffer.store(obs, action_numpy, shaped_reward, next_obs, terminated)
        obs = next_obs
        ep_reward += shaped_reward
        env_reward += reward
        
        total_time_steps += 1
        current_episode_time_steps += 1
        
        # print(f"Episode {episode}, current_episode_time_steps: {current_episode_time_steps}")

        if (total_time_steps % update_period_timestep == 0):
            agent.update()
        
        if terminated:
            print("WIN")
        
        if terminated or truncated:
            break
        
    current_episode_time_steps = 0
    rewards[episode] = env_reward
    print(f"Episode {episode}, ep_reward {ep_reward:.2f}, env_reward {env_reward:.2f}")

env.close()

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Rewards")
plt.show()