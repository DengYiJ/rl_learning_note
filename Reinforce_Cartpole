import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ==========================
# Policy Network
# ==========================

class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        return self.net(x)


# ==========================
# REINFORCE Agent
# ==========================

class REINFORCE:

    def __init__(self,state_dim,action_dim,lr=1e-3,gamma=0.99):

        self.gamma = gamma

        self.policy = PolicyNet(state_dim,action_dim)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr
        )

    def choose_action(self,state):

        state = torch.FloatTensor(state)

        probs = self.policy(state)

        dist = Categorical(probs)

        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def update(self,rewards,log_probs):

        returns = []

        G = 0

        # 计算回报
        for r in reversed(rewards):

            G = r + self.gamma * G

            returns.insert(0,G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        # 标准化
        returns = (
            returns - returns.mean()
        ) / (returns.std() + 1e-8)

        loss = []

        for log_prob,G in zip(log_probs,returns):

            loss.append(-log_prob * G)

        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


# ==========================
# Training
# ==========================

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = REINFORCE(
    state_dim,
    action_dim
)

num_episodes = 1000

for episode in range(num_episodes):

    state,_ = env.reset()

    rewards = []
    log_probs = []

    done = False

    while not done:

        action,log_prob = agent.choose_action(state)

        next_state,reward,terminated,truncated,_ = env.step(action)

        done = terminated or truncated

        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state

    agent.update(rewards,log_probs)

    total_reward = sum(rewards)

    if episode % 10 == 0:

        print(
            f"Episode {episode}, "
            f"Reward={total_reward}"
        )

env.close()
