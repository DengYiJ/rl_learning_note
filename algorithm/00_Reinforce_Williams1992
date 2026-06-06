import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


env = gym.make("CartPole-v1")

policy = PolicyNet(4, 2)

optimizer = optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99

for episode in range(1000):

    state, _ = env.reset()

    rewards = []
    log_probs = []

    done = False

    while not done:

        state_tensor = torch.FloatTensor(state)

        probs = policy(state_tensor)

        dist = Categorical(probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())

        done = terminated or truncated

        rewards.append(reward)

        log_probs.append(log_prob)

        state = next_state

    # -------------------
    # Williams原始REINFORCE
    # -------------------

    R = 0

    for t, r in enumerate(rewards):
        R += (gamma ** t) * r

    loss = []

    for log_prob in log_probs:

        loss.append(-R * log_prob)

    loss = torch.stack(loss).sum()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
