import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_limit: float):
        super().__init__()
        self.backbone = MLP(state_dim, hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.action_limit = action_limit

    def forward(self, state, log_std_min: float = -5.0, log_std_max: float = 1.0):
        features = self.backbone(state)
        mu = self.mu(features)
        log_std = torch.clamp(self.log_std(features), log_std_min, log_std_max)
        std = torch.exp(log_std)
        return mu, std


class ValueCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(state_dim, hidden_dim, 1)

    def forward(self, state):
        return self.net(state)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, action_limit: float):
        super().__init__()
        self.net = MLP(state_dim, hidden_dim, action_dim)
        self.action_limit = action_limit

    def forward(self, state):
        return torch.tanh(self.net(state)) * self.action_limit


class QCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
