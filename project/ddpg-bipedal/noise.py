import numpy as np


class GaussianNoise:
    def __init__(self, action_dim: int, sigma: float):
        self.action_dim = action_dim
        self.sigma = sigma

    def sample(self):
        return np.random.normal(0.0, self.sigma, size=self.action_dim).astype(np.float32)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck 过程：时间相关的探索噪声，更适合连续控制任务。"""

    def __init__(self, action_dim: int, sigma: float = 0.2, theta: float = 0.15, dt: float = 1e-2):
        self.action_dim = action_dim
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_dim, dtype=np.float32)

    def sample(self):
        self.state += (
            self.theta * (-self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        )
        return self.state.astype(np.float32)