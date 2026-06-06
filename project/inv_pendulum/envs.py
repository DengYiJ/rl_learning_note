from collections import deque
from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np


def pendulum_reward_from_observation(observation: np.ndarray, action: np.ndarray) -> float:
    cos_theta, sin_theta, theta_dot = observation
    theta = math.atan2(sin_theta, cos_theta)
    torque = float(np.clip(action[0], -2.0, 2.0))
    cost = theta**2 + 0.1 * theta_dot**2 + 0.001 * torque**2
    return -cost


@dataclass
class StepRecord:
    observation: np.ndarray
    action: np.ndarray


class DelayedRewardPendulum:
    """Return rewards using state/action from `reward_delay` steps ago."""

    def __init__(self, reward_delay: int = 0, seed: int | None = None):
        self.env = gym.make("Pendulum-v1")
        self.reward_delay = reward_delay
        self.seed = seed
        self.pending: deque[StepRecord] = deque()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.last_true_reward = 0.0
        self.current_observation: np.ndarray | None = None

    def reset(self):
        self.pending.clear()
        obs, info = self.env.reset(seed=self.seed)
        self.current_observation = obs.astype(np.float32)
        return self.current_observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if self.current_observation is None:
            raise RuntimeError("Environment must be reset before stepping.")
        state_before_action = self.current_observation.copy()
        next_obs, true_reward, terminated, truncated, info = self.env.step(action)
        next_obs = next_obs.astype(np.float32)
        self.last_true_reward = float(true_reward)
        self.pending.append(StepRecord(observation=state_before_action, action=action.copy()))

        if self.reward_delay == 0:
            delayed_reward = float(true_reward)
        elif len(self.pending) <= self.reward_delay:
            delayed_reward = 0.0
        else:
            delayed = self.pending.popleft()
            delayed_reward = pendulum_reward_from_observation(delayed.observation, delayed.action)

        info["true_reward"] = float(true_reward)
        info["delayed_reward"] = float(delayed_reward)
        self.current_observation = next_obs
        return next_obs, float(delayed_reward), terminated, truncated, info

    def close(self):
        self.env.close()


def make_env(reward_delay: int = 0, seed: int | None = None) -> DelayedRewardPendulum:
    return DelayedRewardPendulum(reward_delay=reward_delay, seed=seed)


def angle_normalize(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class DiscretePendulumEnv:
    """Discretized Pendulum-v1 for DP, MC, SARSA, Q-learning and DQN."""

    def __init__(
        self,
        theta_bins: int = 15,
        theta_dot_bins: int = 15,
        action_bins: int = 9,
        reward_delay: int = 0,
        seed: int | None = None,
    ):
        self.env = gym.make("Pendulum-v1")
        self.theta_bins = theta_bins
        self.theta_dot_bins = theta_dot_bins
        self.action_bins = action_bins
        self.reward_delay = reward_delay
        self.seed = seed
        self.theta_edges = np.linspace(-np.pi, np.pi, theta_bins + 1)
        self.theta_dot_edges = np.linspace(-8.0, 8.0, theta_dot_bins + 1)
        self.theta_centers = (self.theta_edges[:-1] + self.theta_edges[1:]) / 2.0
        self.theta_dot_centers = (self.theta_dot_edges[:-1] + self.theta_dot_edges[1:]) / 2.0
        self.action_values = np.linspace(-2.0, 2.0, action_bins, dtype=np.float32)
        self.pending: deque[StepRecord] = deque()
        self.current_observation: np.ndarray | None = None
        self.observation_space = self.env.observation_space
        self.action_space_n = action_bins
        self.state_space_n = theta_bins * theta_dot_bins

    def state_id_to_continuous(self, state_id: int) -> np.ndarray:
        theta_idx = state_id // self.theta_dot_bins
        theta_dot_idx = state_id % self.theta_dot_bins
        theta = self.theta_centers[theta_idx]
        theta_dot = self.theta_dot_centers[theta_dot_idx]
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def observation_to_state_id(self, observation: np.ndarray) -> int:
        theta = math.atan2(float(observation[1]), float(observation[0]))
        theta_dot = float(np.clip(observation[2], -8.0, 8.0))
        theta_idx = int(np.clip(np.digitize(theta, self.theta_edges) - 1, 0, self.theta_bins - 1))
        theta_dot_idx = int(np.clip(np.digitize(theta_dot, self.theta_dot_edges) - 1, 0, self.theta_dot_bins - 1))
        return theta_idx * self.theta_dot_bins + theta_dot_idx

    def state_id_to_model_state(self, state_id: int) -> np.ndarray:
        obs = self.state_id_to_continuous(state_id)
        theta = math.atan2(float(obs[1]), float(obs[0]))
        return np.array([theta, float(obs[2])], dtype=np.float32)

    def discrete_action_to_continuous(self, action_id: int) -> np.ndarray:
        return np.array([self.action_values[action_id]], dtype=np.float32)

    def reset(self):
        self.pending.clear()
        obs, info = self.env.reset(seed=self.seed)
        obs = obs.astype(np.float32)
        self.current_observation = obs
        return self.observation_to_state_id(obs), obs, info

    def step(self, action_id: int):
        if self.current_observation is None:
            raise RuntimeError("Environment must be reset before stepping.")
        action = self.discrete_action_to_continuous(action_id)
        state_before_action = self.current_observation.copy()
        next_obs, true_reward, terminated, truncated, info = self.env.step(action)
        next_obs = next_obs.astype(np.float32)
        self.pending.append(StepRecord(observation=state_before_action, action=action.copy()))

        if self.reward_delay == 0:
            delayed_reward = float(true_reward)
        elif len(self.pending) <= self.reward_delay:
            delayed_reward = 0.0
        else:
            delayed = self.pending.popleft()
            delayed_reward = pendulum_reward_from_observation(delayed.observation, delayed.action)

        self.current_observation = next_obs
        next_state_id = self.observation_to_state_id(next_obs)
        info["true_reward"] = float(true_reward)
        info["delayed_reward"] = float(delayed_reward)
        info["next_observation"] = next_obs.copy()
        return next_state_id, next_obs, float(delayed_reward), terminated, truncated, info

    def model_step(self, state_id: int, action_id: int):
        action = self.discrete_action_to_continuous(action_id)
        base_env = self.env.unwrapped
        base_env.state = self.state_id_to_model_state(state_id).copy()
        base_env.last_u = None
        obs = base_env._get_obs().astype(np.float32)
        next_obs, true_reward, terminated, truncated, _ = base_env.step(action)
        next_obs = next_obs.astype(np.float32)
        next_state_id = self.observation_to_state_id(next_obs)
        reward = float(true_reward)
        return next_state_id, next_obs, reward, terminated or truncated

    def close(self):
        self.env.close()
