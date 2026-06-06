from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from inverted_pendulum_rl.config import ExperimentConfig
from inverted_pendulum_rl.envs import DiscretePendulumEnv
from inverted_pendulum_rl.plotting import plot_training_curves
from inverted_pendulum_rl.utils import ensure_dir, moving_average, set_seed, write_json, write_metrics


def epsilon_greedy(q_values: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(len(q_values))
    return int(np.argmax(q_values))


def build_tabular_rows(episode_rewards: list[float], true_rewards: list[float]) -> list[dict]:
    smoothed_rewards = moving_average(episode_rewards, 10)
    smoothed_true_rewards = moving_average(true_rewards, 10)
    rows = []
    for idx, (reward, true_reward, smooth_reward, smooth_true) in enumerate(
        zip(episode_rewards, true_rewards, smoothed_rewards, smoothed_true_rewards),
        start=1,
    ):
        rows.append(
            {
                "episode": idx,
                "episodic_reward": reward,
                "true_reward": true_reward,
                "smoothed_reward": smooth_reward,
                "smoothed_true_reward": smooth_true,
            }
        )
    return rows


def save_run(config: ExperimentConfig, rows: list[dict], algorithm_name: str, extra_summary: dict | None = None) -> Path:
    out_dir = config.output_dir()
    ensure_dir(out_dir)
    csv_path = out_dir / "metrics.csv"
    write_metrics(csv_path, rows)
    plot_training_curves(csv_path, out_dir / "training_curve.png", f"{algorithm_name} | delay={config.reward_delay}")
    summary = {
        "algorithm": algorithm_name,
        "reward_delay": config.reward_delay,
        "episodes": config.episodes,
        "final_smoothed_reward": rows[-1]["smoothed_reward"],
        "final_smoothed_true_reward": rows[-1]["smoothed_true_reward"],
        "config": asdict(config),
    }
    if extra_summary:
        summary.update(extra_summary)
    write_json(out_dir / "summary.json", summary)
    return csv_path


def train_value_iteration(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    env = DiscretePendulumEnv(
        theta_bins=config.theta_bins,
        theta_dot_bins=config.theta_dot_bins,
        action_bins=config.action_bins,
        reward_delay=0,
        seed=config.seed,
    )
    q_table = np.zeros((env.state_space_n, env.action_space_n), dtype=np.float32)
    value = np.zeros(env.state_space_n, dtype=np.float32)

    for iteration in range(config.episodes):
        delta = 0.0
        new_value = np.zeros_like(value)
        for state in range(env.state_space_n):
            for action in range(env.action_space_n):
                next_state, _, reward, done = env.model_step(state, action)
                q_table[state, action] = reward + config.gamma * (0.0 if done else value[next_state])
            new_value[state] = float(np.max(q_table[state]))
            delta = max(delta, abs(new_value[state] - value[state]))
        value = new_value
        if (iteration + 1) % 10 == 0:
            print(f"[ValueIteration] iteration={iteration + 1} delta={delta:.6f}")
        if delta < 1e-4:
            print(f"[ValueIteration] converged at iteration {iteration + 1}")
            break

    episode_rewards = []
    true_rewards = []
    for _ in range(config.mc_eval_episodes):
        state_id, _, _ = env.reset()
        total_reward = 0.0
        total_true_reward = 0.0
        for _ in range(config.max_steps):
            action = int(np.argmax(q_table[state_id]))
            state_id, _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_true_reward += float(info["true_reward"])
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
        true_rewards.append(total_true_reward)

    rows = build_tabular_rows(episode_rewards, true_rewards)
    csv_path = save_run(
        config,
        rows,
        "Value Iteration",
        extra_summary={"iterations_used": iteration + 1},
    )
    np.save(config.output_dir() / "q_table.npy", q_table)
    env.close()
    return csv_path


def train_monte_carlo_control(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    env = DiscretePendulumEnv(
        theta_bins=config.theta_bins,
        theta_dot_bins=config.theta_dot_bins,
        action_bins=config.action_bins,
        reward_delay=config.reward_delay,
        seed=config.seed,
    )
    q_table = np.zeros((env.state_space_n, env.action_space_n), dtype=np.float32)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    epsilon = config.epsilon
    episode_rewards = []
    true_rewards = []

    for episode in range(config.episodes):
        state_id, _, _ = env.reset()
        trajectory = []
        total_reward = 0.0
        total_true_reward = 0.0

        for _ in range(config.max_steps):
            action = epsilon_greedy(q_table[state_id], epsilon)
            next_state_id, _, reward, terminated, truncated, info = env.step(action)
            trajectory.append((state_id, action, reward))
            total_reward += reward
            total_true_reward += float(info["true_reward"])
            state_id = next_state_id
            if terminated or truncated:
                break

        g = 0.0
        visited = set()
        for state, action, reward in reversed(trajectory):
            g = config.gamma * g + reward
            if (state, action) in visited:
                continue
            visited.add((state, action))
            returns_sum[(state, action)] += g
            returns_count[(state, action)] += 1
            q_table[state, action] = returns_sum[(state, action)] / returns_count[(state, action)]

        episode_rewards.append(total_reward)
        true_rewards.append(total_true_reward)
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        if (episode + 1) % 20 == 0:
            print(f"[MonteCarlo] episode={episode + 1} true_reward={total_true_reward:.2f}")

    rows = build_tabular_rows(episode_rewards, true_rewards)
    csv_path = save_run(config, rows, "Monte Carlo Control")
    np.save(config.output_dir() / "q_table.npy", q_table)
    env.close()
    return csv_path


def train_sarsa(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    env = DiscretePendulumEnv(
        theta_bins=config.theta_bins,
        theta_dot_bins=config.theta_dot_bins,
        action_bins=config.action_bins,
        reward_delay=config.reward_delay,
        seed=config.seed,
    )
    q_table = np.zeros((env.state_space_n, env.action_space_n), dtype=np.float32)
    epsilon = config.epsilon
    alpha = config.actor_lr
    episode_rewards = []
    true_rewards = []

    for episode in range(config.episodes):
        state_id, _, _ = env.reset()
        action = epsilon_greedy(q_table[state_id], epsilon)
        total_reward = 0.0
        total_true_reward = 0.0

        for _ in range(config.max_steps):
            next_state_id, _, reward, terminated, truncated, info = env.step(action)
            next_action = epsilon_greedy(q_table[next_state_id], epsilon)
            td_target = reward + config.gamma * q_table[next_state_id, next_action] * (0.0 if (terminated or truncated) else 1.0)
            q_table[state_id, action] += alpha * (td_target - q_table[state_id, action])
            state_id = next_state_id
            action = next_action
            total_reward += reward
            total_true_reward += float(info["true_reward"])
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        true_rewards.append(total_true_reward)
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        if (episode + 1) % 20 == 0:
            print(f"[SARSA] episode={episode + 1} true_reward={total_true_reward:.2f}")

    rows = build_tabular_rows(episode_rewards, true_rewards)
    csv_path = save_run(config, rows, "SARSA")
    np.save(config.output_dir() / "q_table.npy", q_table)
    env.close()
    return csv_path


def train_q_learning(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    env = DiscretePendulumEnv(
        theta_bins=config.theta_bins,
        theta_dot_bins=config.theta_dot_bins,
        action_bins=config.action_bins,
        reward_delay=config.reward_delay,
        seed=config.seed,
    )
    q_table = np.zeros((env.state_space_n, env.action_space_n), dtype=np.float32)
    epsilon = config.epsilon
    alpha = config.actor_lr
    episode_rewards = []
    true_rewards = []

    for episode in range(config.episodes):
        state_id, _, _ = env.reset()
        total_reward = 0.0
        total_true_reward = 0.0

        for _ in range(config.max_steps):
            action = epsilon_greedy(q_table[state_id], epsilon)
            next_state_id, _, reward, terminated, truncated, info = env.step(action)
            td_target = reward + config.gamma * np.max(q_table[next_state_id]) * (0.0 if (terminated or truncated) else 1.0)
            q_table[state_id, action] += alpha * (td_target - q_table[state_id, action])
            state_id = next_state_id
            total_reward += reward
            total_true_reward += float(info["true_reward"])
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        true_rewards.append(total_true_reward)
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        if (episode + 1) % 20 == 0:
            print(f"[QLearning] episode={episode + 1} true_reward={total_true_reward:.2f}")

    rows = build_tabular_rows(episode_rewards, true_rewards)
    csv_path = save_run(config, rows, "Q-Learning")
    np.save(config.output_dir() / "q_table.npy", q_table)
    env.close()
    return csv_path


class DQNNet(nn.Module):
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


def train_dqn(config: ExperimentConfig) -> Path:
    from inverted_pendulum_rl.replay_buffer import ReplayBuffer

    set_seed(config.seed)
    env = DiscretePendulumEnv(
        theta_bins=config.theta_bins,
        theta_dot_bins=config.theta_dot_bins,
        action_bins=config.action_bins,
        reward_delay=config.reward_delay,
        seed=config.seed,
    )
    device = torch.device("cpu")
    q_net = DQNNet(3, config.hidden_dim, env.action_space_n).to(device)
    target_net = DQNNet(3, config.hidden_dim, env.action_space_n).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=config.critic_lr)
    replay = ReplayBuffer(config.buffer_size)
    epsilon = config.epsilon
    episode_rewards = []
    true_rewards = []
    total_steps = 0

    for episode in range(config.episodes):
        state_id, obs, _ = env.reset()
        total_reward = 0.0
        total_true_reward = 0.0

        for _ in range(config.max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if random.random() < epsilon:
                action = random.randrange(env.action_space_n)
            else:
                with torch.no_grad():
                    action = int(torch.argmax(q_net(obs_tensor), dim=-1).item())

            next_state_id, next_obs, reward, terminated, truncated, info = env.step(action)
            replay.add(obs, np.array([action], dtype=np.int64), reward, next_obs, float(terminated or truncated))
            obs = next_obs
            state_id = next_state_id
            total_reward += reward
            total_true_reward += float(info["true_reward"])
            total_steps += 1

            if len(replay) >= max(config.batch_size, config.update_after):
                states, actions, rewards, next_states, dones = replay.sample(config.batch_size)
                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)

                q_values = q_net(states_t).gather(1, actions_t)
                with torch.no_grad():
                    next_q_values = target_net(next_states_t).max(dim=1, keepdim=True).values
                    td_target = rewards_t + config.gamma * (1.0 - dones_t) * next_q_values
                loss = nn.functional.mse_loss(q_values, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % 100 == 0:
                    target_net.load_state_dict(q_net.state_dict())

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        true_rewards.append(total_true_reward)
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        if (episode + 1) % 20 == 0:
            print(f"[DQN] episode={episode + 1} true_reward={total_true_reward:.2f}")

    rows = build_tabular_rows(episode_rewards, true_rewards)
    csv_path = save_run(config, rows, "DQN")
    torch.save(q_net.state_dict(), config.output_dir() / "q_net.pt")
    env.close()
    return csv_path


def summarize_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {
        "final_reward": float(df["episodic_reward"].iloc[-1]),
        "final_true_reward": float(df["true_reward"].iloc[-1]),
        "best_smoothed_true_reward": float(df["smoothed_true_reward"].max()),
        "final_smoothed_true_reward": float(df["smoothed_true_reward"].iloc[-1]),
    }
