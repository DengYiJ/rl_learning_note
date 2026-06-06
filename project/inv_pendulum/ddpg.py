from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim

from inverted_pendulum_rl.config import ExperimentConfig
from inverted_pendulum_rl.envs import make_env
from inverted_pendulum_rl.models import DeterministicActor, QCritic
from inverted_pendulum_rl.plotting import plot_training_curves
from inverted_pendulum_rl.replay_buffer import ReplayBuffer
from inverted_pendulum_rl.utils import ensure_dir, moving_average, set_seed, write_json, write_metrics


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def train_ddpg(config: ExperimentConfig):
    set_seed(config.seed)
    env = make_env(reward_delay=config.reward_delay, seed=config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])
    device = torch.device("cpu")

    actor = DeterministicActor(state_dim, config.hidden_dim, action_dim, action_limit).to(device)
    critic = QCritic(state_dim, action_dim, config.hidden_dim).to(device)
    target_actor = deepcopy(actor).to(device)
    target_critic = deepcopy(critic).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr)
    buffer = ReplayBuffer(config.buffer_size)

    out_dir = config.output_dir()
    ensure_dir(out_dir)
    rows: list[dict] = []
    episode_rewards: list[float] = []
    true_rewards: list[float] = []
    total_steps = 0

    for episode in range(config.episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_true_reward = 0.0

        for _ in range(config.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]

            if total_steps < config.warmup_steps:
                action = np.random.uniform(-action_limit, action_limit, size=action_dim).astype(np.float32)
            else:
                noise = np.random.normal(0.0, config.noise_sigma, size=action_dim).astype(np.float32)
                action = np.clip(action + noise, -action_limit, action_limit)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, float(done))
            state = next_state

            episode_reward += reward
            episode_true_reward += float(info["true_reward"])
            total_steps += 1

            if len(buffer) >= max(config.batch_size, config.update_after):
                batch = buffer.sample(config.batch_size)
                states, actions, rewards, next_states, dones = batch
                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)

                with torch.no_grad():
                    next_actions = target_actor(next_states_t)
                    next_q = target_critic(next_states_t, next_actions)
                    target_q = rewards_t + config.gamma * (1.0 - dones_t) * next_q

                current_q = critic(states_t, actions_t)
                critic_loss = nn.functional.mse_loss(current_q, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(states_t, actor(states_t)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(target_actor, actor, config.tau)
                soft_update(target_critic, critic, config.tau)

            if done:
                break

        episode_rewards.append(episode_reward)
        true_rewards.append(episode_true_reward)
        rows.append(
            {
                "episode": episode + 1,
                "episodic_reward": episode_reward,
                "true_reward": episode_true_reward,
                "smoothed_reward": moving_average(episode_rewards, config.smooth_window)[-1],
                "smoothed_true_reward": moving_average(true_rewards, config.smooth_window)[-1],
            }
        )
        if (episode + 1) % 10 == 0:
            print(
                f"[DDPG][delay={config.reward_delay}] "
                f"episode={episode + 1} reward={episode_reward:.2f} true_reward={episode_true_reward:.2f}"
            )

    csv_path = out_dir / "metrics.csv"
    write_metrics(csv_path, rows)
    plot_training_curves(csv_path, out_dir / "training_curve.png", f"DDPG | delay={config.reward_delay}")
    torch.save(actor.state_dict(), out_dir / "actor.pt")
    torch.save(critic.state_dict(), out_dir / "critic.pt")
    write_json(
        out_dir / "summary.json",
        {
            "algorithm": "DDPG",
            "reward_delay": config.reward_delay,
            "episodes": config.episodes,
            "final_smoothed_reward": rows[-1]["smoothed_reward"],
            "final_smoothed_true_reward": rows[-1]["smoothed_true_reward"],
        },
    )
    env.close()
    return csv_path
