from pathlib import Path

import numpy as np
import torch
from torch import optim

from inverted_pendulum_rl.config import ExperimentConfig
from inverted_pendulum_rl.envs import make_env
from inverted_pendulum_rl.models import GaussianActor, ValueCritic
from inverted_pendulum_rl.plotting import plot_training_curves
from inverted_pendulum_rl.utils import ensure_dir, moving_average, set_seed, write_json, write_metrics


def train_actor_critic(config: ExperimentConfig):
    set_seed(config.seed)
    env = make_env(reward_delay=config.reward_delay, seed=config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])
    device = torch.device("cpu")

    actor = GaussianActor(state_dim, config.hidden_dim, action_dim, action_limit).to(device)
    critic = ValueCritic(state_dim, config.hidden_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr)

    out_dir = config.output_dir()
    ensure_dir(out_dir)
    rows: list[dict] = []
    episode_rewards: list[float] = []
    true_rewards: list[float] = []

    for episode in range(config.episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_true_reward = 0.0

        for _ in range(config.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mu, std = actor(state_tensor, config.policy_std_min, config.policy_std_max)
            dist = torch.distributions.Normal(mu, std)
            raw_action = dist.rsample()
            action = torch.clamp(raw_action, -action_limit, action_limit)
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

            next_state, reward, terminated, truncated, info = env.step(action.squeeze(0).detach().cpu().numpy())
            done = terminated or truncated

            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            value = critic(state_tensor).squeeze(-1)
            with torch.no_grad():
                next_value = critic(next_state_tensor).squeeze(-1)
                td_target = reward_tensor + config.gamma * next_value * (0.0 if done else 1.0)
            td_error = td_target - value

            critic_loss = td_error.pow(2).mean()
            actor_loss = -(log_prob * td_error.detach()).mean() - config.entropy_coef * dist.entropy().sum(dim=-1).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
            episode_reward += reward
            episode_true_reward += float(info["true_reward"])

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
                f"[ActorCritic][delay={config.reward_delay}] "
                f"episode={episode + 1} reward={episode_reward:.2f} true_reward={episode_true_reward:.2f}"
            )

    csv_path = out_dir / "metrics.csv"
    write_metrics(csv_path, rows)
    plot_training_curves(csv_path, out_dir / "training_curve.png", f"Actor-Critic | delay={config.reward_delay}")
    torch.save(actor.state_dict(), out_dir / "actor.pt")
    torch.save(critic.state_dict(), out_dir / "critic.pt")
    write_json(
        out_dir / "summary.json",
        {
            "algorithm": "Actor-Critic",
            "reward_delay": config.reward_delay,
            "episodes": config.episodes,
            "final_smoothed_reward": rows[-1]["smoothed_reward"],
            "final_smoothed_true_reward": rows[-1]["smoothed_true_reward"],
        },
    )
    env.close()
    return csv_path
