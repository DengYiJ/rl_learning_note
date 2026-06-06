import json
from pathlib import Path

import numpy as np
import torch

from ddpg_bipedal.agent import DDPGAgent
from ddpg_bipedal.env import make_env, reset_env, step_env
from ddpg_bipedal.evaluate import evaluate_policy
from ddpg_bipedal.plotting import plot_training_curves
from ddpg_bipedal.replay_buffer import ReplayBuffer
from ddpg_bipedal.utils import moving_average, set_seed, write_metrics_csv


class DDPGTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        set_seed(self.config.seed)
        self.config.save()

        env = make_env(self.config.env_name, self.config.seed)
        eval_env = make_env(self.config.env_name, self.config.seed + 100)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_limit = float(env.action_space.high[0])

        agent = DDPGAgent(state_dim, action_dim, action_limit, self.config, self.device)
        replay_buffer = ReplayBuffer(self.config.buffer_size)

        rewards_history = []
        smoothed_rewards = []
        eval_points = []
        eval_rewards = []
        metrics_rows = []
        total_steps = 0
        best_eval_reward = float("-inf")

        for episode in range(1, self.config.max_episodes + 1):
            state = reset_env(env, seed=self.config.seed + episode)
            episode_reward = 0.0
            critic_losses = []
            actor_losses = []
            agent.reset_noise()

            for step in range(self.config.max_steps):
                if total_steps < self.config.warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, add_noise=True)

                next_state, reward, done, _ = step_env(env, action)
                replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                total_steps += 1

                should_update = (
                    total_steps >= self.config.update_after
                    and len(replay_buffer) >= self.config.batch_size
                    and total_steps % self.config.update_every == 0
                )
                if should_update:
                    for _ in range(self.config.update_iters):
                        cl, al = agent.update(replay_buffer, self.config.batch_size)
                        critic_losses.append(cl)
                        actor_losses.append(al)

                if done:
                    break

            rewards_history.append(episode_reward)
            smoothed_rewards = moving_average(rewards_history, self.config.smooth_window)
            avg_actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
            avg_critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
            eval_reward = ""

            if episode % self.config.eval_every == 0:
                eval_reward = evaluate_policy(
                    eval_env,
                    agent,
                    self.config.eval_episodes,
                    seed_offset=self.config.seed + 1000 + episode,
                )
                eval_points.append(episode - 1)
                eval_rewards.append(eval_reward)
                best_eval_reward = max(best_eval_reward, eval_reward)
                print(f"[Eval] episode={episode:03d} | avg_reward={eval_reward:8.2f}")

            metrics_rows.append(
                {
                    "episode": episode,
                    "train_reward": episode_reward,
                    "smoothed_reward": smoothed_rewards[-1],
                    "eval_reward": eval_reward,
                    "actor_loss": avg_actor_loss,
                    "critic_loss": avg_critic_loss,
                    "buffer_size": len(replay_buffer),
                    "total_steps": total_steps,
                }
            )

            print(
                f"Episode {episode:03d} | reward={episode_reward:8.2f} | smooth={smoothed_rewards[-1]:8.2f} | "
                f"buffer={len(replay_buffer):6d} | actor_loss={avg_actor_loss:8.4f} | critic_loss={avg_critic_loss:8.4f}"
            )

            if isinstance(eval_reward, float) and eval_reward >= self.config.solve_score:
                print(f"Environment solved at episode {episode} with eval reward {eval_reward:.2f}.")
                break

        checkpoint_dir = self.config.checkpoint_path()
        agent.save(checkpoint_dir)
        write_metrics_csv(self.config.metrics_path(), metrics_rows)
        plot_training_curves(
            rewards_history,
            smoothed_rewards,
            eval_points,
            eval_rewards,
            self.config.figure_path(),
        )
        self._write_run_summary(
            total_episodes=len(rewards_history),
            final_train_reward=rewards_history[-1] if rewards_history else None,
            best_eval_reward=best_eval_reward if best_eval_reward != float("-inf") else None,
            checkpoint_dir=checkpoint_dir,
            figure_path=self.config.figure_path(),
            metrics_path=self.config.metrics_path(),
        )

        env.close()
        eval_env.close()

    def _write_run_summary(
        self,
        total_episodes: int,
        final_train_reward,
        best_eval_reward,
        checkpoint_dir: Path,
        figure_path: Path,
        metrics_path: Path,
    ):
        summary = {
            "total_episodes": total_episodes,
            "final_train_reward": final_train_reward,
            "best_eval_reward": best_eval_reward,
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "figure_path": str(figure_path.resolve()),
            "metrics_path": str(metrics_path.resolve()),
        }
        self.config.summary_path().write_text(json.dumps(summary, indent=2), encoding="utf-8")
