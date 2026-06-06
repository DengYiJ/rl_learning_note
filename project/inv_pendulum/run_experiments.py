from pathlib import Path

import pandas as pd

from inverted_pendulum_rl.actor_critic import train_actor_critic
from inverted_pendulum_rl.config import ExperimentConfig
from inverted_pendulum_rl.ddpg import train_ddpg
from inverted_pendulum_rl.plotting import plot_comparison
from inverted_pendulum_rl.utils import ensure_dir


def summarize_result(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return {
        "final_reward": float(df["episodic_reward"].iloc[-1]),
        "final_true_reward": float(df["true_reward"].iloc[-1]),
        "best_smoothed_true_reward": float(df["smoothed_true_reward"].max()),
        "final_smoothed_true_reward": float(df["smoothed_true_reward"].iloc[-1]),
    }


def main():
    root = Path("inverted_pendulum_rl/runs")
    ensure_dir(root)

    configs = [
        ExperimentConfig(algo="actor_critic", reward_delay=0, episodes=120, actor_lr=3e-4, critic_lr=1e-3, seed=42),
        ExperimentConfig(algo="actor_critic", reward_delay=5, episodes=120, actor_lr=3e-4, critic_lr=1e-3, seed=42),
        ExperimentConfig(algo="ddpg", reward_delay=0, episodes=100, actor_lr=1e-3, critic_lr=2e-3, hidden_dim=128, seed=42),
        ExperimentConfig(algo="ddpg", reward_delay=5, episodes=100, actor_lr=1e-3, critic_lr=2e-3, hidden_dim=128, seed=42),
    ]

    results = {}
    for config in configs:
        if config.algo == "actor_critic":
            csv_path = train_actor_critic(config)
        elif config.algo == "ddpg":
            csv_path = train_ddpg(config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algo}")
        results[config.tag()] = csv_path

    plot_comparison(
        {
            "Actor-Critic no delay": results["actor_critic_delay0_seed42"],
            "Actor-Critic delay=5": results["actor_critic_delay5_seed42"],
            "DDPG no delay": results["ddpg_delay0_seed42"],
            "DDPG delay=5": results["ddpg_delay5_seed42"],
        },
        root / "comparison_true_reward.png",
        "Pendulum-v1 Comparison on True Reward",
        value_key="smoothed_true_reward",
    )

    summary_rows = []
    for name, csv_path in results.items():
        row = summarize_result(csv_path)
        row["experiment"] = name
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(root / "summary_table.csv", index=False)
    print(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()
