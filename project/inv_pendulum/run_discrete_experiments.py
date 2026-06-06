from pathlib import Path

import pandas as pd

from inverted_pendulum_rl.config import ExperimentConfig
from inverted_pendulum_rl.discrete_algorithms import (
    summarize_csv,
    train_dqn,
    train_monte_carlo_control,
    train_q_learning,
    train_sarsa,
    train_value_iteration,
)
from inverted_pendulum_rl.plotting import plot_comparison
from inverted_pendulum_rl.utils import ensure_dir


def main():
    root = Path("inverted_pendulum_rl/discrete_runs")
    ensure_dir(root)

    configs = [
        ExperimentConfig(algo="value_iteration", reward_delay=0, episodes=80, max_steps=200, experiment_root=str(root)),
        ExperimentConfig(algo="monte_carlo", reward_delay=0, episodes=160, max_steps=200, actor_lr=0.2, experiment_root=str(root)),
        ExperimentConfig(algo="sarsa", reward_delay=0, episodes=180, max_steps=200, actor_lr=0.15, experiment_root=str(root)),
        ExperimentConfig(algo="q_learning", reward_delay=0, episodes=180, max_steps=200, actor_lr=0.15, experiment_root=str(root)),
        ExperimentConfig(
            algo="dqn",
            reward_delay=0,
            episodes=140,
            max_steps=200,
            critic_lr=1e-3,
            batch_size=64,
            update_after=200,
            buffer_size=30000,
            experiment_root=str(root),
        ),
        ExperimentConfig(algo="sarsa", reward_delay=5, episodes=180, max_steps=200, actor_lr=0.15, experiment_root=str(root)),
        ExperimentConfig(algo="q_learning", reward_delay=5, episodes=180, max_steps=200, actor_lr=0.15, experiment_root=str(root)),
        ExperimentConfig(
            algo="dqn",
            reward_delay=5,
            episodes=140,
            max_steps=200,
            critic_lr=1e-3,
            batch_size=64,
            update_after=200,
            buffer_size=30000,
            experiment_root=str(root),
        ),
    ]

    results = {}
    for config in configs:
        if config.algo == "value_iteration":
            csv_path = train_value_iteration(config)
        elif config.algo == "monte_carlo":
            csv_path = train_monte_carlo_control(config)
        elif config.algo == "sarsa":
            csv_path = train_sarsa(config)
        elif config.algo == "q_learning":
            csv_path = train_q_learning(config)
        elif config.algo == "dqn":
            csv_path = train_dqn(config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algo}")
        results[config.tag()] = csv_path

    plot_comparison(
        {
            "Value Iteration": results["value_iteration_delay0_seed42"],
            "Monte Carlo": results["monte_carlo_delay0_seed42"],
            "SARSA": results["sarsa_delay0_seed42"],
            "Q-Learning": results["q_learning_delay0_seed42"],
            "DQN": results["dqn_delay0_seed42"],
        },
        root / "comparison_discrete_no_delay.png",
        "Discrete Pendulum Comparison (No Delay)",
        value_key="smoothed_true_reward",
    )

    plot_comparison(
        {
            "SARSA delay=0": results["sarsa_delay0_seed42"],
            "SARSA delay=5": results["sarsa_delay5_seed42"],
            "Q-Learning delay=0": results["q_learning_delay0_seed42"],
            "Q-Learning delay=5": results["q_learning_delay5_seed42"],
            "DQN delay=0": results["dqn_delay0_seed42"],
            "DQN delay=5": results["dqn_delay5_seed42"],
        },
        root / "comparison_discrete_delay.png",
        "Discrete Pendulum Delay Comparison",
        value_key="smoothed_true_reward",
    )

    summary_rows = []
    for name, csv_path in results.items():
        row = summarize_csv(csv_path)
        row["experiment"] = name
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(root / "summary_table.csv", index=False)
    print(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()
