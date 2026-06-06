from pathlib import Path

from inverted_pendulum_rl.plotting import plot_comparison


def main():
    continuous_root = Path("inverted_pendulum_rl/runs")
    discrete_root = Path("inverted_pendulum_rl/discrete_runs")
    output_path = Path("inverted_pendulum_rl") / "all_algorithms_no_delay.png"

    csv_paths = {
        "DDPG": continuous_root / "ddpg_delay0_seed42" / "metrics.csv",
        "Actor-Critic": continuous_root / "actor_critic_delay0_seed42" / "metrics.csv",
        "DQN": discrete_root / "dqn_delay0_seed42" / "metrics.csv",
        "SARSA": discrete_root / "sarsa_delay0_seed42" / "metrics.csv",
        "Q-Learning": discrete_root / "q_learning_delay0_seed42" / "metrics.csv",
        "Monte Carlo": discrete_root / "monte_carlo_delay0_seed42" / "metrics.csv",
        "Value Iteration": discrete_root / "value_iteration_delay0_seed42" / "metrics.csv",
    }

    missing = [str(path) for path in csv_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing experiment outputs. Please run both experiment suites first:\n"
            "python -m inverted_pendulum_rl.run_experiments\n"
            "python -m inverted_pendulum_rl.run_discrete_experiments\n"
            f"\nMissing files:\n" + "\n".join(missing)
        )

    plot_comparison(
        csv_paths,
        output_path,
        "Pendulum-v1 All Algorithms Comparison (No Delay)",
        value_key="smoothed_true_reward",
    )
    print(output_path)


if __name__ == "__main__":
    main()
