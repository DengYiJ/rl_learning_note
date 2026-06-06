from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(rewards, smoothed_rewards, eval_points, eval_rewards, figure_path: Path):
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="raw train reward", alpha=0.35, color="#7f8c8d")
    plt.plot(smoothed_rewards, label="smoothed train reward", linewidth=2.0, color="#1f77b4")
    if eval_points:
        plt.plot(eval_points, eval_rewards, marker="o", label="eval reward", color="#d62728")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDPG on BipedalWalker")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()
