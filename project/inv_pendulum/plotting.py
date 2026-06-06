from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(csv_path: Path, output_path: Path, title: str):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["episode"], df["episodic_reward"], label="reward", alpha=0.45)
    axes[0].plot(df["episode"], df["smoothed_reward"], label="smoothed", linewidth=2.0)
    axes[0].set_ylabel("Delayed Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df["episode"], df["true_reward"], label="true_reward", alpha=0.45)
    axes[1].plot(df["episode"], df["smoothed_true_reward"], label="smoothed_true_reward", linewidth=2.0)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Environment Reward")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_comparison(csv_paths: dict[str, Path], output_path: Path, title: str, value_key: str = "smoothed_true_reward"):
    plt.figure(figsize=(10, 6))
    for label, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        plt.plot(df["episode"], df[value_key], label=label, linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel(value_key)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
