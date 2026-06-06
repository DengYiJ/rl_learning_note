import random
from pathlib import Path
import csv

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def moving_average(values, window: int):
    if not values:
        return []
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        smoothed.append(float(np.mean(values[start : idx + 1])))
    return smoothed


def write_metrics_csv(path: Path, rows):
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "train_reward",
                "smoothed_reward",
                "eval_reward",
                "actor_loss",
                "critic_loss",
                "buffer_size",
                "total_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
