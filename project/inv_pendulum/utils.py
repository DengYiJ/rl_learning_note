from pathlib import Path
import csv
import json
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    result = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        result.append(float(np.mean(values[left : i + 1])))
    return result


def write_metrics(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
