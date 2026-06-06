from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class DDPGConfig:
    env_name: str = "BipedalWalker-v3"
    seed: int = 42
    max_episodes: int = 600
    max_steps: int = 1600
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4  # critic 学习率从 1e-3 降到 1e-4，防止 Q 值发散
    batch_size: int = 256  # 增大批大小使梯度更稳定
    buffer_size: int = 200000
    warmup_steps: int = 5000
    update_after: int = 2000
    update_every: int = 1  # 每个 step 都更新，更细粒度
    update_iters: int = 1  # 每次只更新 1 步，避免过拟合
    noise_sigma: float = 0.20  # OU 噪声 sigma
    policy_noise: float = 0.20  # 目标策略平滑噪声标准差
    noise_clip: float = 0.50  # 噪声裁剪范围
    eval_every: int = 20
    eval_episodes: int = 5
    solve_score: float = 300.0
    smooth_window: int = 10
    experiment_dir: str = "ddpg_bipedal/runs/default"
    checkpoint_dir: str = "checkpoints"
    figure_name: str = "training_curve.png"
    metrics_name: str = "metrics.csv"
    summary_name: str = "summary.json"

    def experiment_path(self) -> Path:
        return Path(self.experiment_dir)

    def checkpoint_path(self) -> Path:
        return self.experiment_path() / self.checkpoint_dir

    def figure_path(self) -> Path:
        return self.experiment_path() / self.figure_name

    def metrics_path(self) -> Path:
        return self.experiment_path() / self.metrics_name

    def summary_path(self) -> Path:
        return self.experiment_path() / self.summary_name

    def save(self):
        self.experiment_path().mkdir(parents=True, exist_ok=True)
        self.summary_path().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
