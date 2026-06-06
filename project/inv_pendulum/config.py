from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    algo: str
    reward_delay: int = 0
    env_name: str = "Pendulum-v1"
    seed: int = 42
    episodes: int = 120
    max_steps: int = 200
    gamma: float = 0.99
    hidden_dim: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    smooth_window: int = 10
    eval_episodes: int = 5
    experiment_root: str = "inverted_pendulum_rl/runs"

    # DDPG only
    tau: float = 0.005
    batch_size: int = 64
    buffer_size: int = 100000
    warmup_steps: int = 1000
    update_after: int = 1000
    update_every: int = 1
    noise_sigma: float = 0.15

    # Actor-Critic only
    policy_std_min: float = -5.0
    policy_std_max: float = 1.0
    entropy_coef: float = 1e-3

    # Discrete pendulum only
    theta_bins: int = 15
    theta_dot_bins: int = 15
    action_bins: int = 9
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.02
    mc_eval_episodes: int = 20

    def tag(self) -> str:
        delay_tag = f"delay{self.reward_delay}"
        return f"{self.algo}_{delay_tag}_seed{self.seed}"

    def output_dir(self) -> Path:
        return Path(self.experiment_root) / self.tag()
