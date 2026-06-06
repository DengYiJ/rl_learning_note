import argparse

from ddpg_bipedal.config import DDPGConfig


def build_train_parser():
    parser = argparse.ArgumentParser(description="Train DDPG on BipedalWalker.")
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--update-after", type=int, default=2000)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--update-iters", type=int, default=1)
    parser.add_argument("--noise-sigma", type=float, default=0.20)
    parser.add_argument("--policy-noise", type=float, default=0.20)
    parser.add_argument("--noise-clip", type=float, default=0.50)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--solve-score", type=float, default=300.0)
    parser.add_argument("--smooth-window", type=int, default=10)
    parser.add_argument("--experiment-dir", type=str, default="ddpg_bipedal/runs/default")
    return parser


def parse_train_config():
    parser = build_train_parser()
    args = parser.parse_args()
    return DDPGConfig(**vars(args))
