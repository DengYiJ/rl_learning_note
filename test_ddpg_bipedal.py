import argparse
from pathlib import Path

import numpy as np
import torch

from ddpg_bipedal.env import GYM, IS_GYMNASIUM, make_env, reset_env, step_env
from ddpg_bipedal.models import Actor


def evaluate(actor_path: Path, env_name: str, episodes: int, seed: int, render: bool):
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")

    if render:
        env = GYM.make(env_name, render_mode="human")
        if IS_GYMNASIUM:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        else:
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
    else:
        env = make_env(env_name, seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, action_limit).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    rewards = []
    for episode in range(1, episodes + 1):
        state = reset_env(env, seed=seed + episode)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]
            action = np.clip(action, -action_limit, action_limit)
            state, reward, done, _ = step_env(env, action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)
        print(f"Episode {episode:02d} | steps={steps:4d} | reward={episode_reward:8.2f}")

    env.close()
    print(f"Average reward over {episodes} episodes: {np.mean(rewards):.2f}")
    print(f"Best reward: {np.max(rewards):.2f}")
    print(f"Worst reward: {np.min(rewards):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained DDPG actor on BipedalWalker.")
    parser.add_argument("--actor-path", type=str, default="checkpoints/actor.pt")
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v3")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        actor_path=Path(args.actor_path),
        env_name=args.env_name,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
    )
