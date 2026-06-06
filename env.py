import numpy as np


def load_gym_backend():
    try:
        import gymnasium as gym  # type: ignore

        return gym, True
    except ImportError:
        import gym  # type: ignore

        return gym, False


GYM, IS_GYMNASIUM = load_gym_backend()


def seed_env_spaces(env, seed: int):
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def make_env(env_name: str, seed: int, render: bool = False):
    if render and IS_GYMNASIUM:
        env = GYM.make(env_name, render_mode="human")
    else:
        env = GYM.make(env_name)

    if IS_GYMNASIUM:
        env.reset(seed=seed)
    else:
        env.seed(seed)
    seed_env_spaces(env, seed)
    return env


def reset_env(env, seed=None):
    if IS_GYMNASIUM:
        state, _ = env.reset(seed=seed)
    else:
        if seed is not None:
            env.seed(seed)
        state = env.reset()
    return np.asarray(state, dtype=np.float32)


def step_env(env, action):
    result = env.step(action)
    if IS_GYMNASIUM:
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        next_state, reward, done, info = result
    return np.asarray(next_state, dtype=np.float32), float(reward), bool(done), info
