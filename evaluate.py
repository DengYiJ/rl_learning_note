import numpy as np

from ddpg_bipedal.env import reset_env, step_env


def evaluate_policy(env, agent, episodes: int, seed_offset: int = 0):
    rewards = []
    for episode in range(episodes):
        state = reset_env(env, seed=seed_offset + episode)
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, add_noise=False)
            state, reward, done, _ = step_env(env, action)
            total_reward += reward
        rewards.append(total_reward)
    return float(np.mean(rewards))
