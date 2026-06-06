from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ddpg_bipedal.models import Actor, Critic
from ddpg_bipedal.noise import OrnsteinUhlenbeckNoise


class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_limit, config, device: torch.device):
        self.device = device
        self.gamma = config.gamma
        self.tau = config.tau
        self.action_limit = action_limit
        self.noise_clip = config.noise_clip
        self.policy_noise = config.policy_noise

        self.actor = Actor(state_dim, action_dim, action_limit).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_limit).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=config.noise_sigma)

    def select_action(self, state: np.ndarray, add_noise: bool):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if add_noise:
            action = action + self.noise.sample()
        return np.clip(action, -self.action_limit, self.action_limit)

    def reset_noise(self):
        self.noise.reset()

    def update(self, replay_buffer, batch_size: int):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            # 目标策略平滑（Target Policy Smoothing）—— 给目标动作加噪声，改善 Q 函数平滑性
            next_actions = self.actor_target(next_states)
            noise = torch.normal(0.0, self.policy_noise, size=next_actions.shape, device=self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -self.action_limit, self.action_limit)

            next_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        return critic_loss.item(), actor_loss.item()

    def soft_update(self, source: nn.Module, target: nn.Module):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, checkpoint_dir: Path):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), checkpoint_dir / "actor.pt")
        torch.save(self.critic.state_dict(), checkpoint_dir / "critic.pt")

    def load_actor(self, actor_path: Path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor.eval()
