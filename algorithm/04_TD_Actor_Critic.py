import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical


# =====================================
# Actor
# =====================================

class Actor(nn.Module):

    def __init__(self,state_dim,action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):

        return self.net(x)


# =====================================
# Critic
# =====================================

class Critic(nn.Module):

    def __init__(self,state_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),

            nn.Linear(128,1)
        )

    def forward(self,x):

        return self.net(x)


# =====================================
# Agent
# =====================================

class TDActorCritic:

    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            actor_lr=1e-3,
            critic_lr=1e-3):

        self.gamma = gamma

        self.actor = Actor(
            state_dim,
            action_dim
        )

        self.critic = Critic(
            state_dim
        )

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

    # ---------------------------------
    # 采样动作
    # ---------------------------------

    def choose_action(self,state):

        state = torch.FloatTensor(state)

        probs = self.actor(state)

        dist = Categorical(probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    # ---------------------------------
    # TD更新
    # ---------------------------------

    def update(
            self,
            state,
            reward,
            next_state,
            done,
            log_prob):

        state = torch.FloatTensor(state)

        next_state = torch.FloatTensor(next_state)

        # ==========================
        # V(s)
        # ==========================

        value = self.critic(state)

        # ==========================
        # V(s')
        # ==========================

        with torch.no_grad():

            next_value = self.critic(next_state)

        # ==========================
        # TD Target
        # ==========================

        if done:

            td_target = torch.tensor(
                [reward],
                dtype=torch.float32
            )

        else:

            td_target = (
                reward
                + self.gamma * next_value
            )

        # ==========================
        # TD Error
        #
        # δ =
        # r + γV(s')
        # - V(s)
        # ==========================

        td_error = td_target - value

        # ==========================
        # Actor
        #
        # Advantage≈TD Error
        # ==========================

        actor_loss = (
            -log_prob
            * td_error.detach()
        )

        # ==========================
        # Critic
        # ==========================

        critic_loss = td_error.pow(2)

        # ==========================
        # 更新Actor
        # ==========================

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # ==========================
        # 更新Critic
        # ==========================

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()


# =====================================
# Training
# =====================================

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]

action_dim = env.action_space.n

agent = TDActorCritic(
    state_dim,
    action_dim
)

episodes = 1000

for episode in range(episodes):

    state,_ = env.reset()

    episode_reward = 0

    done = False

    while not done:

        action, log_prob = agent.choose_action(
            state
        )

        next_state,reward,terminated,truncated,_ = env.step(
            action
        )

        done = terminated or truncated

        agent.update(
            state,
            reward,
            next_state,
            done,
            log_prob
        )

        state = next_state

        episode_reward += reward

    if episode % 10 == 0:

        print(
            f"Episode={episode}, "
            f"Reward={episode_reward}"
        )

env.close()
