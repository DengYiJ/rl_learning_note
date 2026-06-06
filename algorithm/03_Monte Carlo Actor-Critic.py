```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Categorical
# =====================================================
       #    ┌────────────┐
       #    │   Actor    │
       #    │ π(a|s)     │
       #    └─────┬──────┘
       #          │
       #        action
       #          │
       #          ▼
       #       Environment
       #          │
       #          ▼
       #       reward
       #          │
       #          ▼
       #    ┌────────────┐
       #    │   Critic   │
       #    │   V(s)     │
       #    └─────┬──────┘
       #          │
       #          ▼
       # Advantage = G - V(s)
       #          │
       #          ▼
       #      更新Actor
# =====================================================
# =====================================================
# Actor Network
#
# 输出策略 π(a|s)
# =====================================================

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),

            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# Critic Network
#
# 输出状态价值:
#
# V(s)
#
# 表示:
# 从当前状态开始
# 未来大概还能获得多少累计奖励
#
# =====================================================

class Critic(nn.Module):

    def __init__(self, state_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# Actor-Critic Agent
# =====================================================

class ActorCritic:

    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            lr_actor=1e-3,
            lr_critic=1e-3):

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
            lr=lr_actor
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=lr_critic
        )

    # -------------------------------------------------
    # 根据策略采样动作
    # -------------------------------------------------
    def choose_action(self, state):

        state = torch.FloatTensor(state)

        probs = self.actor(state)

        dist = Categorical(probs)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    # -------------------------------------------------
    # 更新Actor与Critic
    # -------------------------------------------------
    def update(
            self,
            states,
            rewards,
            log_probs):

        # ============================================
        # Step1
        # 计算每个时刻的G_t
        #
        # G_t =
        # r_t
        # + γr_(t+1)
        # + γ²r_(t+2)
        # + ...
        # ============================================

        returns = []

        G = 0

        for r in reversed(rewards):

            G = r + self.gamma * G

            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        # ============================================
        # Step2
        # 计算Critic预测的V(s)
        # ============================================

        states = torch.FloatTensor(states)

        values = self.critic(states).squeeze()

        #
        # values:
        #
        # [V(s0),V(s1),V(s2)...]
        #
        # returns:
        #
        # [G0,G1,G2...]
        #

        # ============================================
        # Step3
        # Advantage
        #
        # A = G - V(s)
        #
        # Critic预测:
        # 当前状态未来值多少
        #
        # 实际结果:
        # G
        #
        # 二者差值就是优势
        #
        # ============================================

        advantages = returns - values.detach()

        #
        # detach()非常重要
        #
        # Actor更新时
        # 不希望梯度传回Critic
        #

        # ============================================
        # Step4
        # Actor Loss
        #
        # L_actor
        #
        # = -A logπ
        #
        # ============================================

        actor_losses = []

        for log_prob, adv in zip(
                log_probs,
                advantages):

            actor_losses.append(
                -log_prob * adv
            )

        actor_loss = torch.stack(
            actor_losses
        ).sum()

        # ============================================
        # Step5
        # Critic Loss
        #
        # 均方误差
        #
        # L_critic
        # =
        # (V-G)^2
        #
        # ============================================

        critic_loss = nn.MSELoss()(
            values,
            returns
        )

        # ============================================
        # Step6
        # 更新Actor
        # ============================================

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # ============================================
        # Step7
        # 更新Critic
        # ============================================

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()


# =====================================================
# Training
# =====================================================

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = ActorCritic(
    state_dim,
    action_dim
)

episodes = 1000

for episode in range(episodes):

    state, _ = env.reset()

    states = []
    rewards = []
    log_probs = []

    done = False

    while not done:

        action, log_prob = agent.choose_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        states.append(state)

        rewards.append(reward)

        log_probs.append(log_prob)

        state = next_state

    agent.update(
        states,
        rewards,
        log_probs
    )

    if episode % 10 == 0:

        print(
            f"Episode={episode}, "
            f"Reward={sum(rewards)}"
        )

env.close()
```
