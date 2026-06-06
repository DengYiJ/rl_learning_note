```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# 离散概率分布
from torch.distributions import Categorical

# =====================================================
# Actor
#  ↓
# π(a|s)
#  ↓
# G
#  ↓
# 更新
# =====================================================

# =====================================================
# Policy Network
# =====================================================
#
# 输入:
#   state
#
# 输出:
#   π(a|s)
#
# CartPole:
#   action=0 -> 向左推
#   action=1 -> 向右推
#
# 输出示例:
#   [0.3, 0.7]
#
# 表示:
#   P(left)=0.3
#   P(right)=0.7
#
# =====================================================

class PolicyNet(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim),

            # 输出动作概率
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================
# REINFORCE Agent
# =====================================================

class REINFORCE:

    def __init__(
            self,
            state_dim,
            action_dim,
            lr=1e-3,
            gamma=0.99):

        self.gamma = gamma

        self.policy = PolicyNet(
            state_dim,
            action_dim
        )

        # θ 即网络参数
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr
        )

    # -------------------------------------------------
    # 根据当前策略采样动作
    #
    # 数学:
    #
    #   a_t ~ π(a|s)
    #
    # -------------------------------------------------
    def choose_action(self, state):

        state = torch.FloatTensor(state)

        # π(a|s)
        probs = self.policy(state)

        # 构造离散概率分布
        #
        # dist = distribution
        #
        # 例如:
        # probs=[0.2,0.8]
        #
        # P(a=0)=0.2
        # P(a=1)=0.8
        #
        dist = Categorical(probs)

        # 按概率随机采样
        #
        # a ~ π(a|s)
        #
        action = dist.sample()

        # REINFORCE需要:
        #
        # log π(a|s)
        #
        log_prob = dist.log_prob(action)

        # action.item():
        #
        # tensor(1)
        # ->
        # 1
        #
        # gym环境需要Python整数
        #
        return action.item(), log_prob

    # -------------------------------------------------
    # 策略更新
    # -------------------------------------------------
    def update(self, rewards, log_probs):

        # ============================================
        # Step1
        # 计算每个时刻的 G_t
        #
        # G_t =
        # r_t
        # + γr_(t+1)
        # + γ²r_(t+2)
        # + ...
        #
        # ============================================

        returns = []

        G = 0

        # 倒序计算
        #
        # rewards:
        # [r0,r1,r2,r3]
        #
        # reversed:
        # [r3,r2,r1,r0]
        #
        for r in reversed(rewards):

            # G_t = r_t + γG_(t+1)
            G = r + self.gamma * G

            # 头插法
            #
            # 最终得到:
            #
            # [G0,G1,G2,G3]
            #
            returns.insert(0, G)

        returns = torch.tensor(
            returns,
            dtype=torch.float32
        )

        # ============================================
        # Step2
        # 回报标准化
        #
        # 降低梯度方差
        # 训练更稳定
        #
        # G'=(G-mean)/std
        # ============================================

        returns = (
            returns - returns.mean()
        ) / (
            returns.std() + 1e-8
        )

        # ============================================
        # Step3
        # 构造Loss
        #
        # REINFORCE目标:
        #
        # J(θ)=E[G0]
        #
        # 希望:
        #
        # max J(θ)
        #
        # 但PyTorch只能:
        #
        # min Loss
        #
        # 因此构造:
        #
        # L =
        # -Σ G_t logπ(a_t|s_t)
        #
        # ============================================

        losses = []

        for log_prob, G in zip(log_probs, returns):

            losses.append(
                -log_prob * G
            )

        # 求和:
        #
        # L =
        # -Σ G_t logπ(a_t|s_t)
        #
        loss = torch.stack(losses).sum()

        # ============================================
        # Step4
        # 梯度清零
        # ============================================

        self.optimizer.zero_grad()

        # ============================================
        # Step5
        #
        # 自动求导
        #
        # 计算:
        #
        # ∂L/∂θ
        #
        # 即:
        #
        # -Σ G_t ∇logπ(a_t|s_t)
        #
        # ============================================

        loss.backward()

        # ============================================
        # Step6
        #
        # 更新网络参数θ
        #
        # Adam内部本质:
        #
        # θ ← θ - α∇L
        #
        # 又因为:
        #
        # L = -J
        #
        # 所以:
        #
        # θ ← θ + α∇J
        #
        # 这正是策略梯度上升
        #
        # θ ← θ + αG_t∇logπ
        #
        # ============================================

        self.optimizer.step()


# =====================================================
# Training
# =====================================================

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = REINFORCE(
    state_dim,
    action_dim
)

num_episodes = 1000

for episode in range(num_episodes):

    state, _ = env.reset()

    rewards = []
    log_probs = []

    done = False

    while not done:

        action, log_prob = agent.choose_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        rewards.append(reward)

        log_probs.append(log_prob)

        state = next_state

    # 一整个Episode结束后更新
    #
    # Monte Carlo Policy Gradient
    #
    # 必须拿到完整轨迹
    #
    agent.update(
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
