# 倒立摆强化学习项目

本项目围绕 `Gymnasium` 的 `Pendulum-v1` 环境，实现两种连续控制算法：

- `Actor-Critic`：随机高斯策略 + 状态价值函数
- `DDPG`：确定性策略 + 经验回放 + 目标网络

同时新增一条离散化倒立摆实验链，用于：

- `Value Iteration`（动态规划）
- `Monte Carlo Control`
- `SARSA`
- `Q-Learning`
- `DQN`

项目同时支持两类 reward 设置：

- 正常 reward：直接使用环境返回的 reward
- 延迟 reward：在第 `t` 步，只把 `t-5` 时刻的状态和输入对应的 reward 返回给智能体

## 环境 reward

`Pendulum-v1` 的即时 reward 为：

```text
r_t = -(theta_t^2 + 0.1 * theta_dot_t^2 + 0.001 * u_t^2)
```

其中：

- `theta_t` 是摆杆相对竖直向上的偏角
- `theta_dot_t` 是角速度
- `u_t` 是电机力矩，范围 `[-2, 2]`

因此 reward 越接近 `0` 越好；越负表示偏离竖直位置越远、角速度越大或控制力矩越大。

## 延迟 reward 设定

若存在 5 个采样周期的延迟，则训练时第 `t` 步收到的是：

```text
r_t^(delay) = r(s_{t-5}, u_{t-5})
```

前 5 步由于没有足够历史，返回 `0`。环境真实运动过程不变，变化的是智能体在学习时看到的 reward 信号。

## 运行方式

```bash
python -m inverted_pendulum_rl.run_experiments
```

连续控制结果输出到：

```text
inverted_pendulum_rl/runs/
```

离散化倒立摆对比运行：

```bash
python -m inverted_pendulum_rl.run_discrete_experiments
```

将连续控制和离散化算法汇总到同一张图：

```bash
python -m inverted_pendulum_rl.plot_all_algorithms
```

离散化结果输出到：

```text
inverted_pendulum_rl/discrete_runs/
```

建议依赖：

```bash
pip install gymnasium torch matplotlib pandas numpy
```

连续控制主要文件：

- `*/metrics.csv`：每回合 reward 记录
- `*/training_curve.png`：单次实验训练曲线
- `comparison_true_reward.png`：四组实验综合对比图
- `summary_table.csv`：汇总表

离散化倒立摆主要文件：

- `comparison_discrete_no_delay.png`：DP / MC / SARSA / Q-learning / DQN 对比
- `comparison_discrete_delay.png`：延迟 reward 对 SARSA / Q-learning / DQN 的影响
- `summary_table.csv`：离散化实验汇总表

总对比图：

- `all_algorithms_no_delay.png`：DDPG / DQN / SARSA / Q-learning / Monte Carlo / Value Iteration / Actor-Critic 同图对比

## 离散化说明

由于 `Pendulum-v1` 原本是连续状态、连续动作环境，而动态规划、SARSA、Q-learning 等经典算法通常假设有限状态动作空间，因此这里做了统一离散化：

- `theta` 离散为 15 个区间
- `theta_dot` 离散为 15 个区间
- 力矩 `u` 离散为 9 个动作

这使得所有离散算法仍然围绕“同一个倒立摆对象”进行比较，只是控制精度会低于连续控制算法。
