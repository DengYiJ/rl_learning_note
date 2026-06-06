# 倒立摆对象上的 Actor-Critic 与 DDPG 实践

## 1. 实验对象与目标

本实验选择 `Pendulum-v1` 作为“倒立摆对象”。该环境的目标是让摆杆稳定在竖直向上的平衡位置，并尽量减少角速度和控制力矩。

我们实现并比较两种连续控制算法：

- Actor-Critic
- DDPG

同时新增一组离散化倒立摆对比算法：

- Value Iteration（动态规划）
- Monte Carlo Control
- SARSA
- Q-Learning
- DQN

并继续考察 reward 延迟 5 个采样周期时的影响。

## 2. reward 如何取

环境 reward 来自 `Gymnasium` 的 `Pendulum-v1` 定义：

```python
costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
reward = -costs
```

解释如下：

- 偏角项 `theta^2`：权重最大，说明“尽快摆正”是主目标
- 角速度项 `0.1 * theta_dot^2`：抑制过快摆动
- 控制项 `0.001 * u^2`：限制控制能量，避免用过大的力矩硬拉

因此：

- 最优情况是 `theta=0, theta_dot=0, u=0`，reward 接近 `0`
- 如果摆杆偏离竖直很多，reward 会显著变负

## 3. 延迟 reward 的实现

题目要求 reward 计算时只能使用 5 步之前的状态和输入，因此我们实现：

```text
r_t^(delay) = r(s_{t-5}, u_{t-5})
```

这意味着：

- 环境本身的动力学没有延迟
- 但学习信号滞后了 5 个采样周期
- 智能体更难把“当前动作”与“当前后果”正确对应起来

## 4. 与在线教程的对照基准

这里把在线教程默认理解为 Keras 官方 `DDPG on Pendulum-v1` 教程：

- 教程链接：<https://keras.io/examples/rl/ddpg_pendulum/>
- 教程中 100 回合训练后，最近阶段平均 episodic reward 大致收敛在 `-130` 到 `-180` 一带，并且整体趋势是从 `-1500` 左右逐步提升到接近 `0` 的方向

这个基准适合用于验证我们的 DDPG 实现是否“跑通且趋势一致”。

## 5. 为什么要做离散化

`Pendulum-v1` 是连续状态、连续动作问题。

- DDPG 和连续 Actor-Critic 可以直接处理它
- 但动态规划、蒙特卡洛控制、SARSA、Q-learning 这类经典算法通常需要有限状态和有限动作

因此本项目增加了统一的离散化倒立摆：

- 将角度 `theta` 离散成 15 个区间
- 将角速度 `theta_dot` 离散成 15 个区间
- 将动作力矩 `u` 离散成 9 个档位

这样做的意义是：

- 所有离散算法仍然围绕同一个倒立摆任务
- 但因为动作与状态分辨率下降，性能通常会弱于连续控制算法

## 6. 连续控制部分的本次实测结果

运行 `python -m inverted_pendulum_rl.run_experiments` 后，得到四组结果：

- Actor-Critic, delay=0
- Actor-Critic, delay=5
- DDPG, delay=0
- DDPG, delay=5

请重点观察：

- `inverted_pendulum_rl/runs/comparison_true_reward.png`
- `inverted_pendulum_rl/runs/summary_table.csv`

本次实验的关键数值如下：

| 实验 | 最优平滑真实回报 | 最终平滑真实回报 |
|---|---:|---:|
| Actor-Critic, delay=0 | -1170.03 | -1240.14 |
| Actor-Critic, delay=5 | -1173.58 | -1261.85 |
| DDPG, delay=0 | -118.27 | -118.36 |
| DDPG, delay=5 | -123.13 | -125.86 |

可以看到：

- DDPG 明显优于基础 Actor-Critic
- DDPG 无延迟时最终平滑真实回报约为 `-118.36`
- DDPG 5 步延迟时最终平滑真实回报约为 `-125.86`
- Actor-Critic 在这组超参数下没有学到稳定控制，始终停留在 `-1200` 左右

## 7. 与在线教程结果的比较

如果以 Keras 官方 `DDPG on Pendulum-v1` 教程作为在线教程基准，那么它的典型现象是：

- `DDPG, delay=0` 的学习速度更快，结果更接近在线教程
- reward 从大约 `-1500` 左右开始
- 经过训练后提升到 `-200` 甚至更接近 `0` 的区间
- 曲线整体单调改善，但中间会有波动

本次 DDPG 无延迟实验的实际结果是：

- 第 10 回合约 `-1489.58`
- 第 20 回合约 `-743.10`
- 第 30 回合约 `-376.78`
- 第 40 回合约 `-120.95`
- 第 100 回合约 `-119.04`

这和在线教程的趋势是高度一致的，而且最终效果甚至更好一些，说明本项目里的 DDPG 实现已经正确跑通。

另一方面，基础 Actor-Critic 没有达到同等级效果，原因主要是：

- 它没有经验回放，样本利用率更低
- 它直接用一步 TD 误差更新随机策略，梯度噪声更大
- 对连续动作倒立摆这类任务，基础 Actor-Critic 的训练稳定性通常不如 DDPG

## 8. 离散化算法的本次对比结果

离散化实验结果保存在：

- `inverted_pendulum_rl/discrete_runs/comparison_discrete_no_delay.png`
- `inverted_pendulum_rl/discrete_runs/comparison_discrete_delay.png`
- `inverted_pendulum_rl/discrete_runs/summary_table.csv`

无延迟时，本次实测的最终平滑真实回报约为：

| 算法 | 最终平滑真实回报 |
|---|---:|
| Value Iteration | -1197.11 |
| Monte Carlo | -1184.66 |
| SARSA | -1128.08 |
| Q-Learning | -1217.77 |
| DQN | -118.93 |

可以看到：

- `DQN` 远强于其他离散算法
- `SARSA` 在表格法里相对最好
- `Q-Learning` 和 `Monte Carlo` 可以学到部分控制能力，但波动明显
- `Value Iteration` 在粗离散模型上只能得到一个较粗糙的近似策略

这里 `Value Iteration` 表现不好的主要原因不是动态规划“理论上不行”，而是：

- 我们对连续倒立摆做了有限网格近似
- 状态和动作都被压缩到较粗的离散分辨率
- 因而它优化的是“近似模型”，不是原始连续系统

## 9. 离散化算法中 5 步延迟 reward 的影响

离散分支中，考虑延迟 reward 的算法包括：

- SARSA
- Q-Learning
- DQN

这里没有把 Value Iteration 和 Monte Carlo 放进“延迟 5 步”的主对比，原因是：

- 对 Value Iteration 来说，5 步延迟 reward 会破坏原始状态的 Markov 性
- 如果要严格做 DP，需要把过去 5 步状态和动作一起并入扩展状态，状态空间会急剧膨胀
- 因此本项目把动态规划保留为“无延迟基准”

本次实测最终平滑真实回报为：

| 算法 | delay=0 | delay=5 |
|---|---:|---:|
| SARSA | -1128.08 | -1144.19 |
| Q-Learning | -1217.77 | -1255.14 |
| DQN | -118.93 | -121.54 |

可以看到：

- `DQN` 受延迟影响最小，几乎仍能维持接近最优的控制效果
- `SARSA` 和 `Q-Learning` 在延迟后均出现退化
- 对表格方法来说，延迟 reward 会进一步加剧样本低效和更新噪声

## 10. 5 步延迟 reward 对连续控制算法的影响

从本次曲线可以看到，延迟 reward 的影响并不是“完全学不会”，而是：

- DDPG 前期明显变慢
- Actor-Critic 原本就不稳定，加入延迟后进一步恶化

更具体地说：

- `DDPG, delay=0` 在第 40 回合附近已经达到 `-120` 左右
- `DDPG, delay=5` 在前 50 回合仍长期停留在 `-1400` 到 `-1600`
- 但它在大约第 60 回合后突然追上，最终仍收敛到 `-125` 左右

这说明：

- 5 步延迟显著拖慢了 DDPG 的学习起步
- 但 DDPG 由于有 replay buffer 和 target network，仍具备一定抗延迟能力
- 最终性能相比无延迟略有下降，但下降幅度不算特别大

对于 Actor-Critic：

- 无延迟最终平滑真实回报 `-1240.14`
- 5 步延迟最终平滑真实回报 `-1261.85`

两者都比较差，说明这个基础版本 Actor-Critic 在该任务上没有形成高质量策略，因此延迟带来的额外退化被“算法本身的高波动”部分掩盖了。

## 11. 为什么延迟会带来性能下降

原因在于 credit assignment（时序归因）变难了。

无延迟时：

- 当前动作产生的效果，几乎立刻通过 reward 反馈回来

有 5 步延迟时：

- 当前动作的“好坏”要过 5 步才通过 reward 表达
- 更新时更容易把错误的状态-动作对当成 reward 来源
- 结果是梯度噪声更大、收敛更慢、最终性能更低

## 12. 结论

- 对倒立摆这类连续控制问题，DDPG 比基础 Actor-Critic 更适合
- `Pendulum-v1` 的 reward 本质上是“姿态误差 + 速度误差 + 控制能量”的负代价
- 在本次实验中，DDPG 无延迟结果约 `-118`，与在线教程表现一致
- 在离散化算法中，DQN 表现最好，几乎追平连续控制 DDPG
- SARSA、Q-learning、Monte Carlo 和 Value Iteration 受离散化精度限制，整体明显弱于 DQN 和 DDPG
- 当 reward 存在 5 步延迟时，DDPG 的主要问题是前期学习明显变慢，最终性能也有小幅下降
- 对离散算法而言，延迟同样会带来退化，其中 DQN 的鲁棒性最好
- 延迟主要破坏了动作与回报之间的即时对应关系，导致训练稳定性下降、收敛速度变慢
