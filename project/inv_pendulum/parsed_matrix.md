# 📊 倒立摆强化学习实验解析矩阵

## 1. 实验目标
| 项目 | 内容 |
|------|------|
| 环境 | Pendulum-v1 (Gymnasium) |
| 算法 | Actor-Critic, DDPG |
| 延迟条件 | delay=0（无延迟）, delay=5（5步延迟） |
| 目标 | 达到与在线教程相同结果，分析延迟影响 |
| 参照 | Keras DDPG on Pendulum-v1 教程 |

## 2. 实验步骤
1. 实现 DelayedRewardPendulum 环境封装（envs.py）
2. 实现 Actor-Critic（actor_critic.py）和 DDPG（ddpg.py）
3. 分别以 delay=0 和 delay=5 运行两组实验
4. 输出 metrics.csv、training_curve.png、summary.json

## 3. 核心参数
| 参数 | Actor-Critic | DDPG |
|------|:-----------:|:----:|
| episodes | 120 | 100 |
| max_steps | 200 | 200 |
| actor_lr | 3e-4 | 1e-3 |
| critic_lr | 1e-3 | 2e-3 |
| hidden_dim | 128 | 128 |
| gamma | 0.99 | 0.99 |
| batch_size | — | 64 |
| buffer_size | — | 100000 |
| noise_sigma | — | 0.15 |

## 4. Reward 机制
**真实 reward（Pendulum-v1 内置）：**
$$r_{\text{true}} = -(\theta^2 + 0.1 \cdot \dot{\theta}^2 + 0.001 \cdot \tau^2)$$

**延迟 reward（delay=5）：**
- 维护 FIFO 队列，第 $t$ 步返回 $r(s_{t-5}, a_{t-5})$
- 前 5 步因历史不足返回 0
- 环境动力学不受影响，仅学习信号滞后

## 5. 实验结果

| 实验 | 最终 Episodic Reward | 最终 Smoothed True Reward | 最佳 True Reward | 收敛？ |
|------|:-------------------:|:------------------------:|:----------------:|:------:|
| Actor-Critic delay=0 | -1526.39 | -1240.14 | -1170.03 | ❌ |
| Actor-Critic delay=5 | -1485.13 | -1261.85 | -1173.58 | ❌ |
| **DDPG delay=0** | **-119.04** | **-118.36** | **-118.27** | **✅** |
| **DDPG delay=5** | **-124.53** | **-125.86** | **-123.13** | **✅** |

## 6. 分析结论
1. DDPG delay=0 收敛到 -118.36，与 Keras 教程一致
2. DDPG delay=5 收敛速度减慢约 50%（第53ep vs 第31ep），最终性能下降约 6%
3. Actor-Critic 两种延迟下均未收敛
4. DDPG 的 off-policy（经验回放+目标网络）对延迟天然鲁棒