# 📊 DDPG BipedalWalker 实验解析矩阵 (Parsed Matrix)

## 1. 实验目标 (Objective)
| 项目 | 内容 |
|------|------|
| 算法 | DDPG (Deep Deterministic Policy Gradient) |
| 环境 | BipedalWalker-v3 (Gymnasium) |
| 目标 | 控制双足机器人在二维地形上稳定行走，最大化累计回报 |
| 求解标准 | 评估平均回报 ≥ 300 |

## 2. 实验步骤 (Steps)
1. 初始化环境、Actor/Critic网络、目标网络、经验回放池
2. Warm-up 阶段（前 5000 steps）：随机动作收集初始经验
3. 主循环：Actor 输出动作 + OU 噪声探索 → 环境交互 → 经验存储 → 采样更新
4. 每 20 episode 执行一次无噪声评估
5. 达到 solve_score (300) 或最大 episode 后停止

## 3. 核心参数 (Parameters)

### 环境参数
| 参数 | 值 |
|------|-----|
| env_name | BipedalWalker-v3 |
| max_episodes | 600 |
| max_steps | 1600 |
| seed | 42 |

### 算法参数
| 参数 | 值 | 说明 |
|------|-----|------|
| γ (gamma) | 0.99 | 折扣因子 |
| τ (tau) | 0.005 | 软更新系数 |
| actor_lr | 1e-4 | Actor 学习率 |
| critic_lr | 1e-4 | Critic 学习率 |
| batch_size | 256 | 训练批大小 |
| buffer_size | 200000 | 经验回放容量 |

### 网络结构
| 网络 | 层结构 | 激活函数 |
|------|--------|---------|
| Actor | 400 → 300 → action_dim | ReLU → ReLU → Tanh |
| Critic | state+action → 400 → 300 → 1 | ReLU → ReLU |

### 探索与训练参数
| 参数 | 值 | 说明 |
|------|-----|------|
| noise_sigma | 0.20 | OU 噪声标准差 |
| policy_noise | 0.20 | 目标策略平滑噪声 |
| noise_clip | 0.50 | 噪声裁剪 |
| warmup_steps | 5000 | 预热步数 |
| update_after | 2000 | 开始更新步数 |
| update_every | 1 | 每步更新 |
| update_iters | 1 | 每次更新1次 |

## 4. 实验现象 (Observations)

### 训练过程分段统计
| 阶段 | Episode | 训练回报范围 | 平滑回报趋势 | 评估回报 |
|------|---------|-------------|-------------|---------|
| 初期探索 | 1–100 | -214 ~ -83 | -138 → -118 | -156 ~ -103 |
| 挣扎期 | 101–200 | -219 ~ 21 | -138 → -139 | -136 ~ -106 |
| 首次突破 | 201–280 | -153 ~ 275 | -95 → 12 | -59 ~ 206 |
| 学习行走 | 281–380 | -153 ~ 297 | 13 → 134 | 118 ~ 293 |
| 稳定行走 | 381–480 | -113 ~ 303 | 74 → 82 | 140 ~ 293 |
| 收敛解决 | 481–560 | -127 ~ 303 | 35 → 132 | 136 ~ 303 |

### 关键里程碑
| Episode | 事件 | 具体值 |
|---------|------|--------|
| 206 | 首次正回报 | train_reward = 20.79 |
| 280 | 评估突破200 | eval_reward = 205.77 |
| 360 | 评估突破250 | eval_reward = 257.98 |
| 420 | 评估接近300 | eval_reward = 296.19 |
| 560 | 环境解决 | eval_reward = 303.49 |

### Loss 变化
| 阶段 | Actor Loss | Critic Loss | 趋势 |
|------|-----------|-------------|------|
| 初期(1-100) | 0 → -8 | 15 → 4 | Critic快速收敛 |
| 中期(101-300) | -8 → -26 | 4 → 2 | Actor持续优化 |
| 后期(301-560) | -25 → -37 | 1.5 → 1.8 | 稳定收敛 |

## 5. 发现的问题 (Issues)

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| 初次训练不收敛 | reward 始终在 -100 附近 | 高斯噪声探索不足；critic_lr=1e-3 导致 Q 值发散；update_iters=50 过拟合 | 改用 OU 噪声；critic_lr 降到 1e-4；update_iters=1；添加梯度裁剪 |
| Actor loss 爆炸 | loss 从 -6 到 -66 持续增大 | 无梯度裁剪，Critic Q 值估计发散 | 添加 clip_grad_norm_；降低 critic_lr |
| 训练回报高方差 | 相邻 episode 差异可达 300+ | BipedalWalker 环境本身随机性 + 探索噪声 | 使用平滑曲线分析趋势；每 20ep 做评估 |
| 过拟合风险 | 单步更新 50 次导致 loss 异常 | update_iters 过大 | 改为 update_iters=1 |

## 6. 结论 (Conclusions)
1. DDPG 在 BipedalWalker 上经 560 episodes 训练达到 303.49 评估回报（超过 300 阈值）
2. OU 噪声比高斯噪声更适合连续控制探索（时间相关性帮助维持动作趋势）
3. 梯度裁剪 + 降低 critic 学习率有效防止 Q 值发散
4. 更细粒度的更新（update_every=1, update_iters=1）比批量更新更稳定
5. 训练曲线呈现「探索 → 首次站起 → 学习迈步 → 稳定行走」的典型学习过程