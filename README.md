大作业：根据给定题目（参见群中列表）开展实验，并完成报告（6页
+），报告中应包含：
⚫问题的详细描述，包括环境，行为，状态，约束等；
⚫方法描述，要针对具体问题进行方法描述；
⚫环境、神经网络及仿真设置，包括环境参数，每个神经网络的输入、
输出及超参数（层数、神经元类型）、仿真参数等；
⚫仿真结果，至少要包含累计回报随训练episodes变化的曲线（原
始曲线和平滑后曲线）
⚫结果分析

题目：
ddpg-bipedal 实验以 OpenAI Gym 中的双足机器人行走任务
（BipedalWalker）为背景，实验深度强化学习在连续动作控制问题
中的应用。
 实验可采用任意算法（如DDPG等），结合 Actor-Critic 结构、经验
回放和目标网络，实现智能体对机器人行走策略的学习。
 通过本实验，掌握连续动作空间下强化学习模型的基本训练流程，并
理解网络结构、探索噪声和参数设置对控制效果的影响。

# DDPG-BipedalWalker

这个目录现在按“配置、模型、训练器、评估、作图、入口脚本”拆分，方便后续调参、换网络和出实验报告。

## 目录结构

```text
ddpg_bipedal/
├── config.py              # 超参数与输出目录配置
├── env.py                 # Gym/Gymnasium 环境封装
├── models.py              # Actor / Critic 网络
├── replay_buffer.py       # 经验回放
├── noise.py               # 探索噪声
├── agent.py               # DDPG 智能体
├── evaluate.py            # 策略评估
├── plotting.py            # 作图
├── trainer.py             # 训练主流程
├── cli.py                 # 命令行参数
├── run_train.py           # 推荐训练入口
├── train_ddpg_bipedal.py  # 兼容旧入口
├── test_ddpg_bipedal.py   # 加载 actor.pt 做测试
└── report_ddpg_bipedal.tex
```

## 安装依赖

推荐使用：

```bash
pip install gymnasium[box2d] torch matplotlib numpy
```

如果你使用旧版 Gym：

```bash
pip install gym[box2d] torch matplotlib numpy
```

## 训练

推荐入口：

```bash
python ddpg_bipedal/run_train.py
```

也兼容旧入口：

```bash
python ddpg_bipedal/train_ddpg_bipedal.py
```

例如，把实验结果单独存到一个目录：

```bash
python ddpg_bipedal/run_train.py --max-episodes 600 --experiment-dir ddpg_bipedal/runs/exp1
```

## 训练输出

训练结束后会在 `experiment-dir` 下生成：

- `checkpoints/actor.pt`
- `checkpoints/critic.pt`
- `training_curve.png`
- `metrics.csv`
- `summary.json`

其中：

- `actor.pt`：测试和部署时真正用来出动作的策略网络
- `critic.pt`：训练阶段评估动作质量的价值网络
- `metrics.csv`：每个 episode 的 reward、平滑 reward、loss 等数据
- `training_curve.png`：原始 reward + 平滑 reward + eval reward 曲线

## 测试

```bash
python ddpg_bipedal/test_ddpg_bipedal.py --actor-path ddpg_bipedal/runs/exp1/checkpoints/actor.pt --episodes 5
```

如果环境支持并且本地配置正常，也可以尝试渲染：

```bash
python ddpg_bipedal/test_ddpg_bipedal.py --actor-path ddpg_bipedal/runs/exp1/checkpoints/actor.pt --episodes 3 --render
```
