from ddpg_bipedal.agent import DDPGAgent
from ddpg_bipedal.cli import parse_train_config
from ddpg_bipedal.config import DDPGConfig
from ddpg_bipedal.env import GYM, IS_GYMNASIUM, load_gym_backend, make_env, reset_env, step_env
from ddpg_bipedal.evaluate import evaluate_policy
from ddpg_bipedal.models import Actor, Critic
from ddpg_bipedal.noise import GaussianNoise
from ddpg_bipedal.plotting import plot_training_curves
from ddpg_bipedal.replay_buffer import ReplayBuffer
from ddpg_bipedal.trainer import DDPGTrainer
from ddpg_bipedal.utils import moving_average, set_seed


def train(config: DDPGConfig):
    DDPGTrainer(config).train()


if __name__ == "__main__":
    train(parse_train_config())
