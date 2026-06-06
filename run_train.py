from ddpg_bipedal.cli import parse_train_config
from ddpg_bipedal.trainer import DDPGTrainer


def main():
    config = parse_train_config()
    DDPGTrainer(config).train()


if __name__ == "__main__":
    main()
