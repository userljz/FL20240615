import hydra
from core.runner import train_fl


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def start_training(config):
    train_fl(config)
    return


if __name__ == "__main__":
    start_training()
    