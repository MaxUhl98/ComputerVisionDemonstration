from demo_configuration import DemonstrationConfig
from classification import demonstrate, train_model


def train_and_demonstrate() -> None:
    """Trains models using the configuration and demonstrates the results after training

    :return: None (writes training information to the used model logfile, saves models in the model folder.
    Also visualizes results in a confusion matrix and via showing labeled samples)
    """
    cfg = DemonstrationConfig()
    train_model(cfg)
    demonstrate(cfg)


def run_demonstration() -> None:
    """Runs the demonstration with the existing configured models.

    :return: None (Visualizes results in a confusion matrix and via showing labeled samples)
    """
    cfg = DemonstrationConfig()
    demonstrate(cfg)


if __name__ == '__main__':
    run_demonstration()
