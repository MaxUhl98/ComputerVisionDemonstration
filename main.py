from demo_configuration import DemonstrationConfig
from classification import demonstrate, train_model
from utils import set_default_device


def train() -> None:
    """Trains models using the configuration and demonstrates the results after training

    :return: None (writes training information to the used model logfile, saves models in the model folder.
    Also visualizes results in a confusion matrix and via showing labeled samples)
    """
    set_default_device()
    cfg = DemonstrationConfig()
    train_model(cfg)


def run_demonstration() -> None:
    """Runs the demonstration with the existing configured models.

    :return: None (Visualizes results in a confusion matrix and shows labeled samples)
    """
    set_default_device()
    cfg = DemonstrationConfig()
    demonstrate(cfg)


def main() -> None:
    """Trains a model and visualizes the results afterward

    :return: None (visualizes results in a confusion matrix and shows labeled samples)
    """
    set_default_device()
    cfg = DemonstrationConfig()
    train_model(cfg)
    demonstrate(cfg)


if __name__ == '__main__':
    main()
