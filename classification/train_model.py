import logging

import torch.cuda

from classification.models.EfficientNetV2 import CustomizedEfficientnetV2
from classification.models.ViT import CustomizedViT
from classification.models.MiniVGG import MiniVGG
from classification.models.ConvNeXt_V2 import CustomizedConvNeXT_V2
from demo_configuration import DemonstrationConfig
from classification.train_functions import k_fold_train
from torch.optim import AdamW
from torch import nn
from utils.helpers import get_logger
from utils.torch_helpers import set_default_device
from pprint import pformat
from torchsummary.torchsummary import summary


def get_models_and_logger(cfg: DemonstrationConfig) -> tuple[list[nn.Module], logging.Logger]:
    """
    Creates and returns a list of model instances and a training logger based on the configuration provided.

    The function supports the following model names specified in the `cfg` parameter:
    - EfficientNetV2
    - ViT
    - MiniVGG

    For each model type, a corresponding list of model instances and a logger for training are initialized.
    If the model name is not recognized, a NotImplementedError is raised.

    :param cfg: A configuration object of type `DemonstrationConfig` which includes the model name and
                other parameters needed for model instantiation and logging.
                - model_name (str): The name of the model to be used. Accepted values are 'EfficientNetV2', 'ViT', 'MiniVGG'.
                - num_folds (int): The number of model instances to create.

    :return:
        - models (list): A list of instantiated model objects.
        - train_logger (Logger): A logger object for training logs.

    :raises NotImplementedError: If the model name specified in `cfg.model_name` is not supported.
    """
    model_mapping = {
        'efficientnetv2': (CustomizedEfficientnetV2, cfg.get_efficientnet_kwargs, 'EfficientNetV2'),
        'vit': (CustomizedViT, cfg.get_vit_kwargs, 'ViT'),
        'minivgg': (MiniVGG, cfg.get_minivgg_arg, 'MiniVGG'),
        'convnext_v2': (CustomizedConvNeXT_V2, cfg.get_convnextv2_kwargs, 'ConvNeXt_V2')
    }

    model_key = cfg.model_name.lower()
    if model_key in model_mapping:
        model_class, model_kwargs_call, log_folder = model_mapping[model_key]
        models = [model_class(model_kwargs_call(), num_classes=len(cfg.class_mappings)) for _ in range(cfg.num_folds)]
        train_logger = get_logger(name=cfg.model_name + '_train',
                                  base_filepath=f'{cfg.log_dir}/{log_folder}')
    else:
        raise NotImplementedError(
            f'The model {cfg.model_name} is not implemented, please check your spelling or the documentation')

    return models, train_logger


def train_model(cfg: DemonstrationConfig) -> None:
    """
    Trains the specified model(s) using k-fold cross-validation and logs the training results.

    The function sets the default device (CPU or CUDA), initializes the loss function,
    retrieves the models and logger based on the provided configuration, and
    initializes the optimizers for each model. It then performs k-fold cross-validation
    training and logs the results. Also logs the summary of the model before training.

    :param cfg: A configuration object of type `DemonstrationConfig` which includes various parameters needed for training.
                - `cfg.model_name` (str): The name of the model to be used.
                - `cfg.num_folds` (int): The number of folds for cross-validation.
                - `cfg.learning_rate` (float): The learning rate for the optimizer.
                - `cfg.data_paths` (list): A list of paths to the training data.

    :return: None. The function performs training and logs the results but does not return any value.
    """
    set_default_device()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.CrossEntropyLoss()
    models, train_logger = get_models_and_logger(cfg=cfg)
    train_logger.info(summary(models[0], cfg.model_input_size))
    optimizers = [AdamW(lr=cfg.learning_rate, params=model.parameters()) for model in models]
    fold_results = k_fold_train(models=models, paths_to_data=cfg.train_data_paths, optimizers=optimizers,
                                loss_fn=loss_fn, logger=train_logger, device=device, cfg=cfg)
    train_logger.info(pformat(fold_results))


if __name__ == '__main__':
    train_model(DemonstrationConfig())
