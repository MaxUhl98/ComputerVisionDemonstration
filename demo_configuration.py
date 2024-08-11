from typing import *
from torchvision.transforms import v2
import os
from utils import get_label_mapping_dictionary
from torch.optim.lr_scheduler import *


class DemonstrationConfig:
    """
    Configuration class for model training and testing.

    :param model_name: The name of the model to be used.
    :param num_folds: The number of folds for cross-validation.
    :param shuffle_folds: Whether to shuffle data before creating folds.
    :param batch_size: The batch size for training.
    :param num_epochs: The number of epochs for training.
    :param patience: The number of epochs to wait for improvement before early stopping.
    :param learning_rate: The learning rate for the optimizer.
    :param max_grad_norm: The maximum gradient norm for gradient clipping.
    :param apex: Whether to use Apex for mixed precision training.
    :param train_data_paths: List of paths to directories containing training image folders.
    :param test_data_paths: List of paths to directories containing test image folders.
    :param use_reduced_dataset: Whether to use a reduced dataset for training.
    :param reduced_percentage: The percentage of the dataset to use if `use_reduced_dataset` is True.
    :param augmentations: The data augmentation transforms to be applied during training.
    :param save_best: Whether to save the best model based on validation performance.
    :param model_save_path: The path where the model will be saved.
    :param lr_scheduling: Whether to use learning rate scheduling.
    :param lr_schedule_class: The learning rate scheduling class to be used.
    :param lr_kwargs: The keyword arguments for learning rate scheduling.
    :param log_dir: Directory where training logs will be saved.
    :param log_batch_loss: Whether to log the loss for each batch.
    :param model_input_size: The input size of the model in (channels, height, width) format.
    :param class_mappings: A dictionary mapping class names to integer labels.
    """
    # Choose one out of ['EfficientNetV2', 'ViT', 'MiniVGG', 'ConvNeXt_V2']
    model_name: str = 'ViT'

    # Cross Validation settings
    num_folds: int = 10
    shuffle_folds: bool = True

    # Train Settings
    batch_size: int = 8  # recommended to be a power of 2, e.g. one of 2, 4, 8, 16, 32, ...
    num_epochs: int = 100
    patience: int = 10
    learning_rate: float = 10 ** -5
    max_grad_norm: Union[int, float] = 10 ** 5  # Clips large gradients to this value
    apex: bool = False

    # Data paths
    train_data_paths: list[Union[str, os.PathLike]] = [
        r'classification/data/Eye_Disease_Detection/train/train']
    test_data_paths: list[Union[str, os.PathLike]] = [
        r'classification/data/Eye_Disease_Detection/validation/validation']

    # Data Settings
    use_reduced_dataset: bool = False
    reduced_percentage: float = .1
    augmentations: v2.Transform = v2.Compose([v2.AutoAugment()])

    # Weight saving settings
    save_best: bool = True
    model_save_path: str = f'classification/models/trained_models/{model_name}/{model_name}.pth'

    # LR Scheduling
    lr_scheduling: bool = True
    lr_schedule_class: Callable = CosineAnnealingWarmRestarts
    lr_kwargs: Dict[str, Any] = {'t_0':3,'eta_min':10**-8}

    # Logging settings
    log_dir: Union[str, os.PathLike] = 'classification/models/training_logs'
    log_batch_loss: bool = False

    # Only touch if you know what you are doing
    model_input_size: tuple[int, int, int] = (3, 224, 224)
    class_mappings: dict[str, int] = get_label_mapping_dictionary(train_data_paths[0])

    @staticmethod
    def get_efficientnet_kwargs() -> dict[str, Any]:
        """
        Get the keyword arguments for the EfficientNet model.

        :return: A dictionary of keyword arguments for the EfficientNet model.
        """
        return {
            'model_name': "tf_efficientnetv2_s.in1k",
            'pretrained': True,
            'drop_rate': 0.1,
            'drop_path_rate': 0.2, }

    @staticmethod
    def get_vit_kwargs() -> dict[str, Any]:
        """
        Get the keyword arguments for the Vision Transformer (ViT) model.

        :return: A dictionary of keyword arguments for the ViT model.
        """
        return {
            'model_name': "vit_base_patch32_clip_224",
            'pretrained': True}

    @staticmethod
    def get_convnextv2_kwargs() -> dict[str, Any]:
        """
        Get the keyword arguments for the ConvNeXtV2 model.

        :return: A dictionary of keyword arguments for the ConvNeXtV2 model.
        """
        return {
            'model_name': "convnextv2_pico.fcmae_ft_in1k",
            'pretrained': True}

    @staticmethod
    def get_minivgg_arg() -> int:
        """
        Get the number of Input Channels for miniVGG.

        :return: Number of input channels for the MiniVGG model.
        """
        return 3
