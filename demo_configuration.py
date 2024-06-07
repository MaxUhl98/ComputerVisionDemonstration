import os
from typing import Any, Union


class DemonstrationConfig:
    model_name: str = 'ConvNeXt_V2'

    # Train Hyperparameters
    num_folds: int = 5
    batch_size: int = 16
    num_epochs: int = 50
    patience: int = 1
    learning_rate: float = 10 ** -5
    max_grad_norm: [int, float] = 10 ** 5
    gradient_accumulation_steps: int = 1

    model_input_size: tuple[int, int, int] = (3, 224, 224)

    # Paths to directories containing train image folders
    train_data_paths: list[Union[str, os.PathLike]] = [
        r'classification\data\Fruit And Vegetable Diseases Dataset\train']
    # Paths to directories containing test image folders
    test_data_path: list[Union[
        str, os.PathLike]] = [
        r'C:classification\data\Fruit And Vegetable Diseases Dataset\test']
    # Train Settings
    save_best: bool = True
    model_save_path: str = f'classification/models/trained_models/{model_name}/{model_name}.pth'
    shuffle_folds: bool = True
    log_batch_loss: bool = False
    apex: bool = False
    lr_scheduling: bool = False

    class_mappings: dict[str, int] = {'Apple__Healthy': 0, 'Apple__Rotten': 1, 'Banana__Healthy': 2,
                                      'Banana__Rotten': 3, 'Bellpepper__Healthy': 4, 'Bellpepper__Rotten': 5,
                                      'Carrot__Healthy': 6, 'Carrot__Rotten': 7, 'Cucumber__Healthy': 8,
                                      'Cucumber__Rotten': 9, 'Grape__Healthy': 10, 'Grape__Rotten': 11,
                                      'Guava__Healthy': 12, 'Guava__Rotten': 13, 'Jujube__Healthy': 14,
                                      'Jujube__Rotten': 15, 'Mango__Healthy': 16, 'Mango__Rotten': 17,
                                      'Orange__Healthy': 18, 'Orange__Rotten': 19, 'Pomegranate__Healthy': 20,
                                      'Pomegranate__Rotten': 21, 'Potato__Healthy': 22, 'Potato__Rotten': 23,
                                      'Strawberry__Healthy': 24, 'Strawberry__Rotten': 25, 'Tomato__Healthy': 26,
                                      'Tomato__Rotten': 27}



    @staticmethod
    def get_efficientnet_kwargs() -> dict[str, Any]:
        return {
            'model_name': "tf_efficientnetv2_s.in1k",
            'pretrained': True,
            'drop_rate': 0.1,
            'drop_path_rate': 0.2, }


    @staticmethod


    def get_vit_kwargs() -> dict[str, Any]:
        return {
            'model_name': "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
            'pretrained': True}


    @staticmethod
    def get_convnextv2_kwargs() -> dict[str, Any]:
        return {
            'model_name': "convnextv2_pico.fcmae_ft_in1k",
            'pretrained': True}


    @staticmethod
    def get_minivgg_arg() -> int:
        return 3
