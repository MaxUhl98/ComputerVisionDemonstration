import os
from typing import *


class MockConfig:
    model_name: str = 'ViT'

    # Train Hyperparameters
    num_folds: int = 2
    batch_size: int = 1
    num_epochs: int = 1
    patience: int = 1
    learning_rate: float = 10 ** -5
    max_grad_norm: [int, float] = 10 ** 5
    gradient_accumulation_steps: int = 1

    model_input_size: tuple[int, int, int] = (3, 224, 224)

    # Paths to directories containing train image folders
    train_data_paths: list[Union[str, os.PathLike]] = [r'test_files/image_files']
    # Paths to directories containing test image folders
    test_data_path: list[Union[str, os.PathLike]] = [r'test_files/image_files']
    # Train Settings
    save_best: bool = True
    model_save_path: str = f'test_files/fake_weight_save_path/{model_name}.pth'
    shuffle_folds: bool = True
    log_batch_loss: bool = False
    apex: bool = False
    lr_scheduling: bool = False

    log_dir = 'test_files/fake_logs'

    class_mappings: dict[str, int] = {'eagle': 0, 'space': 1}

    num_classes = len(class_mappings)

    @staticmethod
    def get_testmodel_kwargs() -> dict[str, Any]:
        return {
            'model_name': "testname",
            'pretrained': True,
            'drop_rate': 0.1,
            'drop_path_rate': 0.2}
