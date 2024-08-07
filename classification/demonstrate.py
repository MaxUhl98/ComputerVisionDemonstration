import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from demo_configuration import DemonstrationConfig
from classification.models.EfficientNetV2 import CustomizedEfficientnetV2
from classification.models.ViT import CustomizedViT
from classification.models.MiniVGG import MiniVGG
from classification.data.image_data_class import ImageDataset
from classification.models.ConvNeXt_V2 import CustomizedConvNeXT_V2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import sample
from utils.torch_helpers import set_default_device
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from torch import nn


def plot_labelled_images(images: list[torch.Tensor], prediction_labels: list[str], true_labels: list[str],
                         num_rows: int,
                         num_cols: int,
                         cmap='gray', show_axis: bool = False) -> None:
    """
    Plot a matrix of labelled images.

    :param images: List of image tensors to be plotted.
    :param prediction_labels: List of predicted labels for the images.
    :param true_labels: List of true labels for the images.
    :param num_rows: Number of rows in the plot grid.
    :param num_cols: Number of columns in the plot grid.
    :param cmap: Colormap used for displaying images (default is 'gray').
    :param show_axis: Boolean flag to show or hide axis (default is False).
    :return: None
    """
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    rows, cols = num_rows, num_cols
    for i in range(cols):
        for j in range(rows):
            image = images[num_cols * i + j]
            ax[i, j].imshow(image, cmap=cmap)
            c = 'g' if prediction_labels[num_cols * i + j] == true_labels[num_cols * i + j] else 'r'
            ax[i, j].set_title(prediction_labels[num_cols * i + j] + " / " + true_labels[num_cols * i + j], color=c)
            ax[i, j].axis(show_axis)
    fig.suptitle('Sample Images with Predicted Label / True Label', fontsize=24)
    plt.show()


def plot_test_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, classnames: list[str]) -> None:
    """
    Plot a confusion matrix for test predictions.

    :param predictions: Tensor of model predictions.
    :param targets: Tensor of true target labels.
    :param classnames: List of class names corresponding to the target labels.
    :return: None
    """
    conf_mat = ConfusionMatrix(num_classes=len(classnames), task='multiclass')
    conf_mat_tensor = conf_mat(preds=predictions, target=targets)
    plot_confusion_matrix(conf_mat=conf_mat_tensor.cpu().numpy(), class_names=classnames, figsize=(10, 10))
    plt.show()


def load_pretrained_models(cfg: DemonstrationConfig) -> list[nn.Module]:
    """Loads pretrained models for the specified configuration.

    :param cfg: Configuration object
    :return: Pretrained models
    """
    model_mapping = {
        'efficientnetv2': (CustomizedEfficientnetV2, cfg.get_efficientnet_kwargs, 'EfficientNetV2'),
        'vit': (CustomizedViT, cfg.get_vit_kwargs, 'ViT'),
        'minivgg': (MiniVGG, cfg.get_minivgg_arg, 'MiniVGG'),
        'convnext_v2': (CustomizedConvNeXT_V2, cfg.get_convnextv2_kwargs, 'ConvNeXt_V2')
    }
    model_class, model_kwarg_getter, dir_name = model_mapping[cfg.model_name.lower()]
    trained_model_weights = [
        torch.load(cfg.model_save_path.split('.')[0] + f'_fold_{num}.pth', map_location=torch.get_default_device(),
                   weights_only=True) for num in range(cfg.num_folds)]
    trained_models = [model_class(model_kwarg_getter(), num_classes=len(cfg.class_mappings)) for _ in
                      trained_model_weights]  # Initialize Models
    [trained_models[num].load_state_dict(model_weights) for num, model_weights in
     enumerate(trained_model_weights)]  # Load model weights
    [model.eval() for model in trained_models]
    return trained_models


def get_test_paths(cfg: DemonstrationConfig) -> list[str]:
    """Finds all jpg, png and jpeg files in the configured test directory and concatenates them into a singular list

    :param cfg: Configuration object.
    :return: List of test image file paths.
    """
    paths = sum([list(Path(dir_path).glob('**/*/*.jpg')) for dir_path in cfg.test_data_paths], [])
    paths += sum([list(Path(dir_path).glob('**/*/*.png')) for dir_path in cfg.test_data_paths], [])
    paths += sum([list(Path(dir_path).glob('**/*/*.jpeg')) for dir_path in cfg.test_data_paths], [])
    return paths


def get_test_predictions(test_data: ImageDataset, cfg: DemonstrationConfig,
                         trained_models: list[nn.Module]) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs model inference to receive predictions for the test dataset

    :param test_data: Test dataset to perform inference on.
    :param cfg: Configuration Object
    :param trained_models: Models to use for inference
    :return: Tuple containing (prediction labels, target labels)
    """
    test_data_loader = DataLoader(test_data, batch_size=cfg.batch_size)
    all_predictions = []
    all_targets = []
    with torch.inference_mode():
        for batch in tqdm(test_data_loader):
            X, y = batch
            folds_ensemble_chances = torch.stack([F.softmax(model(X), dim=1) for model in trained_models], dim=0)
            folds_ensemble_chances = folds_ensemble_chances.sum(dim=0).div(cfg.num_folds).squeeze()
            all_predictions += [folds_ensemble_chances.argmax(dim=1).flatten()]
            all_targets += [y.flatten()]
    all_predictions = torch.concat(all_predictions)
    all_targets = torch.concat(all_targets)
    return all_predictions, all_targets


def get_samples(test_data: ImageDataset, all_predictions: torch.Tensor,
                cfg: DemonstrationConfig) -> tuple[list[torch.Tensor], list[str], list[str]]:
    """Uses the ground truth data and predictions to prepare data samples for the plot generation

    :param test_data: Test dataset
    :param all_predictions: Model predictions for the test dataset
    :param cfg: Configuration object
    :return: Tuple containing (Sample Images, Sample Labels, Sample Predicted Labels)
    """
    sample_datapoints = zip(
        *[(test_data.load_image(idx), test_data.targets[idx].item(), all_predictions[idx].item()) for idx in
          sample(range(len(test_data)), 16)])
    sample_images, sample_labels, sample_predictions = sample_datapoints
    label2_class_mapping = {v: k for k, v in cfg.class_mappings.items()}
    sample_labels, sample_predictions = [label2_class_mapping[prediction] for prediction in sample_predictions], [
        label2_class_mapping[label] for label in sample_labels]
    return sample_images, sample_labels, sample_predictions


def demonstrate(cfg: DemonstrationConfig) -> None:
    """
    Demonstrate model performance by plotting labelled images and a confusion matrix.

    :param cfg: DemonstrationConfig object containing configuration for the demonstration.
    :return: None
    """
    set_default_device()
    trained_models = load_pretrained_models(cfg)
    paths = get_test_paths(cfg)
    data = pd.DataFrame(
        {'path': paths, 'target': [cfg.class_mappings.get(data_path.parent.name) for data_path in paths]})
    test_data = ImageDataset(data)
    all_predictions, all_targets = get_test_predictions(test_data, cfg, trained_models)
    sample_images, sample_labels, sample_predictions = get_samples(test_data, all_predictions, cfg)
    plot_labelled_images(sample_images, prediction_labels=sample_predictions, true_labels=sample_labels, num_cols=4,
                         num_rows=4)
    plot_test_confusion_matrix(all_predictions.to(torch.get_default_device()),
                               all_targets.to(torch.get_default_device()), list(cfg.class_mappings.keys()))
