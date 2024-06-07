import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from demo_configuration import DemonstrationConfig
from classification.models.EfficientNetV2 import CustomizedEfficientnetV2
from classification.models.ViT import CustomizedViT
from classification.models.MiniVGG import MiniVGG
from classification.data.vegetable_data_class import VegetableData
from classification.models.ConvNeXt_V2 import CustomizedConvNeXT_V2
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import sample
from utils.torch_helpers import set_default_device
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def plot_labelled_images(images: list[torch.Tensor], prediction_labels: list[str], true_labels: list[str],
                         num_rows: int,
                         num_cols: int,
                         cmap='gray', show_axis: bool = False) -> None:
    """Plot Matrix of Labelled Images"""
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    rows, cols = num_rows, num_cols
    for i in range(cols):
        for j in range(rows):
            image = images[num_cols * i + j]
            ax[i, j].imshow(image, cmap=cmap)
            c = 'g' if prediction_labels[num_cols * i + j] == true_labels[num_cols * i + j] else 'r'
            ax[i, j].set_title(prediction_labels[num_cols * i + j] + " / " + true_labels[num_cols * i + j], color=c)
            ax[i, j].axis(show_axis)
    plt.show()


def plot_test_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, classnames: list[str]) -> None:
    conf_mat = ConfusionMatrix(num_classes=len(classnames), task='multiclass')
    conf_mat_tensor = conf_mat(preds=predictions, target=targets)
    plot_confusion_matrix(conf_mat=conf_mat_tensor.cpu().numpy(), class_names=classnames, figsize=(10, 10))
    plt.show()


def demonstrate(cfg: DemonstrationConfig):
    set_default_device()
    model_mapping = {
        'efficientnetv2': (CustomizedEfficientnetV2, cfg.get_efficientnet_kwargs, 'EfficientNetV2'),
        'vit': (CustomizedViT, cfg.get_vit_kwargs, 'ViT'),
        'minivgg': (MiniVGG, cfg.get_minivgg_arg, 'MiniVGG'),
        'convnext_v2': (CustomizedConvNeXT_V2, cfg.get_convnextv2_kwargs, 'ConvNeXt_V2')
    }
    model_class, model_kwarg_getter, dir_name = model_mapping[cfg.model_name.lower()]
    trained_model_weight_paths = [torch.load(cfg.model_save_path.split('.')[0] + f'_fold_{num}.pth') for num in
                                  range(cfg.num_folds)]

    trained_models = [model_class(model_kwarg_getter()) for model_weights in
                      trained_model_weight_paths]  # Initialize Models
    [trained_models[num].load_state_dict(model_weights) for num, model_weights in
     enumerate(trained_model_weight_paths)]  # Load model weights
    paths = sum([list(Path(dir_path).glob('**/*/*.jpg')) for dir_path in cfg.test_data_path], [])
    paths += sum([list(Path(dir_path).glob('**/*/*.png')) for dir_path in cfg.test_data_path], [])
    data = pd.DataFrame(
        {'path': paths, 'target': [cfg.class_mappings.get(data_path.parent.name) for data_path in paths]})
    test_data = VegetableData(data)
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

    sample_datapoints = zip(
        *[(test_data.load_image(idx), test_data.targets[idx].item(), all_predictions[idx].item()) for idx in
          sample(range(len(test_data)), 16)])
    sample_images, sample_labels, sample_predictions = sample_datapoints
    label2_class_mapping = {v: k for k, v in cfg.class_mappings.items()}
    sample_labels, sample_predictions = [label2_class_mapping[prediction] for prediction in sample_predictions], [
        label2_class_mapping[label] for label in sample_labels]

    plot_labelled_images(sample_images, prediction_labels=sample_predictions, true_labels=sample_labels, num_cols=4,
                         num_rows=4)
    plot_test_confusion_matrix(all_predictions.to(torch.get_default_device()),
                               all_targets.to(torch.get_default_device()), list(cfg.class_mappings.keys()))
