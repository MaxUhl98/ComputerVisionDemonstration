import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import *
from os import PathLike
from pathlib import Path
import torch
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Transform, Resize
from PIL import Image


class VegetableData(Dataset):
    """Loads Vegetable Dataset from
    https://www.researchgate.net/publication/352846889_DCNN-Based_Vegetable_Image_Classification_Using_Transfer_Learning_A_Comparative_Study
    Published by https://www.kaggle.com/misrakahmed at
    https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset
    Does not augment the data in any kind of way"""

    data_paths: List[Path]
    labels: torch.Tensor
    image_processor: Transform

    def __init__(self, data: pd.DataFrame, data_type: torch.dtype = torch.float32):
        super(VegetableData, self).__init__()
        self.data_paths = data.path.values
        self.targets = torch.from_numpy(data.target.values.astype(np.int64))
        self.image_processor = Compose([ToImage(), Resize((224, 224)), ToDtype(data_type, scale=True)])
        self.images = [self.image_processor(self.load_image(idx)) for idx in range(len(self.targets))]
        # Filter images to guarantee that all datapoints have the right amount of channels
        self.images, self.targets = zip(
            *[(img, target) for img, target in zip(self.images, self.targets) if img.shape[0] == 3])
        self.targets = torch.tensor(self.targets)

    def __len__(self) -> int:
        """Returns total number of samples"""
        return self.targets.shape[0]

    def load_image(self, index: int) -> Image.Image:
        """Opens an Image via path and returns it"""
        image_path = self.data_paths[index]
        return Image.open(image_path)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns one sample of data, data and label (X,y)"""
        return self.images[idx], self.targets[idx]


def get_class_mappings(_data: VegetableData) -> Dict[str, int]:
    """
    Generates a dictionary that maps each unique class name to a number for training / inference usage.

    :param _data: An instance of the VegetableData class containing data paths.
    :return: A dictionary where keys are class names (str) and values are the number representing that class during
    training/inference (int).
    """
    _data = _data.data_paths
    _data = [i.parent.name for i in _data]
    classes = sorted(list(set(_data)))
    return {class_name: class_idx for class_idx, class_name in enumerate(classes)}
