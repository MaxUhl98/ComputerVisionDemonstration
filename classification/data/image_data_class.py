import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import *
from pathlib import Path
import torch
from torchvision.transforms.v2 import Compose, ToPILImage, PILToTensor, ToDtype, Transform, Resize, RGB
from PIL import Image


class ImageDataset(Dataset):
    """Class that brings all given images into a standardized format and prepares them for pytorch.
    Has the option to augment data with transforms specified in the configuration file.

    """

    data_paths: List[Path]
    labels: torch.Tensor
    image_processor: Transform

    def __init__(self, data: pd.DataFrame, data_type: torch.dtype = torch.float32,
                 augmentations: Union[Transform, None] = None):
        super(ImageDataset, self).__init__()
        self.device = torch.get_default_device()
        self.data = data
        self.data_paths = data.path.values
        self.targets = torch.from_numpy(data.target.values.astype(np.int64))
        self.image_processor = Compose(
            [ToPILImage(mode='RGB'), PILToTensor(), Resize((224, 224)), ToDtype(data_type, scale=True)])
        self.images = [self.image_processor(self.load_image(idx))[:3, :, :].to('cpu') for idx in
                       range(len(self.targets))]
        if isinstance(augmentations, Transform):
            self.images = augmentations(self.images)
            self.images = [img.to(self.device) for img in self.images]

    def __len__(self) -> int:
        """Returns total number of samples"""
        return self.targets.shape[0]

    def load_image(self, index: int) -> Image.Image:
        """Opens an Image via path and returns it"""
        image_path = self.data_paths[index]
        return Image.open(image_path)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns one sample of data, data and label (X,y)"""

        return (self.images[idx].to(self.device) if self.images[idx].shape[0] == 3 else torch.cat(
            [self.images[idx], self.images[idx], self.images[idx]], 0).to(self.device),
                self.targets[idx].to(self.device))


def get_class_mappings(_data: ImageDataset) -> Dict[str, int]:
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
