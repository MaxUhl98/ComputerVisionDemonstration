import torch

import classification.demonstrate as d
from classification.data.image_data_class import ImageDataset
from tests.test_classification.mock_config import MockConfig
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib


def get_img_data():
    paths = sum([list(Path(dir_path).glob('**/*/*.jpg')) for dir_path in MockConfig.train_data_paths], [])
    data = pd.DataFrame(
        {'path': paths, 'target': [MockConfig.class_mappings.get(data_path.parent.name) for data_path in paths]})
    return ImageDataset(data)


def test_plot_labelled_images():
    matplotlib.use('Agg')
    images = [i[0].permute([1, 2, 0]) for i in get_img_data()]
    d.plot_labelled_images(images, ['Eagle' for _ in range(4)], ['Eagle', 'Eagle', 'Space', 'Space'], 2, 2)
    plt.close()


def test_plot_test_confusion_matrix():
    test_predictions = torch.tensor([0, 0, 1, 0], dtype=torch.int64, device=torch.get_default_device())
    test_targets = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=torch.get_default_device())
    classnames = ['Eagle', 'Space']
    d.plot_test_confusion_matrix(test_predictions, test_targets, classnames)

