import os

import pytest
from classification.data.image_data_class import VegetableData, get_class_mappings
from pathlib import Path
import pandas as pd
from tests.test_classification.mock_config import MockConfig

class TestVegetableData:
    def set_up(self):
        self.cfg = MockConfig()
        paths = sum([list(Path(dir_path).glob('**/*/*.jpg')) for dir_path in self.cfg.train_data_paths], [])
        data = pd.DataFrame(
            {'path': paths, 'target': [self.cfg.class_mappings.get(data_path.parent.name) for data_path in paths]})
        self.data_class = VegetableData(data)

    def test_vegetable_data_len(self):
        self.set_up()
        assert len(self.data_class) == 2

    def test_vegetable_getitem(self):
        self.set_up()
        img1, label1 = self.data_class[0]
        img2, label2 = self.data_class[1]
        assert img1.shape == (3, 224, 224)
        assert img2.shape == (3, 224, 224)
        assert label1.item() == 0
        assert label2.item() == 1

    def test_get_class_mappings(self):
        self.set_up()
        assert get_class_mappings(self.data_class) == self.cfg.class_mappings
