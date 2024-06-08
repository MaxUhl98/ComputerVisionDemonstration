from classification.models.EfficientNetV2 import CustomizedEfficientnetV2
import torch
from tests.test_classification.mock_config import MockConfig


def test_vit():
    model = CustomizedEfficientnetV2(MockConfig.get_efficientnet_kwargs(), num_classes=2)
    assert model(torch.rand([1, 3, 224, 224])).shape == torch.Size([1, 2])
