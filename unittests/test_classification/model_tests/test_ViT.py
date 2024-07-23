from classification.models.ViT import CustomizedViT
import torch
from unittests.test_classification.mock_config import MockConfig


def test_vit():
    model = CustomizedViT(MockConfig.get_vit_kwargs(), num_classes=2)
    assert model(torch.rand([1, 3, 224, 224])).shape == torch.Size([1, 2])
