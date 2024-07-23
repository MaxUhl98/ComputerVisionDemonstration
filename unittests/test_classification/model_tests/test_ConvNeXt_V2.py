from classification.models.ConvNeXt_V2 import CustomizedConvNeXT_V2
import torch
from unittests.test_classification.mock_config import MockConfig


def test_convnextv2():
    model = CustomizedConvNeXT_V2(MockConfig.get_convnextv2_kwargs(), num_classes=2)
    assert model(torch.rand([1, 3, 224, 224])).shape == torch.Size([1, 2])
