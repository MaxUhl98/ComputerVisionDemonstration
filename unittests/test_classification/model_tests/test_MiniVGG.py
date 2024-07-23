from classification.models.MiniVGG import MiniVGG
import torch


def test_mini_vgg():
    model = MiniVGG(input_channels=3, num_classes=2)
    assert model(torch.rand([1, 3, 224, 224])).shape == torch.Size([1,2])
