import timm
from torch import nn
from typing import *


class CustomizedConvNeXT_V2(nn.Module):
    """ConvNeXT V2 model with a customized head. Has the option to be loaded with pretrained weights.
    Model architecture is based on https://arxiv.org/pdf/2301.00808"""

    def __init__(self, model_kwargs: Dict[str, Any], freeze: bool = False,
                 num_frozen_layers: int = 4, num_classes=28):
        super(CustomizedConvNeXT_V2, self).__init__()
        self.model = timm.create_model(
            **model_kwargs
        )
        if freeze:
            for i, (name, param) in enumerate(list(self.model.named_parameters())[0:num_frozen_layers]):
                param.requires_grad = False

        model = list(self.model.children()) + [nn.ReLU(), nn.Dropout(.25), nn.Linear(1000, num_classes)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
