import timm
from torch import nn
from typing import *


class CustomizedEfficientnetV2(nn.Module):
    def __init__(self, model_kwargs: Dict[str, Any], num_classes: int = 28, freeze: bool = False,
                 num_frozen_layers: int = 4):
        super(CustomizedEfficientnetV2, self).__init__()
        self.model = timm.create_model(
            **model_kwargs
        )
        if freeze:
            for i, (name, param) in enumerate(list(self.model.named_parameters())[0:num_frozen_layers]):
                param.requires_grad = False
        model = list(self.model.children())[:-2] + [nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                                    nn.Linear(self.model.num_features, num_classes)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
