import timm
from torch import nn
from typing import *


class CustomizedViT(nn.Module):
    """ViT model with a customized head. Has the option to be loaded with pretrained weights.
        Model architecture is based on https://arxiv.org/pdf/2010.11929"""
    def __init__(self, model_kwargs: Dict[str, Any], freeze: bool = False,
                 num_frozen_layers: int = 4, num_classes=28):
        super(CustomizedViT, self).__init__()
        self.model = timm.create_model(
            **model_kwargs
        )
        if freeze:
            for i, (name, param) in enumerate(list(self.model.named_parameters())[0:num_frozen_layers]):
                param.requires_grad = False
        model = list(self.model.children())[:-1] + [nn.Flatten(), nn.ReLU(), nn.Linear(37632, num_classes)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
