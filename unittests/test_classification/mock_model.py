from typing import Mapping, Any

import pytest
import torch


class MockModel(torch.nn.Module):
    def __init__(self, num_outputs):
        super(MockModel, self).__init__()
        self.output = torch.tensor([[1., 0.] for _ in range(num_outputs)], requires_grad=True)

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        pass

    def forward(self, x):
        return self.output
