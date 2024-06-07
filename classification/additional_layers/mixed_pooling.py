import torch
from torch import nn
import torch.nn.functional as F


class MixedPool(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, alpha=0.5, device=torch.get_default_device()):
        super(MixedPool, self).__init__()
        alpha = torch.FloatTensor([alpha]).to(device)
        self.alpha = nn.Parameter(alpha).to(device)  # Make alpha learnable
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (
                1 - self.alpha) * F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x
