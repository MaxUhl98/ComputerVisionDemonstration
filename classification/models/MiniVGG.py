from torch import nn
import torch


class MiniVGG(nn.Module):
    """MiniVGG implementation"""

    def __init__(self, input_channels: int, hidden_units: int = 128, num_classes: int = 28, dropout: float = .1,
                 linear_size: int = 64):
        linear1 = (512, linear_size)
        linear2 = (linear_size, num_classes)
        super(MiniVGG, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(hidden_units), nn.MaxPool2d(kernel_size=5), nn.Dropout(dropout))

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(hidden_units), nn.MaxPool2d(kernel_size=4), nn.Dropout(dropout))

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(hidden_units),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.BatchNorm2d(hidden_units), nn.MaxPool2d(kernel_size=4), nn.Dropout(dropout))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=linear1[0], out_features=linear1[1]), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(in_features=linear1[1], out_features=linear2[1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates values for image x (pre softmax)

        :param x: Input image tensor
        :return: Predicted values pre softmax
        """
        x = self.conv_block(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x
