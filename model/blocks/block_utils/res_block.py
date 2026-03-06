import torch.nn as nn

from .deformable_conv2d import DeformableConv2d

class ResBlock(nn.Module):
    """
        ResBlock using bottleneck structure
    """

    def __init__(self, dim, use_dropout):
        super(ResBlock, self).__init__()

        sequence = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out
