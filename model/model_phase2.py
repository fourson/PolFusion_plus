import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel
from .blocks.blocks_phase2 import EdgeGuidedFeatureProjection, CoherenceAwareMutualAggregation, \
    MultiScaleCoherenceInjection
from .blocks.block_utils.dense_block import DenseBlock


class DefaultModel(BaseModel):
    def __init__(self, dim=32):
        super(DefaultModel, self).__init__()

        self.feature_encoder = nn.Sequential(
            nn.Conv2d(12, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            DenseBlock(in_channel=dim, growth_rate=dim, n_blocks=2)
        )

        self.EGFP = EdgeGuidedFeatureProjection(dim=dim // 2)
        self.CAMA = CoherenceAwareMutualAggregation(dim=dim // 2)

        self.coherence1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        )  # no downsampling
        self.coherence2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        )  # downsampling
        self.coherence3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )  # downsampling

        self.MSCI = MultiScaleCoherenceInjection(dim=dim)

    def forward(self, x_temp, x_B, y_temp, y_B, S0_temp, mask):
        # normalize
        x_temp = (x_temp + 1) / 2
        x_B = (x_B + 1) / 2
        y_temp = (y_temp + 1) / 2
        y_B = (y_B + 1) / 2
        S0_temp = S0_temp / 2

        # padding
        x_temp = F.pad(x_temp, pad=[16, 16, 16, 16], mode='reflect')
        x_B = F.pad(x_B, pad=[16, 16, 16, 16], mode='reflect')
        y_temp = F.pad(y_temp, pad=[16, 16, 16, 16], mode='reflect')
        y_B = F.pad(y_B, pad=[16, 16, 16, 16], mode='reflect')
        S0_temp = F.pad(S0_temp, pad=[16, 16, 16, 16], mode='reflect')
        mask = F.pad(mask, pad=[16, 16, 16, 16], mode='reflect')

        # network logic
        fx, fy = self.EGFP(x_temp, x_B, y_temp, y_B, S0_temp, mask)  # dim // 2
        f_fuse = self.CAMA(fx, fy)  # dim
        f_coherence1 = self.coherence1(f_fuse)  # dim
        f_coherence2 = self.coherence2(f_coherence1)  # dim * 2
        f_coherence3 = self.coherence3(f_coherence2)  # dim * 4
        features = self.feature_encoder(torch.cat((x_temp, x_B, y_temp, y_B), dim=1))  # dim
        res = self.MSCI(features, f_coherence1, f_coherence2, f_coherence3)  # 6
        xy_out = torch.clamp(res + torch.cat((x_temp, y_temp), dim=1), min=0, max=1)

        # out x and y
        x_out = xy_out[:, 0:3, :, :]
        y_out = xy_out[:, 3:6, :, :]

        # denormalize
        x_out = x_out * 2 - 1
        y_out = y_out * 2 - 1

        # cropping
        x_out = x_out[:, :, 16:-16, 16:-16]
        y_out = y_out[:, :, 16:-16, 16:-16]

        return x_out, y_out
