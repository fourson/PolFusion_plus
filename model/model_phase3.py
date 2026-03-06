import torch
import torch.nn as nn

from base.base_model import BaseModel
from .blocks.block_utils.res_block import ResBlock
from .blocks.block_utils.se_block import SEBlock
from .blocks.block_utils.unet import AttentionUnetBackbone


class DefaultModel(BaseModel):
    def __init__(self, dim=32):
        super(DefaultModel, self).__init__()

        self.encoder_13 = nn.Sequential(
            nn.Conv2d(6, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim // 2, use_dropout=False)
        )
        self.encoder_24 = nn.Sequential(
            nn.Conv2d(6, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim // 2, use_dropout=False)
        )
        self.fuse = nn.Sequential(
            ResBlock(dim, use_dropout=False),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            SEBlock(in_channel=dim, r=dim // 4)
        )
        self.backbone = AttentionUnetBackbone(input_nc=dim, output_nc=dim, n_downsampling=3,
                                              use_conv_to_downsample=False, use_dropout=False, mode='bottleneck',
                                              deformable=True)
        self.out_block13 = nn.Sequential(
            nn.Conv2d(dim, 6, kernel_size=1, stride=1)
        )
        self.out_block24 = nn.Sequential(
            nn.Conv2d(dim, 6, kernel_size=1, stride=1)
        )

    def forward(self, I1_temp, I2_temp, I3_temp, I4_temp):
        I13_temp = torch.cat([I1_temp, I3_temp], dim=1)
        I24_temp = torch.cat([I2_temp, I4_temp], dim=1)
        f13 = self.encoder_13(I13_temp)
        f24 = self.encoder_24(I24_temp)
        f = self.fuse(torch.cat([f13, f24], dim=1))
        backbone_out = self.backbone(f)
        I13_out = self.out_block13(backbone_out) + I13_temp
        I13_out = torch.clamp(I13_out, min=0, max=1)
        I24_out = self.out_block24(backbone_out) + I24_temp
        I24_out = torch.clamp(I24_out, min=0, max=1)

        I1_out, I3_out = I13_out[:, 0:3, :, :], I13_out[:, 3:6, :, :]
        I2_out, I4_out = I24_out[:, 0:3, :, :], I24_out[:, 3:6, :, :]
        return I1_out, I2_out, I3_out, I4_out
