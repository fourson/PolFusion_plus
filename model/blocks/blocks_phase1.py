import torch.nn as nn

from .block_utils.attention_block import AttentionBlock
from .block_utils.res_block import ResBlock
from .block_utils.deformable_conv2d import DeformableConv2d


class ColorFeatureRectification(nn.Module):
    def __init__(self, dim=16):
        super(ColorFeatureRectification, self).__init__()

        self.proj_B = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.proj_uB = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=1, stride=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.attention = AttentionBlock(in_channel_x=dim, in_channel_g=dim, channel_t=dim // 2)
        self.attention_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

    def forward(self, S0_B, mask, S0_L):
        f_B = self.proj_B(S0_B)
        updated_S0_B = mask * S0_B + (1 - mask) * S0_L
        f_uB = self.proj_uB(updated_S0_B)
        f = self.attention_out(self.attention(f_B, f_uB))
        return f


class ColorAndStructureCueFusion(nn.Module):
    def __init__(self, input_nc, output_nc, dim):
        super(ColorAndStructureCueFusion, self).__init__()

        self.proj_conv_color = nn.Sequential(
            nn.Conv2d(input_nc, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.Sigmoid()
        )
        self.b_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.proj_conv_structure = nn.Sequential(
            nn.Conv2d(input_nc, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.offset_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.modulator_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.align_conv = DeformableConv2d(offset_in_channels=dim, modulator_in_channels=dim, in_channels=dim,
                                           out_channels=output_nc, kernel_size=3, stride=1, padding=1, dilation=1,
                                           bias=False)

    def forward(self, f_in, f_color, f_structure):
        f_color = self.proj_conv_color(f_color)
        m = self.m_conv(f_color)
        b = self.b_conv(f_color)
        f_in = f_in * m + b
        f_structure = self.proj_conv_structure(f_structure)
        offset = self.offset_conv(f_structure)
        modulator = self.modulator_conv(f_structure)
        f_in = self.align_conv(f_in, offset, modulator)
        return f_in
