import torch
import torch.nn as nn

from base.base_model import BaseModel
from .blocks.blocks_phase1 import ColorFeatureRectification, ColorAndStructureCueFusion
from .blocks.block_utils.res_block import ResBlock
from .blocks.block_utils.dense_block import DenseBlock
from .blocks.block_utils.autoencoder import SkipAutoencoderDownsamplingBlock, SkipAutoencoderUpsamplingBlock

from utils.util import torch_laplacian


class DefaultModel(BaseModel):
    def __init__(self, dim=16):
        super(DefaultModel, self).__init__()

        self.color_encoder = ColorFeatureRectification(dim=dim)

        self.color_down1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        )
        self.color_down2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(6, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.structure_down1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        )
        self.structure_down2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )
        self.input_encoder = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.down1 = SkipAutoencoderDownsamplingBlock(in_channel=dim, out_channel=dim * 2, use_dropout=False,
                                                      use_channel_shuffle=False)
        self.down2 = SkipAutoencoderDownsamplingBlock(in_channel=dim * 2, out_channel=dim * 4, use_dropout=False,
                                                      use_channel_shuffle=False)
        self.down3 = SkipAutoencoderDownsamplingBlock(in_channel=dim * 4, out_channel=dim * 8, use_dropout=False,
                                                      use_channel_shuffle=False)
        self.CSCF1 = ColorAndStructureCueFusion(input_nc=dim, output_nc=dim, dim=dim)
        self.CSCF2 = ColorAndStructureCueFusion(input_nc=dim * 2, output_nc=dim * 2, dim=dim * 2)
        self.CSCF3 = ColorAndStructureCueFusion(input_nc=dim * 4, output_nc=dim * 4, dim=dim * 4)
        self.inner = DenseBlock(in_channel=dim * 8, growth_rate=dim, n_blocks=3)
        self.up1 = SkipAutoencoderUpsamplingBlock(in_channel1=dim * 8, in_channel2=dim * 4, out_channel=dim * 4,
                                                  use_dropout=False, use_channel_shuffle=False)
        self.up2 = SkipAutoencoderUpsamplingBlock(in_channel1=dim * 4, in_channel2=dim * 2, out_channel=dim * 2,
                                                  use_dropout=False, use_channel_shuffle=False)
        self.up3 = SkipAutoencoderUpsamplingBlock(in_channel1=dim * 2, in_channel2=dim, out_channel=dim,
                                                  use_dropout=False, use_channel_shuffle=False)

        self.out_block = nn.Conv2d(dim, 3, kernel_size=1, stride=1)

    def forward(self, S0_B, mask, S0_L, S1_L, S2_L):
        # normalize
        S0_B = S0_B / 2
        S0_L = S0_L / 2
        S1_L = (S1_L + 1) / 2
        S2_L = (S2_L + 1) / 2

        # multi-scale color feature
        color_feature1 = self.color_encoder(S0_B, mask, S0_L)  # color feature rectifying
        color_feature2 = self.color_down1(color_feature1)
        color_feature3 = self.color_down2(color_feature2)

        # multi-scale structure feature
        structure_feature1 = self.structure_encoder(torch.cat((torch_laplacian(S1_L), torch_laplacian(S2_L)), dim=1))
        structure_feature2 = self.structure_down1(structure_feature1)
        structure_feature3 = self.structure_down2(structure_feature2)

        # down-sampling
        input_feature1 = self.input_encoder(S0_L)
        input_feature2 = self.down1(input_feature1)
        input_feature3 = self.down2(input_feature2)
        input_feature4 = self.down3(input_feature3)

        # color and structure feature fusing
        fused_feature1 = self.CSCF1(input_feature1, color_feature1, structure_feature1)
        fused_feature2 = self.CSCF2(input_feature2, color_feature2, structure_feature2)
        fused_feature3 = self.CSCF3(input_feature3, color_feature3, structure_feature3)

        # up-sampling
        output_feature1 = self.inner(input_feature4)
        output_feature2 = self.up1(output_feature1, fused_feature3)
        output_feature3 = self.up2(output_feature2, fused_feature2)
        output_feature4 = self.up3(output_feature3, fused_feature1)

        # out
        out = self.out_block(output_feature4) + S0_L
        S0_temp = torch.clamp(out, min=0, max=1)
        # denormalize
        S0_temp = S0_temp * 2
        return S0_temp
