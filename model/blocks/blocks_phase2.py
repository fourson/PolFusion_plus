import torch
import torch.nn as nn

from .block_utils.res_block import ResBlock
from .block_utils.cross_attention import Chuncked_Cross_Attn_FM
from .block_utils.cbam import CBAM
from .block_utils.attention_block import AttentionBlock

from utils.util import torch_laplacian


class EdgeGuidedFeatureProjection(nn.Module):
    def __init__(self, dim):
        super(EdgeGuidedFeatureProjection, self).__init__()

        self.proj_x = nn.Sequential(
            nn.Conv2d(9, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.proj_edge_x = nn.Sequential(
            nn.Conv2d(6, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.attention_x = AttentionBlock(in_channel_x=dim, in_channel_g=dim, channel_t=dim // 2)
        self.out_x = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

        self.proj_y = nn.Sequential(
            nn.Conv2d(9, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.proj_edge_y = nn.Sequential(
            nn.Conv2d(6, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.attention_y = AttentionBlock(in_channel_x=dim, in_channel_g=dim, channel_t=dim // 2)
        self.out_y = nn.Conv2d(dim, dim, kernel_size=1, stride=1)

    def forward(self, x_temp, x_B, y_temp, y_B, S0_temp, mask):
        edges = torch_laplacian(S0_temp)
        f_x = self.proj_x(torch.cat([x_temp, x_B, mask], dim=1))
        f_edge_x = self.proj_edge_x(torch.cat([x_temp, edges], dim=1))
        x_features = self.attention_x(f_x, f_edge_x)
        f_y = self.proj_y(torch.cat([y_temp, y_B, mask], dim=1))
        f_edge_y = self.proj_edge_y(torch.cat([y_temp, edges], dim=1))
        y_features = self.attention_y(f_y, f_edge_y)
        return x_features, y_features


class CoherenceAwareMutualAggregation(nn.Module):
    # output: 2 * dim
    def __init__(self, dim):
        super(CoherenceAwareMutualAggregation, self).__init__()

        self.x_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.y_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(2 * dim),
            nn.LeakyReLU(0.2, inplace=True),
            CBAM(gate_channels=2 * dim, reduction_ratio=dim, pool_types=('avg', 'max'), no_spatial=False),
            nn.Conv2d(2 * dim, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x_features, y_features):
        f_x = self.x_branch(x_features)
        f_y = self.y_branch(y_features)
        f_fuse = self.fuse(torch.cat([f_x, f_y], dim=1))
        return f_fuse


class MultiScaleCoherenceInjection(nn.Module):
    # output: 6
    def __init__(self, dim):
        super(MultiScaleCoherenceInjection, self).__init__()

        self.proj_conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )
        self.cross_attention1 = Chuncked_Cross_Attn_FM(in_channel=dim, r=dim // 2, subsample=True, grid=(32, 32))

        self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.proj_conv2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim * 2, use_dropout=False)
        )
        self.cross_attention2 = Chuncked_Cross_Attn_FM(in_channel=dim * 2, r=dim, subsample=True, grid=(16, 16))

        self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)
        self.proj_conv3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim * 4, use_dropout=False)
        )
        self.cross_attention3 = Chuncked_Cross_Attn_FM(in_channel=dim * 4, r=dim * 2, subsample=True, grid=(8, 8))

        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1)
        self.up_block1 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim * 2, use_dropout=False)
        )

        self.up2 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1)
        self.up_block2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False)
        )

        self.out_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(dim, use_dropout=False),
            nn.Conv2d(dim, 6, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, f_in, f_coherence1, f_coherence2, f_coherence3):
        f_proj1 = self.proj_conv1(f_in)
        f_in1 = self.cross_attention1(f_proj1, f_coherence1)

        f_proj2 = self.proj_conv2(self.down1(f_in1))
        f_in2 = self.cross_attention2(f_proj2, f_coherence2)

        f_proj3 = self.proj_conv3(self.down2(f_in2))
        f_in3 = self.cross_attention3(f_proj3, f_coherence3)

        f_out1 = self.up_block1(torch.cat((self.up1(f_in3), f_in2), dim=1))

        f_out2 = self.up_block2(torch.cat((self.up2(f_out1), f_in1), dim=1))

        f_out = self.out_block(f_out2)

        return f_out
