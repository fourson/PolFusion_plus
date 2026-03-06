import torch
import torch.nn as nn

from .res_block import ResBlock
from .channel_shuffle import ChannelShuffle
from .attention_block import AttentionBlock, DeformableAttentionBlock


class AutoencoderBackbone(nn.Module):
    """
        Autoencoder backbone
        input_nc -> dim
    """

    def __init__(self, input_nc, dim=32, n_downsampling=3, n_blocks=5, use_dropout=False):
        super(AutoencoderBackbone, self).__init__()

        sequence = [
            nn.Conv2d(input_nc, dim, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        ]

        dim = dim
        for i in range(n_downsampling):  # downsample the feature map
            sequence += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * dim),
                nn.ReLU(inplace=True)
            ]
            dim *= 2

        for i in range(n_blocks):  # ResBlock
            sequence += [
                ResBlock(dim, use_dropout)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            sequence += [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.ReLU(inplace=True)
            ]
            dim //= 2

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out


class SkipAutoencoderDownsamplingBlock(nn.Module):
    """
        Autoencoder downsampling block with skip links
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, use_dropout, use_channel_shuffle):
        super(SkipAutoencoderDownsamplingBlock, self).__init__()

        self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            )
        out_sequence = [
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]
        out_sequence += [nn.MaxPool2d(2)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        x_ = self.projection(x)
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class SkipAutoencoderUpsamplingBlock(nn.Module):
    """
        Autoencoder upsampling block with skip links
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, use_dropout, use_channel_shuffle):
        super(SkipAutoencoderUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled
        # in_channel2: channels from skip link
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            )
        out_sequence = [
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled
        # x2: skip link
        upsampled_x1 = self.upsample(x1)
        x_ = self.projection(torch.cat((x2, upsampled_x1), dim=1))
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class SkipAutoencoderBackbone(nn.Module):
    """
        Autoencoder backbone with skip links
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=5, use_dropout=False,
                 use_channel_shuffle=True):
        super(SkipAutoencoderBackbone, self).__init__()

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                SkipAutoencoderDownsamplingBlock(dim, 2 * dim, use_dropout, use_channel_shuffle)
            )
            dim *= 2

        res_blocks_seq = n_blocks * [ResBlock(dim, use_dropout)]
        self.dense_blocks = nn.Sequential(*res_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                SkipAutoencoderUpsamplingBlock(dim, dim // 2, dim // 2, use_dropout, use_channel_shuffle)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.dense_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat((x_, out), dim=1))
        return out


class AttentionAutoencoderUpsamplingBlock(nn.Module):
    """
        Attention autoencoder upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, use_dropout, use_channel_shuffle, deformable=False):
        super(AttentionAutoencoderUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled (gating signal)
        # in_channel2: channels from skip link (input signal)
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1)
        if deformable:
            self.attention = DeformableAttentionBlock(in_channel2, in_channel1 // 2, in_channel2)
        else:
            self.attention = AttentionBlock(in_channel2, in_channel1 // 2, in_channel2)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        if use_channel_shuffle:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                ChannelShuffle(groups=8),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
            )
        out_sequence = [
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled (gating signal)
        # x2: skip link (input signal)
        upsampled_x1 = self.upsample(x1)
        attentioned_x2 = self.attention(x2, upsampled_x1)
        x_ = self.projection(torch.cat((attentioned_x2, upsampled_x1), dim=1))
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class AttentionAutoencoderBackbone(nn.Module):
    """
        Attention autoencoder backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=5, use_dropout=False,
                 use_channel_shuffle=True):
        super(AttentionAutoencoderBackbone, self).__init__()

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_nc),
            nn.ReLU(inplace=True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                SkipAutoencoderDownsamplingBlock(dim, 2 * dim, use_dropout, use_channel_shuffle)
            )
            dim *= 2

        res_blocks_seq = n_blocks * [ResBlock(dim, use_dropout)]
        self.dense_blocks = nn.Sequential(*res_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                AttentionAutoencoderUpsamplingBlock(dim, dim // 2, dim // 2, use_dropout, use_channel_shuffle)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.dense_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat((x_, out), dim=1))
        return out
