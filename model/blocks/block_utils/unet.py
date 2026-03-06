import torch
import torch.nn as nn

from .attention_block import AttentionBlock, DeformableAttentionBlock


class UnetDoubleConvBlock(nn.Module):
    """
        Unet double Conv block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, use_dropout, mode='default'):
        super(UnetDoubleConvBlock, self).__init__()

        self.mode = mode

        if self.mode == 'default':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'bottleneck':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1),
                nn.InstanceNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'res-bottleneck':
            self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
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
        else:
            raise NotImplementedError('mode [%s] is not found' % self.mode)

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        if self.mode == 'res-bottleneck':
            x_ = self.projection(x)
            out = self.out_block(x_ + self.bottleneck(x_))
        else:
            out = self.out_block(self.model(x))
        return out


class UnetDownsamplingBlock(nn.Module):
    """
        Unet downsampling block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, use_dropout, use_conv, mode='default'):
        super(UnetDownsamplingBlock, self).__init__()

        downsampling_layers = list()
        if use_conv:
            downsampling_layers += [
                nn.Conv2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(in_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            downsampling_layers += [nn.MaxPool2d(2)]

        self.model = nn.Sequential(
            nn.Sequential(*downsampling_layers),
            UnetDoubleConvBlock(in_channel, out_channel, use_dropout, mode)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UnetUpsamplingBlock(nn.Module):
    """
        Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, use_dropout, mode='default'):
        super(UnetUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled
        # in_channel2: channels from skip link
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, use_dropout, mode)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled
        # x2: skip link
        out = torch.cat([x2, self.upsample(x1)], dim=1)
        out = self.double_conv(out)
        return out


class UnetBackbone(nn.Module):
    """
        Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=True, use_dropout=False,
                 mode='default'):
        super(UnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, use_dropout, mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, use_dropout, use_conv_to_downsample, mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                UnetUpsamplingBlock(dim, dim // 2, dim // 2, use_dropout, mode)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out


class AttentionUnetUpsamplingBlock(nn.Module):
    """
        attention Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, use_dropout, mode='default', deformable=False):
        super(AttentionUnetUpsamplingBlock, self).__init__()
        # in_channel1: channels from the signal to be upsampled (gating signal)
        # in_channel2: channels from skip link (input signal)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(in_channel1 // 2),
            nn.ReLU(inplace=True)
        )
        if deformable:
            self.attention = DeformableAttentionBlock(in_channel2, in_channel1 // 2, in_channel2)
        else:
            self.attention = AttentionBlock(in_channel2, in_channel1 // 2, in_channel2)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, use_dropout, mode)

    def forward(self, x1, x2):
        # x1: the signal to be upsampled (gating signal)
        # x2: skip link (input signal)
        upsampled_x1 = self.upsample(x1)
        attentioned_x2 = self.attention(x2, upsampled_x1)
        out = torch.cat([attentioned_x2, upsampled_x1], dim=1)
        out = self.double_conv(out)
        return out


class AttentionUnetBackbone(nn.Module):
    """
        attention Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=False, use_dropout=False,
                 mode='default', deformable=False):
        super(AttentionUnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, use_dropout, mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, use_dropout, use_conv_to_downsample, mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                AttentionUnetUpsamplingBlock(dim, dim // 2, dim // 2, use_dropout, mode, deformable)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out
