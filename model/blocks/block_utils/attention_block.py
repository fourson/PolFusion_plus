import torch.nn as nn

from .deformable_conv2d import DeformableConv2d


class AttentionBlock(nn.Module):
    """
        attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t):
        # in_channel_x: input signal channels
        # in_channel_g: gating signal channels
        super(AttentionBlock, self).__init__()
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channel_x, channel_t, kernel_size=1, stride=1),
            nn.InstanceNorm2d(channel_t)
        )

        self.g_block = nn.Sequential(
            nn.Conv2d(in_channel_g, channel_t, kernel_size=1, stride=1),
            nn.InstanceNorm2d(channel_t)
        )

        self.t_block = nn.Sequential(
            nn.Conv2d(channel_t, 1, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: (N, in_channel_x, H, W)
        # g: (N, in_channel_g, H, W)
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t_in = self.relu(x_out + g_out)  # (N, 1, H, W)
        attention_map = self.t_block(t_in)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)


class DeformableAttentionBlock(nn.Module):
    """
        deformable attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t):
        # in_channel_x: input signal channels
        # in_channel_g: gating signal channels
        super(DeformableAttentionBlock, self).__init__()
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channel_x, channel_t, kernel_size=1, stride=1),
            nn.InstanceNorm2d(channel_t)
        )

        self.g_block = nn.Sequential(
            nn.Conv2d(in_channel_g, channel_t, kernel_size=1, stride=1),
            nn.InstanceNorm2d(channel_t)
        )

        self.relu = nn.ReLU(inplace=True)

        self.offset_conv = nn.Conv2d(channel_t, channel_t, kernel_size=3, stride=1, padding=1)
        self.modulator_conv = nn.Conv2d(channel_t, channel_t, kernel_size=3, stride=1, padding=1)
        self.filter_conv = DeformableConv2d(offset_in_channels=channel_t, modulator_in_channels=channel_t,
                                            in_channels=channel_t, out_channels=channel_t, kernel_size=3, stride=1,
                                            padding=1, dilation=1, bias=False)
        self.t_block = nn.Sequential(
            nn.Conv2d(channel_t, 1, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        # x: (N, in_channel_x, H, W)
        # g: (N, in_channel_g, H, W)
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t = self.relu(x_out + g_out)  # (N, 1, H, W)

        offset = self.offset_conv(t)
        modulator = self.modulator_conv(t)
        t = self.filter_conv(t, offset, modulator)

        attention_map = self.t_block(t)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)
