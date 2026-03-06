import torch
import torch.nn as nn
import torch.nn.functional as F


def logsumexp_2d(tensor):
    N, C, H, W = tensor.shape
    tensor_flatten = tensor.view(N, C, -1)  # (N, C, H*W)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)  # (N, C, 1)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()  # (N, C, 1)
    return outputs


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        N, C, H, W = x.shape
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = x.view(N, C, -1).mean(-1)  # (N, C)
                channel_att_raw = self.mlp(avg_pool)  # (N, C)
            elif pool_type == 'max':
                max_pool = x.view(N, C, -1).max(-1)[0]  # (N, C)
                channel_att_raw = self.mlp(max_pool)  # (N, C)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, kernel_size=(H, W), stride=(H, W))  # (N, C, 1, 1)
                channel_att_raw = self.mlp(lp_pool.view(N, C))  # (N, C)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)  # (N, C, 1)
                channel_att_raw = self.mlp(lse_pool.view(N, C))  # (N, C)
            else:
                raise Exception(f'no such pool_type: {pool_type}')

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum)[:, :, None, None]  # (N, C, 1, 1)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_channel_max = x.max(1, keepdim=True)[0]  # (N, 1, H, W)
        x_channel_mean = x.mean(1, keepdim=True)  # (N, 1, H, W)
        x_compress = torch.cat((x_channel_max, x_channel_mean), dim=1)  # (N, 2, H, W)
        x_out = self.spatial(x_compress)  # (N, 1, H, W)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max'), no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
