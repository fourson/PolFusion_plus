import torch.nn as nn


class ChannelShuffle(nn.Module):
    def __init__(self, groups=8):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, H, W = x.shape
        return x.reshape(N, self.groups, C // self.groups, H, W).transpose(1, 2).reshape(N, C, H, W)
