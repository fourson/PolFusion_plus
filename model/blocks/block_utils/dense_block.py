import torch
import torch.nn as nn


class DenseCell(nn.Module):
    def __init__(self, in_channel, growth_rate, kernel_size=3):
        super(DenseCell, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=growth_rate, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return torch.cat((x, self.conv_block(x)), dim=1)


class DenseBlock(nn.Module):
    """
        DenseBlock using bottleneck structure
        in_channel -> in_channel
    """

    def __init__(self, in_channel, growth_rate=32, n_blocks=3):
        super(DenseBlock, self).__init__()

        sequence = nn.ModuleList()

        dim = in_channel
        for i in range(n_blocks):
            sequence.append(DenseCell(dim, growth_rate))
            dim += growth_rate

        self.dense_cells = nn.Sequential(*sequence)
        self.fusion = nn.Conv2d(in_channels=dim, out_channels=in_channel, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.fusion(self.dense_cells(x)) + x
