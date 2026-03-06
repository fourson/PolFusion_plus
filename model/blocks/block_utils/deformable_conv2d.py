import torch
import torch.nn as nn
from torchvision import ops


class DeformableConv2d(nn.Module):
    def __init__(self, offset_in_channels, modulator_in_channels, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(offset_in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size,
                                     stride=stride, padding=self.padding, dilation=self.dilation, bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(modulator_in_channels, 1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size, stride=stride, padding=self.padding,
                                        dilation=self.dilation, bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=self.padding, dilation=self.dilation, bias=bias)

    def forward(self, x, offset_map, modulator_map):
        # offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width])
        # mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width])

        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(offset_map)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(modulator_map))
        # op = (n - (k * d - 1) + 2p / s)
        x = ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
                              padding=self.padding, mask=modulator, stride=self.stride, dilation=self.dilation)
        return x
