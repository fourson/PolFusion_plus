import torch
import torch.nn as nn


class Cross_Attn_C(nn.Module):
    """ Cross attention Layer for Channel dimension"""

    def __init__(self, in_dim, r=8):
        super(Cross_Attn_C, self).__init__()
        self.channel_latent = in_dim // r
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(in_channels=self.channel_latent, out_channels=in_dim, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q):
        """
            inputs :
                x, q : input feature maps(B x C x H x W)
            returns :
                out : self attention value + input feature
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(q).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_key: reshape to B x c x N, N = H x W
        proj_key = self.key_conv(x).view(batchsize, -1, height * width)
        # energy: B x c x c
        energy = torch.bmm(proj_key, proj_query)
        # attention: B x c x c
        attention = self.softmax(energy)
        # proj_value: B x c x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # attention_out: B x c x N
        attention_out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        # out: B x C x H x W
        out = self.out_conv(attention_out.view(batchsize, self.channel_latent, height, width))

        out = self.gamma * out + x
        return out


class Cross_Attn_FM(nn.Module):
    """ Cross attention Layer for Feature Map dimension"""

    def __init__(self, in_dim, r=8, subsample=True):
        super(Cross_Attn_FM, self).__init__()
        self.channel_latent = in_dim // r
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=self.channel_latent, kernel_size=1, stride=1)
        self.out_conv = nn.Conv2d(in_channels=self.channel_latent, out_channels=in_dim, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        if subsample:
            self.key_conv = nn.Sequential(
                self.key_conv,
                nn.MaxPool2d(2)
            )
            self.value_conv = nn.Sequential(
                self.value_conv,
                nn.MaxPool2d(2)
            )

    def forward(self, x, q):
        """
            inputs :
                x, q : input feature maps(B x C x H x W)
            returns :
                out : self attention value + input feature
        """
        batchsize, C, height, width = x.size()
        c = self.channel_latent
        # proj_query: reshape to B x N x c, N = H x W
        proj_query = self.query_conv(q).view(batchsize, c, -1).permute(0, 2, 1)
        # proj_key: reshape to B x c x N_, N_ = H_ x W_
        proj_key = self.key_conv(x).view(batchsize, c, -1)
        # energy: B x N x N_, N = H x W, N_ = H_ x W_
        energy = torch.bmm(proj_query, proj_key)
        # attention: B x N_ x N, N = H x W, N_ = H_ x W_
        attention = self.softmax(energy).permute(0, 2, 1)
        # proj_value: B x c x N_, N_ = H_ x W_
        proj_value = self.value_conv(x).view(batchsize, c, -1)
        # attention_out: B x c x N, N = H x W
        attention_out = torch.bmm(proj_value, attention)
        # out: B x C x H x W
        out = self.out_conv(attention_out.view(batchsize, c, height, width))

        out = self.gamma * out + x
        return out


class Chuncked_Cross_Attn_FM(nn.Module):
    """
        in_channel -> in_channel
    """

    def __init__(self, in_channel, r=8, subsample=True, grid=(8, 8)):
        super(Chuncked_Cross_Attn_FM, self).__init__()

        self.self_attn_fm = Cross_Attn_FM(in_channel, r=r, subsample=subsample)
        self.grid = grid

    def forward(self, x, q):
        N, C, H, W = x.shape
        chunk_size_H, chunk_size_W = H // self.grid[0], W // self.grid[1]
        x_ = x.reshape(N, C, self.grid[0], chunk_size_H, self.grid[1], chunk_size_W).permute(0, 2, 4, 1, 3, 5).reshape(
            N * self.grid[0] * self.grid[1], C, chunk_size_H, chunk_size_W)
        q_ = q.reshape(N, C, self.grid[0], chunk_size_H, self.grid[1], chunk_size_W).permute(0, 2, 4, 1, 3, 5).reshape(
            N * self.grid[0] * self.grid[1], C, chunk_size_H, chunk_size_W)
        output = self.self_attn_fm(x_, q_).reshape(N, self.grid[0], self.grid[1], C, chunk_size_H,
                                                   chunk_size_W).permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
        return output
