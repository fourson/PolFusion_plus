import torch


def total_variation(pred, weight=1):
    N, C, H, W = pred.size()
    count_h = C * (H - 1) * W
    count_w = C * H * (W - 1)
    h_tv = torch.pow((pred[:, :, 1:, :] - pred[:, :, :H - 1, :]), 2).sum()
    w_tv = torch.pow((pred[:, :, :, 1:] - pred[:, :, :, :W - 1]), 2).sum()
    return weight * 2 * (h_tv / count_h + w_tv / count_w) / N
