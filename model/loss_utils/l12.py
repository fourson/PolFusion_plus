import torch.nn.functional as F


def l1(pred, gt):
    return F.l1_loss(pred, gt)


def l2(pred, gt):
    return F.mse_loss(pred, gt)
