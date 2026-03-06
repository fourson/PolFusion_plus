import torch
import torch.nn.functional as F

from .l12 import l2


def torch_laplacian(img_tensor):
    # (N, C, H, W) image tensor -> (N, C, H, W) edge tensor, the same as cv2.Laplacian
    padded = F.pad(img_tensor, pad=[1, 1, 1, 1], mode='reflect')
    return padded[:, :, 2:, 1:-1] + padded[:, :, 0:-2, 1:-1] + padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, 0:-2] - \
           4 * img_tensor


def gradient(pred, gt):
    gradient_pred = torch.abs(torch_laplacian(pred))
    gradient_gt = torch.abs(torch_laplacian(gt))
    return l2(gradient_pred, gradient_gt)
