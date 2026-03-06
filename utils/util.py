import os

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import math


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda_tag):
    if lr_lambda_tag == 'default':
        # keep the same
        return lambda epoch: 1
    else:
        raise NotImplementedError('lr_lambda_tag [%s] is not found' % lr_lambda_tag)


def convert_DoP(DoP_tensor):
    DoP_tensor = DoP_tensor.cpu()
    # convert DoP tensor (N, 3, H, W) for visualization
    converted_tensor = torch.zeros(DoP_tensor.shape, dtype=torch.float32, requires_grad=False)
    for i in range(converted_tensor.shape[0]):
        DoP = DoP_tensor[i].numpy().transpose((1, 2, 0))  # (H, W, 3)
        DoP = cv2.applyColorMap(cv2.cvtColor(np.uint8(DoP * 255), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)  # (H, W, 3)
        DoP = np.float32(cv2.cvtColor(DoP, cv2.COLOR_BGR2RGB)) / 255
        converted_tensor[i] = torch.from_numpy(DoP.transpose((2, 0, 1)))  # (3, H, W)
    return converted_tensor


def convert_AoP(AoP_tensor):
    AoP_tensor = AoP_tensor.cpu()
    # convert AoP tensor (N, 3, H, W) for visualization
    AoP_tensor /= torch.pi
    converted_tensor = torch.zeros(AoP_tensor.shape, dtype=torch.float32, requires_grad=False)
    for i in range(converted_tensor.shape[0]):
        AoP = AoP_tensor[i].numpy().transpose((1, 2, 0))  # (H, W, 3)
        AoP = cv2.applyColorMap(cv2.cvtColor(np.uint8(AoP * 255), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)  # (H, W, 3)
        AoP = np.float32(cv2.cvtColor(AoP, cv2.COLOR_BGR2RGB)) / 255
        converted_tensor[i] = torch.from_numpy(AoP.transpose((2, 0, 1)))  # (3, H, W)
    return converted_tensor


def convert_to_colormap(img_tensor):
    # convert the grayscale image tensor(N, H, W) to colormap tensor(N, 3, H, W) for visualization
    N, H, W = img_tensor.shape
    colormap_tensor = torch.zeros((N, 3, H, W), dtype=torch.float32, requires_grad=False)
    for i in range(N):
        img = img_tensor[i].numpy()  # (H, W)
        img = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)  # (H, W, 3) in BGR uint8
        img = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255  # (3, H, W) in RGB float32
        colormap_tensor[i] = torch.from_numpy(img)
    return colormap_tensor


@torch.jit.script
def torch_laplacian(img_tensor):
    # (N, C, H, W) image tensor -> (N, C, H, W) edge tensor, the same as cv2.Laplacian
    padded = F.pad(img_tensor, pad=[1, 1, 1, 1], mode='reflect')
    return padded[:, :, 2:, 1:-1] + padded[:, :, 0:-2, 1:-1] + padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, 0:-2] - \
           4 * img_tensor


@torch.jit.script
def convolve_with_kernel(img_tensor, kernel_tensor):
    # (N, C, H, W) image tensor and (h, w) kernel tensor -> (N, C, H, W) output tensor
    # kernel_tensor should be a buffer in the model to avoid runtime error when DataParallel
    # eg:
    # self.register_buffer('laplace_kernel',
    #                      torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, requires_grad=False),
    #                      persistent=False
    #                      )
    N, C, H, W = img_tensor.shape
    h, w = kernel_tensor.shape
    return F.conv2d(
        F.pad(img_tensor.reshape(N * C, 1, H, W),
              pad=[(h - 1) // 2, (h - 1) // 2, (w - 1) // 2, (w - 1) // 2],
              mode='reflect'
              ),
        kernel_tensor.reshape(1, 1, h, w)
    ).reshape(N, C, H, W)


@torch.jit.script
def compute_Si_from_Ii(I1, I2, I3, I4):
    S0 = (I1 + I2 + I3 + I4) / 2  # I
    S1 = I3 - I1  # I*p*cos(2*theta)
    S2 = I4 - I2  # I*p*sin(2*theta)
    DoP = torch.clamp(torch.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), min=0, max=1)  # in [0, 1]
    AoP = torch.atan2(S2, S1) / 2  # in [-pi/2, pi/2]
    AoP = (AoP < 0) * math.pi + AoP  # convert to [0, pi] by adding pi to negative values
    return S0, S1, S2, DoP, AoP


@torch.jit.script
def compute_Ii_from_Si(S0, S1, S2):
    I1 = (S0 - S1) / 2
    I2 = (S0 - S2) / 2
    I3 = (S0 + S1) / 2
    I4 = (S0 + S2) / 2
    DoP = torch.clamp(torch.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), min=0, max=1)  # in [0, 1]
    AoP = torch.atan2(S2, S1) / 2  # in [-pi/2, pi/2]
    AoP = (AoP < 0) * math.pi + AoP  # convert to [0, pi] by adding pi to negative values
    return I1, I2, I3, I4, DoP, AoP


@torch.jit.script
def compute_stokes(S0, DoP, AoP):
    S1 = S0 * DoP * torch.cos(2 * AoP)
    S2 = S0 * DoP * torch.sin(2 * AoP)
    return S0, S1, S2
