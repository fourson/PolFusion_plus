import os
import fnmatch

import numpy as np
import cv2


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_img(path, rgb=True):
    img = cv2.imread(path, -1)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    return img


def write_img(path, img, rgb=True):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    cv2.imwrite(path, img)


def compute_Si_from_Ii(I1, I2, I3, I4):
    S0 = (I1 + I2 + I3 + I4) / 2  # I
    S1 = I3 - I1  # I*p*cos(2*theta)
    S2 = I4 - I2  # I*p*sin(2*theta)
    DoP = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), a_min=0, a_max=1)  # in [0, 1]
    AoP = np.arctan2(S2, S1) / 2  # in [-pi/2, pi/2]
    AoP = (AoP < 0) * np.pi + AoP  # convert to [0, pi] by adding pi to negative values
    return S0, S1, S2, DoP, AoP


base_dir = '../data/test'
names = [file_name[:-4] for file_name in sorted(fnmatch.filter(os.listdir(os.path.join(base_dir, 'L1')), '*.png'))]

for name in names:
    print(name)
    # for L1234: L_S0, L_DoP, L_AoP
    L1 = read_img(os.path.join(base_dir, 'L1', f'{name}.png'), rgb=True)
    L2 = read_img(os.path.join(base_dir, 'L2', f'{name}.png'), rgb=True)
    L3 = read_img(os.path.join(base_dir, 'L3', f'{name}.png'), rgb=True)
    L4 = read_img(os.path.join(base_dir, 'L4', f'{name}.png'), rgb=True)
    L_S0, L_S1, L_S2, L_DoP, L_AoP = compute_Si_from_Ii(L1, L2, L3, L4)

    L_S0_dir = os.path.join(base_dir, 'L_S0')
    ensure_dir(L_S0_dir)
    np.save(os.path.join(L_S0_dir, f'{name}.npy'), L_S0)
    L_DoP_dir = os.path.join(base_dir, 'L_DoP')
    ensure_dir(L_DoP_dir)
    np.save(os.path.join(L_DoP_dir, f'{name}.npy'), L_DoP)
    L_AoP_dir = os.path.join(base_dir, 'L_AoP')
    ensure_dir(L_AoP_dir)
    np.save(os.path.join(L_AoP_dir, f'{name}.npy'), L_AoP)

    # for B1234: B_S0, B_DoP, B_AoP
    B1 = read_img(os.path.join(base_dir, 'B1', f'{name}.png'), rgb=True)
    B2 = read_img(os.path.join(base_dir, 'B2', f'{name}.png'), rgb=True)
    B3 = read_img(os.path.join(base_dir, 'B3', f'{name}.png'), rgb=True)
    B4 = read_img(os.path.join(base_dir, 'B4', f'{name}.png'), rgb=True)
    B_S0, B_S1, B_S2, B_DoP, B_AoP = compute_Si_from_Ii(B1, B2, B3, B4)

    B_S0_dir = os.path.join(base_dir, 'B_S0')
    ensure_dir(B_S0_dir)
    np.save(os.path.join(B_S0_dir, f'{name}.npy'), B_S0)
    B_DoP_dir = os.path.join(base_dir, 'B_DoP')
    ensure_dir(B_DoP_dir)
    np.save(os.path.join(B_DoP_dir, f'{name}.npy'), B_DoP)
    B_AoP_dir = os.path.join(base_dir, 'B_AoP')
    ensure_dir(B_AoP_dir)
    np.save(os.path.join(B_AoP_dir, f'{name}.npy'), B_AoP)

    # for B1234_ori: only compute B_S0_ori, since B_DoP_ori==B_DoP and B_AoP_ori==B_AoP
    B1_ori = read_img(os.path.join(base_dir, 'B1_ori', f'{name}.png'), rgb=True)
    B2_ori = read_img(os.path.join(base_dir, 'B2_ori', f'{name}.png'), rgb=True)
    B3_ori = read_img(os.path.join(base_dir, 'B3_ori', f'{name}.png'), rgb=True)
    B4_ori = read_img(os.path.join(base_dir, 'B4_ori', f'{name}.png'), rgb=True)
    B_S0_ori, B_S1_ori, B_S2_ori, B_DoP_ori, B_AoP_ori = compute_Si_from_Ii(B1_ori, B2_ori, B3_ori, B4_ori)

    B_S0_ori_dir = os.path.join(base_dir, 'B_S0_ori')
    ensure_dir(B_S0_ori_dir)
    np.save(os.path.join(B_S0_ori_dir, f'{name}.npy'), B_S0_ori)

    # for I1234: I_S0, I_DoP, I_AoP
    I1 = read_img(os.path.join(base_dir, 'I1', f'{name}.png'), rgb=True)
    I2 = read_img(os.path.join(base_dir, 'I2', f'{name}.png'), rgb=True)
    I3 = read_img(os.path.join(base_dir, 'I3', f'{name}.png'), rgb=True)
    I4 = read_img(os.path.join(base_dir, 'I4', f'{name}.png'), rgb=True)
    I_S0, I_S1, I_S2, I_DoP, I_AoP = compute_Si_from_Ii(I1, I2, I3, I4)

    I_S0_dir = os.path.join(base_dir, 'I_S0')
    ensure_dir(I_S0_dir)
    np.save(os.path.join(I_S0_dir, f'{name}.npy'), I_S0)
    I_DoP_dir = os.path.join(base_dir, 'I_DoP')
    ensure_dir(I_DoP_dir)
    np.save(os.path.join(I_DoP_dir, f'{name}.npy'), I_DoP)
    I_AoP_dir = os.path.join(base_dir, 'I_AoP')
    ensure_dir(I_AoP_dir)
    np.save(os.path.join(I_AoP_dir, f'{name}.npy'), I_AoP)
