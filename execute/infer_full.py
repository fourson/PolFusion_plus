import os
import argparse
import importlib
import sys
import time

import torch
from tqdm import tqdm
import numpy as np


def infer_default():
    S0_out_dir = os.path.join(result_dir, 'S0_out')
    util.ensure_dir(S0_out_dir)
    DoP_out_dir = os.path.join(result_dir, 'DoP_out')
    util.ensure_dir(DoP_out_dir)
    AoP_out_dir = os.path.join(result_dir, 'AoP_out')
    util.ensure_dir(AoP_out_dir)
    time_all = 0

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            name = sample['name'][0][:-4]

            # get data and send them to GPU
            # (N, 3, H, W) GPU tensor
            L1 = sample['L1'].to(device)
            L2 = sample['L2'].to(device)
            L3 = sample['L3'].to(device)
            L4 = sample['L4'].to(device)
            S0_L, S1_L, S2_L, DoP_L, AoP_L = util.compute_Si_from_Ii(L1, L2, L3, L4)

            B1 = sample['B1'].to(device)
            B2 = sample['B2'].to(device)
            B3 = sample['B3'].to(device)
            B4 = sample['B4'].to(device)
            mask = sample['mask'].to(device)
            S0_B, S1_B, S2_B, DoP_B, AoP_B = util.compute_Si_from_Ii(B1, B2, B3, B4)

            x_B = S1_B / (S0_B + 1e-7)
            y_B = S2_B / (S0_B + 1e-7)

            # get network output
            # (N, 3, H, W) GPU tensor
            start_time = time.time()
            S0_temp, x_out, y_out, I1_out, I2_out, I3_out, I4_out = model(S0_B, mask, S0_L, S1_L, S2_L, x_B, y_B)
            S0_out, S1_out, S2_out, DoP_out, AoP_out = util.compute_Si_from_Ii(I1_out, I2_out, I3_out, I4_out)
            end_time = time.time()
            delta_time = end_time - start_time
            time_all += delta_time

            # save data
            S0_out_numpy = np.transpose(S0_out.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(S0_out_dir, f'{name}.npy'), S0_out_numpy)
            DoP_out_numpy = np.transpose(DoP_out.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(DoP_out_dir, f'{name}.npy'), DoP_out_numpy)
            AoP_out_numpy = np.transpose(AoP_out.squeeze().cpu().numpy(), (1, 2, 0))
            np.save(os.path.join(AoP_out_dir, f'{name}.npy'), AoP_out_numpy)

    print(time_all)


if __name__ == '__main__':
    MODULE = 'full'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', required=True, type=str, help='path to latest checkpoint')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--data_dir', required=True, type=str, help='dir of input data')
    parser.add_argument('--result_dir', required=True, type=str, help='dir to save result')
    parser.add_argument('--data_loader_type', default='WithoutGroundTruthDataLoader', type=str,
                        help='which data loader to use')
    subparsers = parser.add_subparsers(help='which func to run', dest='func')

    # add subparsers and their args for each func
    subparser_default = subparsers.add_parser("default")

    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to PATH
    from utils import util

    # load checkpoint
    checkpoint = torch.load(args.resume)
    config = checkpoint['config']
    assert config['module'] == MODULE

    # setup data_loader instances
    # we choose batch_size=1(default value)
    # module_data = importlib.import_module('.data_loader_' + MODULE, package='data_loader')
    module_data = importlib.import_module('.data_loader', package='data_loader')  # share the same dataloader
    data_loader_class = getattr(module_data, args.data_loader_type)
    data_loader = data_loader_class(data_dir=args.data_dir)

    # build model architecture
    module_arch = importlib.import_module('.model_' + MODULE, package='model')
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])

    # set the model to validation mode
    model.eval()

    # ensure result_dir
    result_dir = args.result_dir
    util.ensure_dir(result_dir)

    # run the selected func
    if args.func == 'default':
        infer_default()
    else:
        infer_default()
