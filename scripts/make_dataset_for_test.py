import os
import fnmatch

import numpy as np
import cv2

from scripts.script_utils.CameraMotion import TrajectoryGenerator, PatchWiseTrajectoryGenerator, \
    blur_image_by_patch_wise_trajectory


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


class Maker:
    """
        output for test: I1, I2, I3, I4, B1, B2, B3, B4, L1, L2, L3, L4
        * all images are in [0, 1+]
        * images are in [R, G, B] manner
    """

    def __init__(self, base_dir, out_base_dir, base_trajectory_args, patch_wise_trajectory_args, trajectory_num=10,
                 patch_size=16, pre_amp=10):
        self.base_dir = base_dir
        self.out_base_dir = out_base_dir

        self.base_trajectory_args = base_trajectory_args
        self.patch_wise_trajectory_args = patch_wise_trajectory_args
        self.trajectory_num = trajectory_num
        self.patch_size = patch_size

        self.pre_amp = pre_amp

        self.names = [file_name[:-4] for file_name in
                      sorted(fnmatch.filter(os.listdir(os.path.join(self.base_dir, 'L1')), '*.png'))]
        print(f'Pre-processing the following files: {self.names}')

    def _generate_random_trajectory(self):
        trajectory_fail_cnt = 0
        while (base_trajectory := TrajectoryGenerator(**self.base_trajectory_args).generate()) is None:
            print('Sanity check fail, regenerate a trajectory')
            trajectory_fail_cnt += 1
        if trajectory_fail_cnt > 0:
            print(f'A trajectory is successfully generated after failing {trajectory_fail_cnt} times.')

        patch_wise_trajectory_fail_cnt = 0
        while (
                patch_wise_trajectory := PatchWiseTrajectoryGenerator(base_trajectory,
                                                                      **self.patch_wise_trajectory_args).generate()
        ) is None:
            print('Sanity check fail, regenerate a patch_wise_trajectory')
            patch_wise_trajectory_fail_cnt += 1
        if patch_wise_trajectory_fail_cnt > 0:
            print(
                f'A patch_wise_trajectory is successfully generated after failing {patch_wise_trajectory_fail_cnt} times.')
        return base_trajectory, patch_wise_trajectory

    def __len__(self):
        return len(self.names) * self.trajectory_num

    def make(self, save_gt=True):
        cnt = 1
        for name in self.names:
            print(f'Fetching {name}')
            # all images should be resized to 512 * 512 first
            I_list = [
                cv2.resize(
                    read_img(
                        os.path.join(self.base_dir, f'I{i}', f'{name}0.png')
                    ), (512, 512), interpolation=cv2.INTER_LINEAR
                ) for i in range(1, 5)
            ]  # note that in PLIE dataset the image names in Li and Ii are not the same (with a "0" difference)
            L_list = [
                cv2.resize(
                    read_img(
                        os.path.join(self.base_dir, f'L{i}', f'{name}.png')
                    ) * self.pre_amp, (512, 512), interpolation=cv2.INTER_LINEAR
                ) for i in range(1, 5)
            ]

            for i in range(self.trajectory_num):
                name_i = f'{name}_{i:0>2d}'
                # blur
                print(f'Generating {i}-th trajectory')
                base_trajectory, patch_wise_trajectory = self._generate_random_trajectory()
                B_list = [
                    blur_image_by_patch_wise_trajectory(I, patch_wise_trajectory, self.patch_size) for I in I_list
                ]

                if save_gt:
                    # save gt only for the first kernel to save space
                    data_item = {
                        'I1': I_list[0],
                        'I2': I_list[1],
                        'I3': I_list[2],
                        'I4': I_list[3],
                        'B1': B_list[0],
                        'B2': B_list[1],
                        'B3': B_list[2],
                        'B4': B_list[3],
                        'L1': L_list[0],
                        'L2': L_list[1],
                        'L3': L_list[2],
                        'L4': L_list[3],
                    }
                else:
                    data_item = {
                        'B1': B_list[0],
                        'B2': B_list[1],
                        'B3': B_list[2],
                        'B4': B_list[3],
                        'L1': L_list[0],
                        'L2': L_list[1],
                        'L3': L_list[2],
                        'L4': L_list[3],
                    }

                # write to files
                for tag, data in data_item.items():
                    out_subdir = os.path.join(self.out_base_dir, tag)
                    ensure_dir(out_subdir)
                    write_img(os.path.join(out_subdir, name_i + '.png'), data, rgb=True)
                print(f'==========Finishing {cnt}/{len(self)}==========')
                cnt += 1


if __name__ == '__main__':
    maker_args = {
        'base_dir': '../raw_images/data_test_temp',
        'out_base_dir': '../data_NeurIPS24/test',
        'base_trajectory_args': {
            'canvas': 32,
            'samples': 13,
            'expl': 0.075,
            'big_expl_count_max': None,
            'max_len': None,
            'downsampling': 1,
        },
        'patch_wise_trajectory_args': {
            'canvas': 32,
            'patch_number': (16, 16),
            'z_center': None,
            'z_translation_size': None,
            'z_rotation_angle': None,
            'z_rotation_size': None,
            'downsampling': 1,
        },
        'trajectory_num': 10,
        'patch_size': 32
    }

    maker = Maker(**maker_args)
    maker.make(save_gt=True)
