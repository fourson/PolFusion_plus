from pathlib import Path
import shutil

import numpy as np
import cv2

from scripts.script_utils.ExposureAdjustment import ExposureAdjuster


def read_img(path, rgb=True):
    img = cv2.imread(str(path), -1)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    return img


def write_img(path, img, rgb=True):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    cv2.imwrite(str(path), img)


class Maker:
    """
        output for test:
            I1, I2, I3, I4: GT (long-exposure t_l)
            L1, L2, L3, L4: noisy (short-exposure t_s, normalized to GT's exposure)
            B1, B2, B3, B4: blurry (over-exposure t_o, normalized to GT's exposure)
            B1_ori, B2_ori, B3_ori, B4_ori: blurry (over-exposure t_o, without normalizing)
            e_factor: a number used for normalizing the exposure from B1234_ori to B1234

        making from the dataset of NeurIPS24:
            I1, I2, I3, I4: GT (long-exposure t_l)
            L1, L2, L3, L4: noisy (short-exposure t_s, normalized to GT's exposure)
            B1, B2, B3, B4: blurry (long-exposure t_l, the same exposure as GT)

        goal: adjusting the exposure time of B1234 from t_l to t_o for simulating the saturation of pixels

        * all images are in [0, 1+]
        * images are in [R, G, B] manner
    """

    def __init__(self, base_dir, out_base_dir, exposure_adjuster_args):
        self.base_dir = Path(base_dir)
        self.out_base_dir = Path(out_base_dir)

        self.exposure_adjuster_args = exposure_adjuster_args

        self.img_names = [p.name for p in (self.base_dir / 'I1').glob('*.png')]

    def make(self):
        fail_list = []
        for img_name in self.img_names:

            # handle the images with changes (B1234) and the new data (B1234_ori and e_factor)
            pol_imgs = [read_img(self.base_dir / f'B{i}' / img_name, rgb=True) for i in range(1, 5)]
            exposure_adjuster = ExposureAdjuster(pol_imgs, **self.exposure_adjuster_args)
            status = exposure_adjuster.adjust()

            if status:
                B1234_ori = exposure_adjuster.adjusted_pol_imgs
                B1234 = [Bi_ori / exposure_adjuster.e_factor for Bi_ori in B1234_ori]

                for i in range(1, 5):
                    dst_dir = self.out_base_dir / f'B{i}'
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = dst_dir / img_name
                    write_img(dst_path, B1234[i - 1], rgb=True)

                    dst_dir = self.out_base_dir / f'B{i}_ori'
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = dst_dir / img_name
                    write_img(dst_path, B1234_ori[i - 1], rgb=True)

                    dst_dir = self.out_base_dir / 'e_factor'
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = dst_dir / f'{img_name[:-4]}.npy'
                    np.save(str(dst_path), exposure_adjuster.e_factor)

                # handle the images without change (I1234 and L1234)
                for i in range(1, 5):
                    src_path = self.base_dir / f'I{i}' / img_name
                    dst_dir = self.out_base_dir / f'I{i}'
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = dst_dir / img_name
                    shutil.copyfile(src_path, dst_path)

                    src_path = self.base_dir / f'L{i}' / img_name
                    dst_dir = self.out_base_dir / f'L{i}'
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst_path = dst_dir / img_name
                    shutil.copyfile(src_path, dst_path)
            else:
                fail_list.append(img_name)

            print(f'{img_name}: success={status}, {exposure_adjuster.log}')

        print(f'failed images: {fail_list}')


if __name__ == '__main__':
    train_maker_args = {
        'base_dir': '../data_NeurIPS24/train',
        'out_base_dir': '../data/train',
        'exposure_adjuster_args': {
            'bpr_ubound': 0.15,
            'bpr_lbound': 0.025,
            'iter_max': 40
        }
    }
    train_maker = Maker(**train_maker_args)
    train_maker.make()

    test_maker_args = {
        'base_dir': '../data_NeurIPS24/test',
        'out_base_dir': '../data/test',
        'exposure_adjuster_args': {
            'bpr_ubound': 0.15,
            'bpr_lbound': 0.025,
            'iter_max': 40
        }
    }
    test_maker = Maker(**test_maker_args)
    test_maker.make()
