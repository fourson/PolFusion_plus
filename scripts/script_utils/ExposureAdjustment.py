import numpy as np


class ExposureAdjuster:
    """
        Adjust the exposure of polarized images
            search an appropriate e_factor so that the bad_pixel rate is in [bpr_lbound, bpr_ubound]

        :pol_imgs: four polarized images to be adjusted
        :param bpr_ubound: upper ubound of the bad_pixel rate
        :param bpr_lbound: lower ubound of the bad_pixel rate
        :param iter_max: maximum iteration number for searching
    """

    def __init__(self, pol_imgs, bpr_ubound=0.1, bpr_lbound=0.025, iter_max=20):
        self.pol_imgs = pol_imgs

        # we hope the bad_pixel rate is between [bpr_lbound, bpr_ubound]
        self.bpr_ubound = bpr_ubound
        self.bpr_lbound = bpr_lbound

        # set the search space of e_factor
        self.e_factor_ubound = 100
        self.e_factor_lbound = 1

        # set the maximum number of iterations
        self.iter_max = iter_max

        # results
        self.e_factor = None
        self.success = None
        self.adjusted_pol_imgs = None

        # for log
        self.iter_cnt = None
        self.final_bpr = None
        self.log = None

    @property
    def _bpr(self):
        th = 1.0
        img1, img2, img3, img4 = self.pol_imgs

        # 0/1: saturated/not saturated
        ns_mask1 = np.float32((img1 * self.e_factor) < th)
        ns_mask2 = np.float32((img2 * self.e_factor) < th)
        ns_mask3 = np.float32((img3 * self.e_factor) < th)
        ns_mask4 = np.float32((img4 * self.e_factor) < th)

        # 0/1: not bad/bad
        mask_bp = np.float32((ns_mask1 + ns_mask3) <= 1) * np.float32((ns_mask2 + ns_mask4) <= 1)
        mask_bp_ = (np.sum(mask_bp, axis=-1) >= 1)

        bpr = np.sum(mask_bp_) / mask_bp_.size
        return bpr

    def _search_e_factor(self):
        left, right = self.e_factor_lbound, self.e_factor_ubound

        for iter_cnt in range(self.iter_max):
            self.e_factor = (left + right) / 2
            bpr = self._bpr

            if bpr > self.bpr_ubound:
                right = self.e_factor
            elif bpr < self.bpr_lbound:
                left = self.e_factor
            else:
                self.iter_cnt, self.final_bpr = iter_cnt, bpr
                return True

        return False

    def adjust(self):
        status = self._search_e_factor()
        if status:
            self.log = f'iter_cnt={self.iter_cnt}, final_bpr={self.final_bpr}'
            self.adjusted_pol_imgs = [np.clip(pol_img * self.e_factor, a_min=0, a_max=1) for pol_img in self.pol_imgs]
        else:
            self.log = f'iter_cnt exceeds the limit ({self.iter_max})'
        return status
