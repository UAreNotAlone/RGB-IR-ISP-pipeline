# File: blc.py
# Description: Black Level Compensation
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule
from .helpers import split_bayer, reconstruct_bayer, get_rgbir_sub_array, get_mask_rgbir


class BLC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.alpha = np.array(self.params.alpha, dtype=np.int32)  # x1024
        self.beta = np.array(self.params.beta, dtype=np.int32)  # x1024

    def execute(self, data):
        bayer = data['bayer'].astype(np.int32)
        raw_r, raw_g, raw_b, raw_ir = get_rgbir_sub_array(bayer)
        #  Get mean value for each channel
        #  Noted that raw_r is the sub-array of the original picture
        mean_r = np.mean(raw_r) / 2
        mean_g = np.mean(raw_g) / 8
        mean_b = np.mean(raw_b) / 2
        mean_ir = np.mean(raw_ir) / 4

        mask_r, mask_g, mask_b, mask_ir = get_mask_rgbir(bayer)
        RED = bayer * mask_r
        GREEN = bayer * mask_g
        BLUE = bayer * mask_b
        IR = bayer * mask_ir

        m_RED = RED
        m_GREEN = GREEN
        m_BLUE = BLUE
        m_IR = IR

        m_RED = np.clip(m_RED, m_RED - 0, None)
        m_BLUE = np.clip(m_BLUE, m_BLUE - 0, None)
        m_IR = np.clip(m_IR, m_IR - 0, None)
        m_GREEN -= (0 - np.right_shift(0, 10))
        #  m_RED = np.clip(m_RED, m_RED - 0, None)

        #  Restore
        m_RGBIR = m_RED * mask_r + m_GREEN * mask_g + m_BLUE * mask_b + m_IR * mask_ir
        #  data['bayer'] = m_RGBIR.astype(np.uint16)
        

        # r, gr, gb, b = split_bayer(bayer, self.cfg.hardware.bayer_pattern)
        # r = np.clip(r - self.params.bl_r, 0, None)
        # b = np.clip(b - self.params.bl_b, 0, None)
        # gr -= (self.params.bl_gr - np.right_shift(r * self.alpha, 10))
        # gb -= (self.params.bl_gb - np.right_shift(b * self.beta, 10))
        # blc_bayer = reconstruct_bayer(
        #     (r, gr, gb, b), self.cfg.hardware.bayer_pattern
        # )

        data['bayer'] = np.clip(m_RGBIR, 0, None).astype(np.uint16)
