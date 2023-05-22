# File: awb.py
# Description: Auto White Balance (actually not Auto)
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule
from .helpers import split_bayer, reconstruct_bayer, split_rgbir_bayer, get_rgbir_sub_array, get_mask_rgbir


class AWB(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.r_gain = np.array(self.params.r_gain, dtype=np.uint32)  # x1024
        self.gr_gain = np.array(self.params.gr_gain, dtype=np.uint32)  # x1024
        self.gb_gain = np.array(self.params.gb_gain, dtype=np.uint32)  # x1024
        self.b_gain = np.array(self.params.b_gain, dtype=np.uint32)  # x1024

    def getUV(self, max, max_k, mean, mean_k):
        u = (max_k/max - mean_k / mean) / (max - mean)
        v = (max_k / max - max*u)
        return u, v


    def execute(self, data):
        bayer = data['bayer'].astype(np.uint32)

        #  Using QCGP to do the AWB
        bayer = np.clip(bayer, 0, self.cfg.saturation_values.hdr)
        raw_r, raw_g, raw_b, raw_ir = get_rgbir_sub_array(bayer)
        #  Get mean value for each channel
        #  Noted that raw_r is the sub-array of the original picture
        mean_r = np.mean(raw_r) / 2
        mean_g = np.mean(raw_g) / 8
        mean_b = np.mean(raw_b) / 2
        mean_ir = np.mean(raw_ir) / 4
        mean_k = (mean_r + mean_g + mean_b + mean_ir) / 4

        #  Get the max value for each channel
        mask_r, mask_g, mask_b, mask_ir = get_mask_rgbir(bayer)
        RED = bayer * mask_r
        GREEN = bayer * mask_g
        BLUE = bayer * mask_b
        IR = bayer * mask_ir

        max_r = np.max(RED)
        max_g = np.max(GREEN)
        max_b = np.max(BLUE)
        max_ir = np.max(IR)
        max_k = (max_r + max_g + max_b + max_ir) / 4

        #  Get the uv for each channel
        u_red, v_red = self.getUV(max_r, max_k, mean_r, mean_k)
        u_green, v_green = self.getUV(max_g, max_k, mean_g, mean_k)
        u_blue, v_blue = self.getUV(max_b, max_k, mean_b, mean_k)
        u_ir, v_ir = self.getUV(max_ir, max_k, mean_ir, mean_k)



        m_RED = (u_red * RED * RED) + v_red * RED
        m_GREEN = (u_green * GREEN * GREEN) + v_green * GREEN
        m_BLUE = (u_blue * BLUE * BLUE) + v_blue * BLUE
        m_IR = (u_ir * IR * IR) + v_ir * IR

        m_RGBIR = m_RED * mask_r + m_GREEN * mask_g + m_BLUE * mask_b + m_IR * mask_ir
        data['bayer'] = m_RGBIR.astype(np.uint16)

        # sub_arrays = split_bayer(bayer, self.cfg.hardware.bayer_pattern)
        # gains = (self.r_gain, self.gr_gain, self.gb_gain, self.b_gain)

        # wb_sub_arrays = []
        # for sub_array, gain in zip(sub_arrays, gains):
        #     wb_sub_arrays.append(
        #         np.right_shift(gain * sub_array, 10)
        #     )
        # wb_bayer = reconstruct_bayer(wb_sub_arrays, self.cfg.hardware.bayer_pattern)
        # wb_bayer = np.clip(wb_bayer, 0, self.cfg.saturation_values.hdr)

        # data['bayer'] = wb_bayer.astype(np.uint16)
