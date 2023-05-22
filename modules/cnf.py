# File: cnf.py
# Description: Chroma Noise Filtering
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule
from .helpers import split_bayer, reconstruct_bayer, mean_filter, split_rgbir_bayer

#  Modify the pipeline to adapt to the RGB-IR
class CNF(BasicModule):
    def cnd(self, bayer):
        sub_arrays = split_rgbir_bayer(bayer)

        B0, G0, R0, G1 = sub_arrays[0], sub_arrays[1], sub_arrays[2], sub_arrays[3]
        G2, IR0, G3, IR1 = sub_arrays[4], sub_arrays[5], sub_arrays[6], sub_arrays[7]
        R1, G4, B1, G5 = sub_arrays[8], sub_arrays[9], sub_arrays[10], sub_arrays[11]
        r = np.right_shift(R1 + R0, 1)
        b = np.right_shift(B1 + B0, 1)
        avg_r = np.right_shift(mean_filter(R0, filter_size=5) + mean_filter(R1, filter_size=5), 1)
        avg_g = np.right_shift(mean_filter(G0, filter_size=5) + mean_filter(G1, filter_size=5) + mean_filter(G2, filter_size=5) + 
                               mean_filter(G3, filter_size=5) + mean_filter(G4, filter_size=5) + mean_filter(G5, filter_size=5), 3)
        avg_b = np.right_shift(mean_filter(B0, filter_size=5)+mean_filter(B1, filter_size=5), 1)

        is_r_noise = (r - avg_g > self.params.diff_threshold) * \
                     (r - avg_b > self.params.diff_threshold) * \
                     (avg_r - avg_g > self.params.diff_threshold) * \
                     (avg_r - avg_b < self.params.diff_threshold)
        is_b_noise = (b - avg_g > self.params.diff_threshold) * \
                     (b - avg_r > self.params.diff_threshold) * \
                     (avg_b - avg_g > self.params.diff_threshold) * \
                     (avg_b - avg_r < self.params.diff_threshold)

        return avg_r, avg_g, avg_b, is_r_noise, is_b_noise

    @staticmethod
    def cnc(array, avg_g, avg_c1, avg_c2, y, gain):
        assert array.dtype == np.int32

        if gain <= 1024:  # x1024
            damp_factor = 256  # x256
        elif 1024 < gain <= 1229:
            damp_factor = 128
        else:
            damp_factor = 77

        max_avg = np.maximum(avg_g, avg_c2)
        signal_gap = array - max_avg
        chroma_corrected = max_avg + np.right_shift(damp_factor * signal_gap, 8)

        fade1 = (y <= 30) * 1.0 + \
                (y > 30) * (y <= 50) * 0.9 + \
                (y > 50) * (y <= 70) * 0.8 + \
                (y > 70) * (y <= 100) * 0.7 + \
                (y > 100) * (y <= 150) * 0.6 + \
                (y > 150) * (y <= 200) * 0.3 + \
                (y > 200) * (y <= 250) * 0.1
        fade2 = (avg_c1 <= 30) * 1.0 + \
                (avg_c1 > 30) * (avg_c1 <= 50) * 0.9 + \
                (avg_c1 > 50) * (avg_c1 <= 70) * 0.8 + \
                (avg_c1 > 70) * (avg_c1 <= 100) * 0.6 + \
                (avg_c1 > 100) * (avg_c1 <= 150) * 0.5 + \
                (avg_c1 > 150) * (avg_c1 <= 200) * 0.3
        fade = fade1 * fade2

        cnc = fade * chroma_corrected + (1 - fade) * array
        return cnc.astype(np.int32)

    def execute(self, data):
        bayer = data['bayer'].astype(np.int32)

        sub_arrays = split_rgbir_bayer(bayer)

        B0, G0, R0, G1 = sub_arrays[0], sub_arrays[1], sub_arrays[2], sub_arrays[3]
        G2, IR0, G3, IR1 = sub_arrays[4], sub_arrays[5], sub_arrays[6], sub_arrays[7]
        R1, G4, B1, G5 = sub_arrays[8], sub_arrays[9], sub_arrays[10], sub_arrays[11]
        avg_r, avg_g, avg_b, is_r_noise, is_b_noise = self.cnd(bayer)

        # y = 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b
        y = np.right_shift(306 * avg_r + 601 * avg_g + 117 * avg_b, 10)
        r0_cnc = self.cnc(R0, avg_g, avg_r, avg_b, y, self.params.r_gain)
        r1_cnc = self.cnc(R1, avg_g, avg_r, avg_b, y, self.params.r_gain)
        b0_cnc = self.cnc(B0, avg_g, avg_r, avg_b, y, self.params.b_gain)
        b1_cnc = self.cnc(B1, avg_g, avg_r, avg_b, y, self.params.b_gain)

        r0_cnc = is_r_noise * r0_cnc + ~is_r_noise * R0
        r1_cnc = is_r_noise * r1_cnc + ~is_r_noise * R1
        b0_cnc = is_b_noise * b0_cnc + ~is_b_noise * B0
        b1_cnc = is_b_noise * b1_cnc + ~is_b_noise * B1
        # r_cnc = self.cnc(r, avg_g, avg_r, avg_b, y, self.params.r_gain)
        # b_cnc = self.cnc(b, avg_g, avg_b, avg_r, y, self.params.b_gain)
        # r_cnc = is_r_noise * r_cnc + ~is_r_noise * r
        # b_cnc = is_b_noise * b_cnc + ~is_b_noise * b

        cnf_bayer = reconstruct_bayer((b0_cnc, G0, r0_cnc, G1, G2, IR0, G3, IR1, r1_cnc, G4, b1_cnc, G5), self.cfg.hardware.bayer_pattern)
        cnf_bayer = np.clip(cnf_bayer, 0, self.cfg.saturation_values.hdr)

        data['bayer'] = cnf_bayer.astype(np.uint16)
