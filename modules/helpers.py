# File: helpers.py
# Description: Numpy helpers for image processing
# Created: 2021/10/22 20:34
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np
from scipy.signal import convolve2d


#  Guided Filter Algorithm
def guided_filter(p, I, r, eps):
    # Step 1: Compute means
    mean_I = convolve2d(I, np.ones((r, r)), mode='same') / (r * r)
    mean_p = convolve2d(p, np.ones((r, r)), mode='same') / (r * r)
    corr_I = convolve2d(I * I, np.ones((r, r)), mode='same') / (r * r)
    corr_Ip = convolve2d(I * p, np.ones((r, r)), mode='same') / (r * r)

    # Step 2: Compute variances and covariances
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    # Step 3: Compute filter parameters
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Step 4: Compute means of parameters
    mean_a = convolve2d(a, np.ones((r, r)), mode='same') / (r * r)
    mean_b = convolve2d(b, np.ones((r, r)), mode='same') / (r * r)

    # Step 5: Compute filtered output
    q = mean_a * I + mean_b

    return q



def get_bayer_indices(pattern):
    """
    Get (x_start_idx, y_start_idx) for R, Gr, Gb, and B channels
    in Bayer array, respectively
    """
    return {'gbrg': ((0, 1), (1, 1), (0, 0), (1, 0)),
            'rggb': ((0, 0), (1, 0), (0, 1), (1, 1)),
            'bggr': ((1, 1), (0, 1), (1, 0), (0, 0)),
            'grbg': ((1, 0), (0, 0), (1, 1), (0, 1)),
            'rgb-ir': ((0,0), (0, 1), (0, 2), (0, 3),
                       (1, 0), (1, 1), (1, 2), (1, 3),
                       (2, 0), (2, 1), (2, 2), (2, 3),
                       (3, 0), (3, 1), (3, 2), (3, 3))
                       }[pattern.lower()]

def split_rgbir_bayer(bayer_array, bayer_pattern = 'rgb-ir'):
    """
    Split R, G, B, and IR channels sub-arrays from a RGB-IR Bayer array
    :param bayer_array: np.ndarray(H, W)
    :param bayer_pattern: 'rggb' | 'gbrg' | 'bggr' | 'grbg'
    :return: 4-element list of R, G, B, and IR channel sub-arrays, each as an np.ndarray(H/2, W/2)
    """
    rgb_ir_indices = get_bayer_indices(bayer_pattern)

    sub_arrays = []
    for idx in rgb_ir_indices:
        x0, y0 = idx
        sub_arrays.append(
            bayer_array[y0::4, x0::4]
        )
    return sub_arrays

def get_rgbir_sub_array(bayer_array, bayer_pattern='rgb-ir'):
    sub_arrays = split_rgbir_bayer(bayer_array)

    B0, G0, R0, G1 = sub_arrays[0], sub_arrays[1], sub_arrays[2], sub_arrays[3]
    G2, IR0, G3, IR1 = sub_arrays[4], sub_arrays[5], sub_arrays[6], sub_arrays[7]
    R1, G4, B1, G5 = sub_arrays[8], sub_arrays[9], sub_arrays[10], sub_arrays[11]
    G6, IR2, G7, IR3 = sub_arrays[12], sub_arrays[13], sub_arrays[14], sub_arrays[15]

    return R0 + R1, G0 + G1 + G2 + G3 + G4 + G5 + G6 + G7, B0 + B1, IR0 + IR1 + IR3 + IR2

def get_mask_rgbir(bayer_array):
    mask_r = np.zeros(bayer_array.shape)
    mask_g = np.zeros(bayer_array.shape)
    mask_b = np.zeros(bayer_array.shape)
    mask_ir = np.zeros(bayer_array.shape)

    mask_r[0::4, 2::4] = 1
    mask_r[2::4, 0::4] = 1

    mask_b[0::4, 0::4] = 1
    mask_b[2::4, 2::4] = 1

    mask_ir[1::4, 1::4] = 1
    mask_ir[3::4, 3::4] = 1
    mask_ir[1::4, 3::4] = 1
    mask_ir[3::4, 1::4] = 1

    mask_g = np.ones(bayer_array.shape) - mask_b - mask_r - mask_ir

    return mask_r, mask_g, mask_b, mask_ir



def split_bayer(bayer_array, bayer_pattern):
    """
    Split R, Gr, Gb, and B channels sub-array from a Bayer array
    :param bayer_array: np.ndarray(H, W)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: 4-element list of R, Gr, Gb, and B channel sub-arrays, each is an np.ndarray(H/2, W/2)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    sub_arrays = []
    for idx in rggb_indices:
        x0, y0 = idx
        sub_arrays.append(
            bayer_array[y0::2, x0::2]
        )

    return sub_arrays


def reconstruct_bayer(sub_arrays, bayer_pattern = 'rgb-ir'):
    """
    Inverse implementation of split_bayer: reconstruct a Bayer array from a list of
        R, Gr, Gb, and B channel sub-arrays
    :param sub_arrays: 4-element list of R, Gr, Gb, and B channel sub-arrays, each np.ndarray(H/2, W/2)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg' - > only rgb-ir
    :return: np.ndarray(H, W)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)
    if(bayer_pattern == 'rgb-ir'):
        height, width = sub_arrays[0].shape
        bayer_array = np.empty(shape=(4 * height, 4 * width), dtype=sub_arrays[0].dtype)

        for idx, sub_array in zip(rggb_indices, sub_arrays):
            x0, y0 = idx
            bayer_array[y0::4, x0::4] = sub_array
    else:
        height, width = sub_arrays[0].shape
        bayer_array = np.empty(shape=(2 * height, 2 * width), dtype=sub_arrays[0].dtype)
        for idx, sub_array in zip(rggb_indices, sub_arrays):
            x0, y0 = idx
            bayer_array[y0::2, x0::2] = sub_array

    return bayer_array


def pad(array, pads, mode='reflect'):
    """
    Pad an array with given margins
    :param array: np.ndarray(H, W, ...)
    :param pads: {int, sequence}
        if int, pad top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction pad, x-direction pad)
        if 4-element sequence: (top pad, bottom pad, left pad, right pad)
    :param mode: padding mode, see np.pad
    :return: padded array: np.ndarray(H', W', ...)
    """
    if isinstance(pads, (list, tuple, np.ndarray)):
        if len(pads) == 2:
            pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (array.ndim - 2)
        elif len(pads) == 4:
            pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (array.ndim - 2)
        else:
            raise NotImplementedError

    return np.pad(array, pads, mode=mode)


def crop(array, crops):
    """
    Crop an array by given margins
    :param array: np.ndarray(H, W, ...)
    :param crops: {int, sequence}
        if int, crops top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction crop, x-direction crop)
        if 4-element sequence: (top crop, bottom crop, left crop, right crop)
    :return: cropped array: np.ndarray(H', W', ...)
    """
    if isinstance(crops, (list, tuple, np.ndarray)):
        if len(crops) == 2:
            top_crop = bottom_crop = crops[0]
            left_crop = right_crop = crops[1]
        elif len(crops) == 4:
            top_crop, bottom_crop, left_crop, right_crop = crops
        else:
            raise NotImplementedError
    else:
        top_crop = bottom_crop = left_crop = right_crop = crops

    height, width = array.shape[:2]
    return array[top_crop: height - bottom_crop, left_crop: width - right_crop, ...]


def shift_array(padded_array, window_size):
    """
    Shift an array within a window and generate window_size**2 shifted arrays
    :param padded_array: np.ndarray(H+2r, W+2r)
    :param window_size: 2r+1
    :return: a generator of length (2r+1)*(2r+1), each is an np.ndarray(H, W), and the original
        array before padding locates in the middle of the generator
    """
    wy, wx = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
    assert wy % 2 == 1 and wx % 2 == 1, 'only odd window size is valid'

    height = padded_array.shape[0] - wy + 1
    width = padded_array.shape[1] - wx + 1

    for y0 in range(wy):
        for x0 in range(wx):
            yield padded_array[y0:y0 + height, x0:x0 + width, ...]


def gen_gaussian_kernel(kernel_size, sigma):
    if isinstance(kernel_size, (list, tuple)):
        assert len(kernel_size) == 2
        wy, wx = kernel_size
    else:
        wy = wx = kernel_size

    x = np.arange(wx) - wx // 2
    if wx % 2 == 0:
        x += 0.5

    y = np.arange(wy) - wy // 2
    if wy % 2 == 0:
        y += 0.5

    y, x = np.meshgrid(y, x)

    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def generic_filter(array, kernel):
    """
    Filter input image array with given kernel
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param kernel: np.ndarray(h, w)
    :return: filtered array: np.ndarray(H, W, ...)
    """
    kh, kw = kernel.shape[:2]
    kernel = kernel.flatten()

    padded_array = pad(array, pads=(kh // 2, kw // 2))
    shifted_arrays = shift_array(padded_array, window_size=(kh, kw))

    filtered_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        filtered_array += kernel[i] * shifted_array
        weights += kernel[i]

    filtered_array = (filtered_array / weights).astype(array.dtype)
    return filtered_array


def mean_filter(array, filter_size=3):
    """
    A faster reimplementation of the mean filter
    :param array: array to be filter: np.ndarray(H, W, ...)
    :param filter_size: int, diameter of the mean-filter
    :return: filtered array: np.ndarray(H, W, ...)
    """

    assert filter_size % 2 == 1, 'only odd filter size is valid'

    padded_array = pad(array, pads=filter_size // 2)
    shifted_arrays = shift_array(padded_array, window_size=filter_size)
    return (sum(shifted_arrays) / filter_size ** 2).astype(array.dtype)


def bilateral_filter(array, spatial_weights, intensity_weights_lut, right_shift=0):
    """
    A faster reimplementation of the bilateral filter
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param spatial_weights: np.ndarray(h, w): predefined spatial gaussian kernel, where h and w are
        kernel height and width respectively
    :param intensity_weights_lut: a predefined exponential LUT that maps intensity distance to the weight
    :param right_shift: shift the multiplication result of the spatial- and intensity weights to the
        right to avoid integer overflow when multiply this result to the input array
    :return: filtered array: np.ndarray(H, W, ...)
    """
    filter_height, filter_width = spatial_weights.shape[:2]
    spatial_weights = spatial_weights.flatten()

    padded_array = pad(array, pads=(filter_height // 2, filter_width // 2))
    shifted_arrays = shift_array(padded_array, window_size=(filter_height, filter_width))

    bf_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        intensity_diff = (shifted_array - array) ** 2
        weight = intensity_weights_lut[intensity_diff] * spatial_weights[i]
        weight = np.right_shift(weight, right_shift)  # to avoid overflow

        bf_array += weight * shifted_array
        weights += weight

    bf_array = (bf_array / weights).astype(array.dtype)

    return bf_array
