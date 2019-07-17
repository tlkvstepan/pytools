# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
from scipy.ndimage import morphology
import numpy as np
import cv2
import torch as th


def dilate(image, window_size):
    """Returns dilated image.

    Args:
        image: is a boolen tensor of size (height x width) image
               that will be dilated.
        window_size: size of the window that will be used for the
                     dilation.
    """
    dilation_window = np.ones((window_size, window_size), dtype=bool)
    dilated_image = th.from_numpy(
        morphology.binary_dilation(image.cpu().numpy(),
                                   dilation_window).astype(np.uint8))
    if image.is_cuda:
        return dilated_image.cuda()
    return dilated_image


def median_filter(image, window_size):
    """Returns median filted image.

    Args:
        image: is a tensor of (height x width).
        window_size: size of median filter.
    """
    return th.from_numpy(cv2.medianBlur(image.cpu().numpy(),
                                        window_size)).type_as(image)
