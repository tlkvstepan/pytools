# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
from collections import defaultdict

from scipy.ndimage import morphology
import numpy as np
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


def create_meshgrid(width, height, is_cuda):
    x, y = th.meshgrid([th.arange(0, width), th.arange(0, height)])
    x, y = (x.transpose(0, 1).float(), y.transpose(0, 1).float())
    if is_cuda:
        x = x.cuda()
        y = y.cuda()
    return x, y


def list_of_dictionaries_to_dictionary_of_lists(list_of_dictionaries):
    dictionary_of_lists = defaultdict(lambda: [])
    for dictionary in list_of_dictionaries:
        for item_name, item_value in dictionary.items():
            dictionary_of_lists[item_name].append(item_value)
    return dictionary_of_lists
