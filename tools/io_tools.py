# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
import re
import PIL

import numpy as np

from torchvision.transforms import functional


def read_rgb_to_tensor(filename):
    """Returns tensor with indices [color_index, y, x]."""
    return functional.to_tensor(PIL.Image.open(filename))


def read_pfm_to_tensor(filename):
    """Returns numpy array with indices [y, x] or [color_index, y, x]."""
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().decode("utf-8").rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().decode("utf-8").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.ascontiguousarray(np.flipud(data))
    return data
