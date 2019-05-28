# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
import os
import pkg_resources

from tools import io_tools

FOLDER_WITH_TEST_DATA = \
    pkg_resources.resource_filename(__name__, "data")


def test_read_rgb_to_tensor():
    tensor = io_tools.read_rgb_to_tensor(
        os.path.join(FOLDER_WITH_TEST_DATA, '0006.png'))
    assert tensor.dim() == 3
    assert tensor.size(0) == 3


def test_read_pfm_to_tensor():
    tensor = io_tools.read_pfm_to_tensor(
        os.path.join(FOLDER_WITH_TEST_DATA, '0006.pfm'))
    assert tensor.ndim == 2
