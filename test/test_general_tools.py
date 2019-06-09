# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
import torch as th

from tools import general_tools


def test_dilate():
    # yapf: dispable
    image = th.ByteTensor([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    expected_dilated_image = th.ByteTensor([[1, 1, 0, 0], [1, 1, 1, 1],
                                            [0, 0, 1, 1]])
    # yapf: enable
    dilated_image = general_tools.dilate(image, 3)
    assert (dilated_image == expected_dilated_image).all()
