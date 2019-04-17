# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
import torch as th

from tools import stereo_tools


def test_warper_output():
    source = th.rand(2, 3, 14, 17)
    x_shift = th.rand(2, 14, 17)
    disparity_warper = stereo_tools.Warper()
    target, invalid = disparity_warper(source, x_shift)
    assert target.size() == source.size()


def test_warper_logic():
    source = th.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).view(1, 1, 2, 4)
    source.requires_grad = True
    x_shift = th.Tensor([[0, 0, 2, 1], [1, 0, float('inf'), 0]]).view(1, 2, 4)
    x_shift.requires_grad = True
    expected_target = th.Tensor([[1, 2, 1, 3], [0, 6, 0, 8]])
    disparity_warper = stereo_tools.Warper()
    target, invalid = disparity_warper(source, x_shift)
    assert target.isclose(expected_target).all()
    target.mean().backward()
    assert not th.isnan(source.grad).any()
    assert not th.isnan(x_shift.grad).any()


def test_compute_right_disparity_score():
    th.manual_seed(0)
    left_disparity_score = th.rand(1, 2, 3, 4)
    right_disparity_score = stereo_tools.compute_right_disparity_score(
        left_disparity_score, disparity_step=2)
    assert (right_disparity_score[:, 0, ...] == left_disparity_score[:, 0, ...]
            ).all()
    assert (right_disparity_score[:, 1, :, 0:-2] ==
            left_disparity_score[:, 1, :, 2:]).all()
