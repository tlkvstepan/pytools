# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
import torch as th
from torch import nn


class Warper(nn.Module):
    def __init__(self):
        super(Warper, self).__init__()

    def _create_meshgrid(self, width, height, is_cuda):
        x_target, y_target = th.meshgrid(
            [th.arange(0, width), th.arange(0, height)])
        x_target, y_target = (x_target.transpose(0, 1).float(),
                              y_target.transpose(0, 1).float())
        if is_cuda:
            x_target = x_target.cuda()
            y_target = y_target.cuda()
        return x_target, y_target

    def forward(self, source, x_shift):
        """Returns warped source image.

        In value in (x, y) location in the target image in take from
        (x - x_shift, y) location in the source image.

        If the location in the source image is outside of borders,
        the location in the target image is filled with zeros.

        If the shift is 'inf', the location in the target image is
        filled with zeros.

        Args:
            source: is tensor with indices [example_index, channel_index,
                    y, x].
            x_shift: is tensor with indices [example_index, y, x].
        """
        width, height = source.size(-1), source.size(-2)
        x_target, y_target = self._create_meshgrid(width, height,
                                                   source.is_cuda)
        x_source = x_target - x_shift
        y_source = y_target.unsqueeze(0)
        # Normalize coordinates to [-1,1]
        invalid = (x_source.detach() < 0) | (x_source.detach() >= width)
        x_source = (2.0 / float(width - 1)) * x_source - 1
        y_source = (2.0 / float(height - 1)) * y_source - 1
        grid_source = th.stack([x_source, y_source.expand_as(x_source)], -1)
        target = nn.functional.grid_sample(source, grid_source)
        target.masked_fill_(invalid.unsqueeze(1).expand_as(target), 0)
        return target, invalid.float()


def compute_right_disparity_score(left_disparity_score, disparity_step=2):
    """Returns right disparity score given left disparity score.

    Note that the left disparities correspond to shift to the left in a
    right image, while the right disparities correspond to shift to the
    right in a left image.

    Args:
        left_disparity_score: is a tensor with indices [example_index,
                              disparity_index, y, x].
        disparity_step: difference between nearby disparities in
                        disparity_score tensor.

    Returns:
        tensor with indices [example_index, disparity_index, y, x].
    """
    right_disparity_score = th.zeros(
        left_disparity_score.size()).type_as(left_disparity_score)
    maximum_disparity_index = left_disparity_score.size(1)
    right_disparity_score[:, 0, ...] = left_disparity_score[:, 0, ...]
    for disparity_index in range(1, maximum_disparity_index):
        disparity = disparity_index * 2
        right_disparity_score[:, disparity_index, :, 0:-disparity] = \
            left_disparity_score[:, disparity_index, :, disparity:]
    return right_disparity_score


def find_locations_with_consistent_disparities(
        left_disparity,
        right_disparity,
        maximum_allowed_disparity_difference=0.5,
        averaging_window_size=5):
    """Returns mask of consistent disparities.

    To check consistency the right view disparity is warped to the
    left view using the left view disparity and then compared to
    the left view disparities. Location where the difference between
    the disparities is < maximumu_allowed_disparity_difference.
    """
    disparity_warper = Warper()
    warped_right_disparity = disparity_warper(
        right_disparity.unsqueeze(1), left_disparity)[0].squeeze(1)
    difference = (warped_right_disparity - left_disparity
                  ).abs()
    window_difference = nn.functional.avg_pool2d(
        difference, kernel_size=averaging_window_size, stride=1,
        padding=averaging_window_size // 2)
    return window_difference <= maximum_allowed_disparity_difference


def find_cycle_consistent_locations(left_disparity,
                                    right_disparity,
                                    left_image=None,
                                    maximum_allowed_intensity_difference=50,
                                    averaging_window_size=5):
    """Returns mask of cycle consistent locations.

    To check cycle consistency random noise image or actual left image is
    warped to the right viewpoint using right disparity and to then back
    to the left viewpoint using the left disparity. The resulting image
    is compared to the original random noise image or the actual left
    image, and locations with intensity error <
    maximum_allowed_intensity_difference are reported as consistent.

    Note, that if we use actual left image this consistency check is equal
    to the original cyclic loss.
    """
    if left_image is None:
        height, width = left_disparity.size()[-2:]
        texture = th.rand(1, 1, height, width).type_as(left_disparity) * 255.0
    else:
        texture = left_image.mean(dim=1, keepdim=True)
    stereo_warper = Warper()
    warped_texture = stereo_warper(texture, -right_disparity)[0]
    warped2x_texture = stereo_warper(warped_texture, left_disparity)[0]
    difference = (warped2x_texture - texture).abs()
    window_difference = nn.functional.avg_pool2d(
        difference, kernel_size=averaging_window_size, stride=1,
        padding=averaging_window_size // 2)
    return window_difference <= maximum_allowed_intensity_difference
