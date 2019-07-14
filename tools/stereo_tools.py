# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.
from torch import nn
import torch as th

from tools import general_tools


def another_camera_name(camera_name):
    if camera_name == 'right':
        return 'left'
    elif camera_name == 'left':
        return 'right'
    else:
        ValueError('"camera_name" should be left or right.')


def warp_2d(source, y_displacement, x_displacement):
    """Returns warped source image and occlusion_mask.

    Value in location (x, y) in output image in taken from
    (x + x_displacement, y + y_displacement) location of the
    source image.

    If the location in the source image is outside of its borders,
    the location in the target image is filled with zeros and the
    location is added to the "occlusion_mask".

    Args:
        source: is a tensor with indices
                [example_index, channel_index, y, x].
        x_displacement,
        y_displacement: are tensors with indices [example_index,
                        1, y, x].

    Returns:
        target: is a tensor with indices
                [example_index, channel_index, y, x].
        occlusion_mask: is a tensor with indices [example_index, y, x].
    """
    width, height = source.size(-1), source.size(-2)
    x_target, y_target = general_tools.create_meshgrid(width, height,
                                                       source.is_cuda)
    x_source = x_target + x_displacement.squeeze(1)
    y_source = y_target + y_displacement.squeeze(1)
    # Normalize coordinates to [-1,1]
    occlusion_mask = ((x_source.detach() < 0) | (x_source.detach() >= width) |
                      (y_source.detach() < 0) | (y_source.detach() >= height))
    x_source = (2.0 / float(width - 1)) * x_source - 1
    y_source = (2.0 / float(height - 1)) * y_source - 1
    x_source = x_source.masked_fill(occlusion_mask, 0)
    y_source = y_source.masked_fill(occlusion_mask, 0)
    grid_source = th.stack([x_source, y_source], -1)
    target = nn.functional.grid_sample(source, grid_source)
    target.masked_fill_(occlusion_mask.unsqueeze(1).expand_as(target), 0)
    return target, occlusion_mask


def warp_1d(source, disparity):
    """Returns warped source image and occlusion_mask.

    Value in (x, y) location in the output image is taken from
    (x - disparity, y) location of the source image.

    If the location in the source image is outside of its borders,
    the location in the output image is filled with zeros and the
    location is added to the "occlusion_mask".

    Args:
        source: is a tensor with indices
                [example_index, channel_index, y, x].
        disparity: is tensor with indices [example_index, 1, y, x].

    Returns:
        target: is a tensor with indices
                [example_index, channel_index, y, x].
        occlusion_mask: is a tensor with indices [example_index, y, x].
    """
    width, height = source.size(-1), source.size(-2)
    x_target, y_target = general_tools.create_meshgrid(width, height,
                                                       source.is_cuda)
    x_source = x_target - disparity.squeeze(1)
    y_source = y_target.unsqueeze(0).expand_as(x_source)
    # Normalize coordinates to [-1,1]
    occlusion_mask = (x_source.detach() < 0) | (x_source.detach() >= width)
    x_source = (2.0 / float(width - 1)) * x_source - 1
    y_source = (2.0 / float(height - 1)) * y_source - 1
    x_source = x_source.masked_fill(occlusion_mask, 0)
    y_source = y_source.masked_fill(occlusion_mask, 0)
    grid_source = th.stack([x_source, y_source], -1)
    target = nn.functional.grid_sample(source, grid_source)
    occlusion_mask = occlusion_mask.unsqueeze(1)
    target.masked_fill_(occlusion_mask.expand_as(target), 0)
    return target, occlusion_mask


def compute_occlusion_mask(opposite_view_disparity):
    """Returns occlusion mask.

    Args:
        opposite_view_disparity: a tensor with indices [example_index, 1,
                                 y, x] with disparities. Positive disparity
                                 correspond to shift to the left. To detect
                                 occlusions in the left view, one
                                 should provide right view disparity.

    Returns:
        occusion mask: is a tensor with indices [example_index, 1, y, x].
    """
    with th.no_grad():
        (number_of_examples, _, height, width) = opposite_view_disparity.size()
        occlusion_masks = []
        for example_index in range(number_of_examples):
            occlusion_mask = th.ones(height, width).byte()
            if opposite_view_disparity.is_cuda:
                occlusion_mask = occlusion_mask.cuda()
            (x, y) = general_tools.create_meshgrid(
                width, height, opposite_view_disparity.is_cuda)
            x = (x - opposite_view_disparity[example_index].squeeze(0)
                 ).round().long()
            y = y.long()
            valid_locations = (x >= 0) & (x < width)
            x = x[valid_locations]
            y = y[valid_locations]
            occlusion_mask[y, x] = 0
            occlusion_masks.append(occlusion_mask)
        occlusion_masks = th.stack(occlusion_masks, dim=0)
    return occlusion_masks.unsqueeze(1)


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
        disparity = disparity_index * disparity_step
        right_disparity_score[:, disparity_index, :, 0:-disparity] = \
            left_disparity_score[:, disparity_index, :, disparity:]
    return right_disparity_score


def find_locations_with_consistent_disparities(
        left_disparity, right_disparity,
        maximum_allowed_disparity_difference=0.5):
    """Returns mask of consistent disparities.

    To check consistency the right view disparity is warped to the
    left view using the left view disparity and then compared to
    the left view disparities. Location where the difference between
    the disparities is < maximumu_allowed_disparity_difference.
    """
    warped_right_disparity = warp_1d(right_disparity, left_disparity)[0]
    difference = (warped_right_disparity - left_disparity).abs()
    return difference <= maximum_allowed_disparity_difference


def find_cycle_consistent_locations(left_disparity,
                                    right_disparity,
                                    left_image=None,
                                    maximum_allowed_intensity_difference=50):
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
    warped_texture = warp_1d(texture, -right_disparity)[0]
    warped2x_texture = warp_1d(warped_texture, left_disparity)[0]
    difference = (warped2x_texture - texture).abs()
    return difference <= maximum_allowed_intensity_difference
