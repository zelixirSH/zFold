"""Mathematics-related utility functions."""

import numpy as np
import torch
from torch.nn.functional import one_hot

def cdist(x1, x2=None):
    """Calculate the pairwise distance matrix.

    Args:
    * x1: input tensor of size N x D or B x N x D
    * x2: (optional) input tensor of size M x D or B x M x D

    Returns:
    * dist_mat: pairwise distance of size N x M or B x N x M

    Note:
    * If <x2> is not provided, then pairwise distance will be computed within <x1>.
    * The matrix multiplication approach will not be used to avoid the numerical stability issue.
    """

    # initialization
    compute_mode = 'donot_use_mm_for_euclid_dist'
    x2 = x1 if x2 is None else x2

    # calculate the pairwise distance matrix
    if x1.ndim == 2 and x2.ndim == 2:
        dist_mat = torch.cdist(
            x1.unsqueeze(dim=0), x2.unsqueeze(dim=0), compute_mode=compute_mode).squeeze(dim=0)
    elif x1.ndim == 3 and x2.ndim == 3:
        dist_mat = torch.cdist(x1, x2, compute_mode=compute_mode)
    else:
        raise ValueError('<x1> and <x2> must be either in the 2- or 3-dimension')

    '''
    # calculate the pairwise distance matrix
    if x1.ndim == 2 and x2.ndim == 2:
        dist_mat = torch.norm(x1.unsqueeze(dim=1) - x2.unsqueeze(dim=0), dim=2)
    elif x1.ndim == 3 and x2.ndim == 3:
        dist_mat = torch.norm(x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1), dim=3)
    else:
        raise ValueError('<x1> and <x2> must be either in the 2- or 3-dimension')
    '''

    return dist_mat

def cvt_to_one_hot(tensor, depth):
    """Convert an integer array into one-hot encodings.

    Args:
    * tensor: integer array of size D1 x D2 x ... x Dk
    * depth: one-hot encodings's depth - C

    Returns:
    * onht_tns: one-hot encodings of size D1 x D2 x ... x Dk x C
    """

    if isinstance(tensor, np.ndarray):
        assert np.min(tensor) >= 0 and np.max(tensor) < depth
        onht_tns = np.reshape(
            np.eye(depth)[tensor.ravel()], list(tensor.shape) + [depth]).astype(np.float32)
    elif isinstance(tensor, torch.Tensor):
        onht_tns = one_hot(tensor, depth)
    else:
        raise TypeError('invalid tensor type: %s' % type(tensor))

    return onht_tns


def split_by_head(tensor, n_heads):
    """Split the k-dimensional tensor by number of heads.

    Args:
    * tensor: input tensor of size D1 x D2 x ... x Dk
    * n_heads: number of heads - H

    Returns:
    * mhead_tns: multi-head tensor of size D1 x D2 x ... x H x Dk' (where Dk = H * Dk')
    """

    assert tensor.shape[-1] % n_heads == 0, \
        'the last dimension (%d) is not divisiable by # of heads (%d)' % (tensor.shape[-1], n_heads)

    mhead_tns_shape = list(tensor.shape)[:-1] + [n_heads, tensor.shape[-1] // n_heads]
    if isinstance(tensor, np.ndarray):
        mhead_tns = np.reshape(tensor, mhead_tns_shape)
    elif isinstance(tensor, torch.Tensor):
        mhead_tns = torch.reshape(tensor, mhead_tns_shape)
    else:
        raise TypeError('invalid tensor type: %s' % type(tensor))

    return mhead_tns


def check_tensor_shape(tensor, shape):
    """Check the k-dimensional tensor's shape.

    Args:
    * tensor: input tensor of size D1 x D2 x ... x Dk
    * shape: tensor shape (-1: no restraint)

    Returns: n/a
    """

    assert tensor.ndim == len(shape), \
        'mismatched number of tensor dimensions: %d vs. %d' % (tensor.ndim, len(shape))
    for idx, dim_len in enumerate(shape):
        assert dim_len in [tensor.shape[idx], -1], \
            'mismatched dimension length: %d vs. %d' % (tensor.shape[idx], dim_len)


def calc_denc_tns(dist_tns, dist_min=1.0, base=2.0, n_dims=11):
    """Calculate distance encodings with log-spaced thresholds.

    Args:
    * dist_tns: distance tensor of size D1 x D2 x ... x Dk
    * dist_min: minimal distance threshold
    * base: distance threshold's exponential base
    * n_dims: number of dimensions in distance encodings - C

    Returns:
    * denc_tns: distance encoding tensor of size D1 x D2 x ... x Dk x C

    Note:
    * The i-th distance threshold is given by: dist_min * pow(base, i - 1)
    """

    # wrapper for Numpy arrays
    if isinstance(dist_tns, np.ndarray):
        return calc_denc_tns(torch.tensor(dist_tns), dist_min, base, n_dims).detach().cpu().numpy()

    # calculate distance encodings with log-spaced thresholds
    dist_vals_np = np.reshape(
        dist_min * np.power(base, np.arange(n_dims)), [1] * dist_tns.ndim + [n_dims])
    dist_vals = torch.tensor(dist_vals_np, dtype=torch.float32, device=dist_tns.device)
    denc_tns = torch.sigmoid(torch.unsqueeze(dist_tns, dim=-1) / dist_vals - 1.0)

    return denc_tns


def calc_plane_angle(cord_1, cord_2, cord_3):
    """Calculate the plane angle defined by 3 points' 3-D coordinates.

    Args:
    * cord_1: 3-D coordinate of the 1st point
    * cord_2: 3-D coordinate of the 2nd point
    * cord_3: 3-D coordinate of the 3rd point

    Returns:
    * rad: planar angle (in radian)
    """

    eps = 1e-6
    a1 = cord_1 - cord_2
    a2 = cord_3 - cord_2
    rad = np.arccos(np.clip(
        np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + eps), -1.0, 1.0))

    return rad


def calc_dihedral_angle(cord_1, cord_2, cord_3, cord_4):
    """Calculate the dihedral angle defined by 4 points' 3-D coordinates.

    Args:
    * cord_1: 3-D coordinate of the 1st point
    * cord_2: 3-D coordinate of the 2nd point
    * cord_3: 3-D coordinate of the 3rd point
    * cord_4: 3-D coordinate of the 4th point

    Returns:
    * rad: dihedral angle (in radian)
    """

    eps = 1e-6
    a1 = cord_2 - cord_1
    a2 = cord_3 - cord_2
    a3 = cord_4 - cord_3
    v1 = np.cross(a1, a2)
    v1 = v1 / np.sqrt((v1 * v1).sum(-1) + eps)
    v2 = np.cross(a2, a3)
    v2 = v2 / np.sqrt((v2 * v2).sum(-1) + eps)
    sign = np.sign((v1 * a3).sum(-1))
    rad = np.arccos(np.clip(
        (v1 * v2).sum(-1) / np.sqrt((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1) + eps), -1.0, 1.0))
    if sign != 0:
        rad *= sign

    return rad
