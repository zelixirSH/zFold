"""Utility function."""

import torch

def calc_rot_n_tsl(x1, x2, x3):
    """Calculate the rotation matrix & translation vector.

    Args:
    * x1: 1st atom's 3D coordinate of size 3
    * x2: 2nd atom's 3D coordinate of size 3
    * x3: 3rd atom's 3D coordinate of size 3

    Returns:
    * rot_mat: rotation matrix of size 3 x 3
    * tsl_vec: translation vector of size 3

    Note:
    * <x2> is the origin point
    * <x3> - <x2> defines the direction of X-axis
    * <x1> lies in the X-Y plane

    Reference:
    * Jumper et al., Highly accurate protein structure prediction with AlphaFold. Nature, 2021.
      - Supplementary information, Section 1.8.1, Algorithm 21.
    """

    eps = 1e-6
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / (torch.norm(v1) + eps)
    u2 = v2 - torch.inner(e1, v2) * e1
    e2 = u2 / (torch.norm(u2) + eps)
    e3 = torch.cross(e1, e2)
    rot_mat = torch.stack([e1, e2, e3], dim=0).permute(1, 0)
    tsl_vec = x2

    return rot_mat, tsl_vec


def calc_rot_n_tsl_batch(cord_tns):
    """Calculate rotation matrices & translation vectors in the batch mode.

    Args:
    * cord_tns: 3D coordinates of size N x 3 x 3

    Returns:
    * rot_tns: rotation matrices of size N x 3 x 3
    * tsl_mat: translation vectors of size N x 3
    """

    eps = 1e-6
    x1, x2, x3 = [torch.squeeze(x, dim=1) for x in torch.split(cord_tns, 1, dim=1)]
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)
    u2 = v2 - torch.sum(e1 * v2, dim=1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=1)
    rot_tns = torch.stack([e1, e2, e3], dim=1).permute(0, 2, 1)
    tsl_mat = x2

    return rot_tns, tsl_mat


def calc_dihd_angl(x1, x2, x3, x4):
    """Calculate the dihedral angle.

    Args:
    * x1: 1st atom's 3D coordinate of size 3
    * x2: 2nd atom's 3D coordinate of size 3
    * x3: 3rd atom's 3D coordinate of size 3
    * x4: 4th atom's 3D coordinate of size 3

    Returns:
    * rad: dihedral angle (in radian, ranging from -pi to pi)
    """

    eps = 1e-6
    a1 = x2 - x1
    a2 = x3 - x2
    a3 = x4 - x3
    v1 = torch.cross(a1, a2)
    v1 = v1 / (torch.norm(v1) + eps)
    v2 = torch.cross(a2, a3)
    v2 = v2 / (torch.norm(v2) + eps)
    sign = torch.sign(torch.inner(v1, a3))
    rad = torch.arccos(torch.clip(
        torch.inner(v1, v2) / (torch.norm(v1) * torch.norm(v2) + eps), -1.0, 1.0))
    if sign != 0:
        rad *= sign

    return rad


def calc_dihd_angl_batch(cord_tns):
    """Calculate dihedral angles in the batch mode.

    Args:
    * cord_tns: 3D coordinates of size N x 4 x 3

    Returns:
    * rad_vec: dihedral angles (in radian, ranging from -pi to pi) of size N
    """

    eps = 1e-6
    x1, x2, x3, x4 = [torch.squeeze(x, dim=1) for x in torch.split(cord_tns, 1, dim=1)]
    a1 = x2 - x1
    a2 = x3 - x2
    a3 = x4 - x3
    v1 = torch.cross(a1, a2, dim=1)
    v1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)  # is this necessary?
    v2 = torch.cross(a2, a3, dim=1)
    v2 = v2 / (torch.norm(v2, dim=1, keepdim=True) + eps)  # is this necessary?
    sign = torch.sign(torch.sum(v1 * a3, dim=1))
    sign[sign == 0.0] = 1.0  # to avoid multiplication with zero
    rad_vec = sign * torch.arccos(torch.clip(
        torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1) + eps), -1.0, 1.0))

    return rad_vec


def quat2rot(quat_vecs):
    """Convert partial quaternion vectors into rotation matrices.

    Args:
    * quat_vecs: partial quaternion vectors of size L x 3

    Returns:
    * rot_mats: rotation matrices of size L x 3 x 3

    Note:
    * The first component in the quaternion vector is fixed to 1 and therefore omitted.

    Reference:
    * J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
      Technical report, 2020. - Section 2.4.1
    """

    tns_type = quat_vecs.dtype
    # convert float32
    quat_vecs = quat_vecs.float()

    # obtain normalized quaternion vectors
    norm_vec = torch.sqrt(1.0 + torch.sum(torch.square(quat_vecs), dim=1))
    qr = 1.0 / norm_vec
    qx = quat_vecs[:, 0] / norm_vec
    qy = quat_vecs[:, 1] / norm_vec
    qz = quat_vecs[:, 2] / norm_vec

    '''
    # calculate each entry in the rotation matrix
    r00 = qr ** 2 + qx ** 2 - qy ** 2 - qz ** 2
    r01 = 2 * (qx * qy - qr * qz)
    r02 = 2 * (qz * qx + qr * qy)
    r10 = 2 * (qx * qy + qr * qz)
    r11 = qr ** 2 - qx ** 2 + qy ** 2 - qz ** 2
    r12 = 2 * (qy * qz - qr * qx)
    r20 = 2 * (qz * qx - qr * qy)
    r21 = 2 * (qy * qz + qr * qx)
    r22 = qr ** 2 - qx ** 2 - qy ** 2 + qz ** 2

    # stack all the entries into rotation matrices
    rot_mats = torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1),
    ], dim=1)
    '''

    # calculate intermediate results
    qrr = torch.square(qr)
    qxx = torch.square(qx)
    qyy = torch.square(qy)
    qzz = torch.square(qz)
    qrx = 2 * qr * qx
    qry = 2 * qr * qy
    qrz = 2 * qr * qz
    qxy = 2 * qx * qy
    qxz = 2 * qx * qz
    qyz = 2 * qy * qz

    # calculate each entry in the rotation matrix
    r11 = qrr + qxx - qyy - qzz
    r12 = qxy - qrz
    r13 = qxz + qry
    r21 = qxy + qrz
    r22 = qrr - qxx + qyy - qzz
    r23 = qyz - qrx
    r31 = qxz - qry
    r32 = qyz + qrx
    r33 = qrr - qxx - qyy + qzz

    # stack all the entries into rotation matrices
    rot_mats = torch.stack([
        torch.stack([r11, r12, r13], dim=1),
        torch.stack([r21, r22, r23], dim=1),
        torch.stack([r31, r32, r33], dim=1),
    ], dim=1)

    return rot_mats.to(tns_type)


def rot2quat(rot_mats):
    """Convert rotation matrices into partial quaternion vectors.

    Args:
    * rot_mats: rotation matrices of size L x 3 x 3

    Returns:
    * quat_vecs: partial quaternion vectors of size L x 3

    Note:
    * The first component in the quaternion vector is fixed to 1 and therefore omitted.

    Reference:
    * J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
      Technical report, 2020. - Section 2.1.1 & Section 2.5.1.
    """

    tns_type = rot_mats.dtype
    rot_mats = rot_mats.float()

    # configurations
    eps = 1e-6

    # extract entries from rotation matrices
    r11, r12, r13 = [torch.squeeze(x, dim=1) for x in torch.split(rot_mats[:, 0], 1, dim=1)]
    r21, r22, r23 = [torch.squeeze(x, dim=1) for x in torch.split(rot_mats[:, 1], 1, dim=1)]
    r31, r32, r33 = [torch.squeeze(x, dim=1) for x in torch.split(rot_mats[:, 2], 1, dim=1)]

    # recover pitch, yaw, and roll angles (naive implementation)
    ptc = torch.atan2(-r31, torch.sqrt(torch.square(r11) + torch.square(r21)))
    yaw = torch.atan2(r21, r11)
    rll = torch.atan2(r32, r33)

    '''
    # recover pitch, yaw, and roll angles (numerically stable implementation)
    ptc = torch.atan2(-r31, torch.sqrt(torch.square(r11) + torch.square(r21)))
    yaw = torch.where(
        torch.abs(torch.abs(ptc) - np.pi / 2.0) >= eps,
        torch.atan2(r21, r11),
        torch.where(ptc < 0.0, torch.atan2(-r23, -r13), torch.atan2(r23, r13)),
    )
    rll = torch.where(
        torch.abs(torch.abs(ptc) - np.pi / 2.0) >= eps,
        torch.atan2(r32, r33),
        torch.zeros_like(ptc),
    )
    '''

    # calculate normalized quaternion vectors
    ptc_s, ptc_c = torch.sin(ptc / 2.0), torch.cos(ptc / 2.0)
    yaw_s, yaw_c = torch.sin(yaw / 2.0), torch.cos(yaw / 2.0)
    rll_s, rll_c = torch.sin(rll / 2.0), torch.cos(rll / 2.0)
    qr = rll_c * ptc_c * yaw_c + rll_s * ptc_s * yaw_s
    qx = rll_s * ptc_c * yaw_c - rll_c * ptc_s * yaw_s
    qy = rll_c * ptc_s * yaw_c + rll_s * ptc_c * yaw_s
    qz = rll_c * ptc_c * yaw_s - rll_s * ptc_s * yaw_c

    # unnormalize quaternion vectors so that the first component is fixed to 1
    qa = qx / (qr + eps)
    qb = qy / (qr + eps)
    qc = qz / (qr + eps)
    quat_vecs = torch.stack([qa, qb, qc], dim=1)

    return quat_vecs.to(tns_type)


def apply_trans(cord_tns_raw, rot_tns_raw, tsl_tns_raw, grouped=False, reverse=False):
    """Apply the global transformation on 3D coordinates.

    Args:
    * cord_tns_raw: 3D coordinates of size M x 3 (grouped: False) or L x M x 3 (grouped: True)
    * rot_tns_raw: rotation matrices of size L x 3 x 3
    * tsl_tns_raw: translation vectors of size L x 3

    Returns:
    * cord_tns_out: projected 3D coordinates of size L x M x 3

    Note:
    * If <grouped> is False, then <cord_tns_raw> should be of size M x 3 (or equivalent size) and
      each coordinate will be transformed multiple times, one per frame, resulting in output 3D
      coordinates of size L x M x 3.
    * If <grouped> is True, then <cord_tns_raw> should be of size L x M x 3 (or equivalent size) and
      each coordinate will be transformed only once (by the corresponding frame), resulting in
      output 3D coordinates of size L x M x 3. This is only useful when computing point components
      of attention affinities in the AlphaFold2's IPA module.
    * If <reverse> is False, then x' = R * x + t; otherwise, x' = R^(-1) * (x - t).
    """

    # re-organize the layout of input arguments
    rot_tns = rot_tns_raw.view(-1, 3, 3)  # L x 3 x 3
    n_frams = rot_tns.shape[0]  # number of local frames
    tsl_tns = tsl_tns_raw.view(n_frams, 3)  # L x 3
    if not grouped:
        cord_tns = cord_tns_raw.view(1, -1, 3)  # 1 x M x 3
    else:
        cord_tns = cord_tns_raw.view(n_frams, -1, 3)  # L x M x 3

    # apply the global transformation
    if not reverse:
        cord_tns_out = tsl_tns.unsqueeze(dim=1) + torch.sum(
            rot_tns.unsqueeze(dim=1) * cord_tns.unsqueeze(dim=2), dim=3)
    else:
        rot_tns_inv = rot_tns.permute(0, 2, 1).unsqueeze(dim=1)  # R x R^T = R^T x R = I
        cord_tns_out = torch.sum(
            rot_tns_inv * (cord_tns - tsl_tns.unsqueeze(dim=1)).unsqueeze(dim=2), dim=3)

    return cord_tns_out
