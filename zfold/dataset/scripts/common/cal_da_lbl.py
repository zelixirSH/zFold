"""Build additional da lbl files."""

import os
import logging
import torch
import numpy as np
from scipy.spatial.distance import cdist

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

def build_npz_file(aa_seq, cord_tns, mask_mat, npz_fpath):
    """Build a NPZ file w/ ground-truth labels for inter-residue distance & orientation."""

    if os.path.exists(npz_fpath):
        logging.info('NPZ file exists: %s', npz_fpath)
        return

    # initialization
    n_resds = len(aa_seq)
    atom_names = ['N', 'CA', 'CB']

    # obtain 3D coordinates for N/CA/CB atoms
    cord_tns_sel = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)
    mask_mat_sel = ProtStruct.get_atoms(aa_seq, mask_mat, atom_names)
    x_n, x_ca, x_cb = [x.squeeze_().numpy() for x in torch.split(cord_tns_sel, 1, dim=1)]
    m_n, m_ca, m_cb = [x.squeeze_().numpy() for x in torch.split(mask_mat_sel, 1, dim=1)]

    # use GLY's CA atom as the replacement for its missing CB atom
    is_gly = np.array([1 if aa_seq[x] == 'G' else 0 for x in range(n_resds)], dtype=np.int8)
    x_cab = is_gly[:, None] * x_ca + (1 - is_gly[:, None]) * x_cb
    m_cab = is_gly * m_ca + (1 - is_gly) * m_cb

    # build ground-truth labels
    labl_data = {}
    labl_data['cb-val'] = cdist(x_cab, x_cab).astype(np.float16)
    labl_data['cb-msk'] = np.outer(m_cab, m_cab).astype(np.int8)
    labl_data['om-val'] = np.zeros((n_resds, n_resds), dtype=np.float16)
    labl_data['om-msk'] = np.zeros((n_resds, n_resds), dtype=np.int8)
    labl_data['th-val'] = np.zeros((n_resds, n_resds), dtype=np.float16)
    labl_data['th-msk'] = np.zeros((n_resds, n_resds), dtype=np.int8)
    labl_data['ph-val'] = np.zeros((n_resds, n_resds), dtype=np.float16)
    labl_data['ph-msk'] = np.zeros((n_resds, n_resds), dtype=np.int8)
    for ir in range(n_resds):
        for ic in range(n_resds):
            labl_data['om-val'][ir, ic] = calc_dihedral_angle(x_ca[ir], x_cb[ir], x_cb[ic], x_ca[ic])
            labl_data['th-val'][ir, ic] = calc_dihedral_angle(x_n[ir], x_ca[ir], x_cb[ir], x_cb[ic])
            labl_data['ph-val'][ir, ic] = calc_plane_angle(x_ca[ir], x_cb[ir], x_cb[ic])
            labl_data['om-msk'][ir, ic] = m_ca[ir] * m_cb[ir] * m_cb[ic] * m_ca[ic]
            labl_data['th-msk'][ir, ic] = m_n[ir] * m_ca[ir] * m_cb[ir] * m_cb[ic]
            labl_data['ph-msk'][ir, ic] = m_ca[ir] * m_cb[ir] * m_cb[ic]

    # build a NPZ file
    os.makedirs(os.path.dirname(os.path.realpath(npz_fpath)), exist_ok=True)
    np.savez(npz_fpath, **labl_data)
    logging.info('NPZ file built: %s', npz_fpath)
