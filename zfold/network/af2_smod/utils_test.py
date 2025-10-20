"""Unit-tests for utility functions."""

import logging

import torch

from zfold.dataset.utils import zfold_init
from zfold.network.af2_smod.utils import quat2rot
from zfold.network.af2_smod.utils import rot2quat
from zfold.network.af2_smod.utils import apply_trans

def main():
    """Main entry."""

    # configurations
    n_frams = 64
    n_atoms = 256

    # initialization
    zfold_init(verb_levl='DEBUG')

    # test the conversion between partial quaternion vectors and rotation matrices
    quat_vecs_old = torch.randn((n_frams, 3), dtype=torch.float32)
    rot_mats = quat2rot(quat_vecs_old)
    quat_vecs_new = rot2quat(rot_mats)
    logging.info('diff. in quaternion vectors: %.4f', torch.norm(quat_vecs_new - quat_vecs_old))

    # test the global transformation on 3D coordinates
    cord_tns_old = torch.randn((n_atoms, 3), dtype=torch.float32)
    quat_tns = torch.randn((1, 3), dtype=torch.float32)
    trsl_tns = torch.randn((1, 3), dtype=torch.float32)
    rot_tns = quat2rot(quat_tns)
    cord_tns_med = apply_trans(cord_tns_old, rot_tns, trsl_tns).view(n_atoms, 3)
    cord_tns_new = apply_trans(cord_tns_med, rot_tns, trsl_tns, reverse=True).view(n_atoms, 3)
    logging.info('diff. in 3D coordinates: %.4f', torch.norm(cord_tns_new - cord_tns_old))


if __name__ == '__main__':
    main()
