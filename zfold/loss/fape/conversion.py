"""Structure conversion routines in AlphaFold2."""

from collections import defaultdict

import torch

from zfold.network.af2_smod.utils import calc_rot_n_tsl_batch
from zfold.network.af2_smod.utils import calc_dihd_angl_batch
from zfold.loss.fape.constants import RESD_MAP_1TO3
from zfold.loss.fape.constants import RESD_NAMES_3C
from zfold.loss.fape.constants import N_ATOMS_PER_RESD_MAX
from zfold.loss.fape.constants import N_ANGLS_PER_RESD_MAX
from zfold.loss.fape.constants import ATOM_NAMES_PER_RESD
from zfold.loss.fape.constants import ATOM_INFOS_PER_RESD
from zfold.loss.fape.constants import ANGL_INFOS_PER_RESD
from zfold.network.af2_smod.prot_struct import ProtStruct


def fa2cord_impl(resd_name, fram_tns, fmsk_vec, angl_tns, amsk_mat, atom_set):
    """Convert per-residue local frames & torsion angles to per-atom 3D coordinates - core implementation."""

    # initialization
    device = fram_tns.device
    n_resds = fram_tns.shape[0]
    atom_names_all = ATOM_NAMES_PER_RESD[resd_name]  # list of atom names
    atom_names_pad = atom_names_all + ['X'] * (N_ATOMS_PER_RESD_MAX - len(atom_names_all))
    atom_infos_all = ATOM_INFOS_PER_RESD[resd_name]  # list of (atom name, RG index, cord.) tuples
    angl_infos_all = ANGL_INFOS_PER_RESD[resd_name]  # list of (angl name, symm, atom names) tuples

    # initialize 3D coordinates & validness masks, indexed by the atom name
    cord_mat_dict = defaultdict(
        lambda: torch.zeros((n_resds, 3), dtype=torch.float32, device=device))
    cmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

    # determine 3D coordinates of CA atoms
    cord_mat_dict['CA'] = fram_tns[:, 3]  # CA atom is the origin point of backbone frame
    cmsk_vec_dict['CA'] = fmsk_vec

    # determine 3D coordinates of other atoms belonging to the backbone rigid-group
    if atom_set in ['bb', 'fa']:
        rot_tns, tsl_mat = fram_tns[:, :3], fram_tns[:, 3]
        atom_infos_sel = [x for x in atom_infos_all if x[0] != 'CA' and x[1] == 0]
        for atom_name_sel, _, cord_vals in atom_infos_sel:
            cord_vec = torch.tensor(cord_vals, dtype=torch.float32, device=device)
            cord_mat_dict[atom_name_sel] = \
                tsl_mat + torch.sum(rot_tns * cord_vec.view(1, 1, 3), dim=2)
            cmsk_vec_dict[atom_name_sel] = fmsk_vec

    # determine 3D coordinates of atoms belonging to side-chain rigid-groups
    if atom_set == 'fa':
        for idx_rgrp, (_, _, atom_names_sel) in enumerate(angl_infos_all):
            # obtain validness masks for the current rigid-group
            cmsk_vec = fmsk_vec * amsk_mat[:, idx_rgrp + 2]  # skip the first two torsion angles

            # calculate rotation matrices & translation vectors from the first 3 atoms
            x0 = cord_mat_dict[atom_names_sel[0]]
            x1 = cord_mat_dict[atom_names_sel[1]]
            x2 = cord_mat_dict[atom_names_sel[2]]
            cord_tns_sel = torch.stack([x0, x2, x2 + (x2 - x1)], dim=1)
            rot_tns_init, tsl_mat = calc_rot_n_tsl_batch(cord_tns_sel)

            # determine final rotation matrices w/ dihedral angle considered
            one_vec = torch.ones((n_resds), dtype=torch.float32, device=device)
            zro_vec = torch.zeros((n_resds), dtype=torch.float32, device=device)
            cos_vec = angl_tns[:, idx_rgrp + 2, 0]
            sin_vec = angl_tns[:, idx_rgrp + 2, 1]
            rot_tns_addi = torch.stack([
                torch.stack([one_vec, zro_vec, zro_vec], dim=1),
                torch.stack([zro_vec, cos_vec, -sin_vec], dim=1),
                torch.stack([zro_vec, sin_vec, cos_vec], dim=1),
            ], dim=1)
            rot_tns_finl = torch.sum(
                rot_tns_init.view(n_resds, 3, 3, 1) * rot_tns_addi.view(n_resds, 1, 3, 3), dim=2)

            # map idealized 3D coordinates to the current rigid-group frame
            atom_infos_sel = [x for x in atom_infos_all if x[1] == idx_rgrp + 3]
            for atom_name_sel, _, cord_vals in atom_infos_sel:
                cord_vec = torch.tensor(cord_vals, dtype=torch.float32, device=device)
                cord_mat_dict[atom_name_sel] = \
                    tsl_mat + torch.sum(rot_tns_finl * cord_vec.view(1, 1, 3), dim=2)
                cmsk_vec_dict[atom_name_sel] = cmsk_vec

    # packing 3D coordinates & validness masks into tensors
    cmsk_mat = torch.stack([cmsk_vec_dict[x] for x in atom_names_pad], dim=1)
    cord_tns = cmsk_mat.unsqueeze(dim=-1) * \
        torch.stack([cord_mat_dict[x] for x in atom_names_pad], dim=1)

    return cord_tns, cmsk_mat


def fa2cord(aa_seq, fram_tns, fmsk_vec, angl_tns, amsk_mat, atom_set='fa'):
    """Convert per-residue local frames & torsion angles to per-atom 3D coordinates.

    Args:
    * aa_seq: amino-acid sequence
    * fram_tns: per-residue local frames of size L x 4 x 3
    * fmsk_vec: per-residue local frames' validness masks of size L
    * angl_tns: per-residue torsion angles of size L x K x 2
    * amsk_mat: per-residue torsion angles' validness masks of size L x K
    * atom_set: (optional) atom set (choices: 'ca' / 'bb' / 'fa')

    Returns:
    * cord_tns: per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Note:
    * This function handles the whole amino-acid sequence, which should be used in most cases.
    """

    # force to use CPU-based conversion routine (faster!)
    if fram_tns.device != torch.device('cpu'):
        device = fram_tns.device
        cord_tns, cmsk_mat = fa2cord(
            aa_seq, fram_tns.cpu(), fmsk_vec.cpu(), angl_tns.cpu(), amsk_mat.cpu(), atom_set)
        return cord_tns.to(device), cmsk_mat.to(device)

    # initialization
    device = fram_tns.device
    n_resds = fram_tns.shape[0]

    # initialize 3D coordinates & validness masks
    cord_tns = torch.zeros((n_resds, N_ATOMS_PER_RESD_MAX, 3), dtype=torch.float32, device=device)
    cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD_MAX), dtype=torch.int8, device=device)

    # enumerate over all the residue types
    for resd_name_1c, resd_name_3c in RESD_MAP_1TO3.items():
        idxs = [x for x in range(n_resds) if aa_seq[x] == resd_name_1c]
        if len(idxs) == 0:
            continue
        cord_tns[idxs], cmsk_mat[idxs] = fa2cord_impl(
            resd_name_3c, fram_tns[idxs], fmsk_vec[idxs], angl_tns[idxs], amsk_mat[idxs], atom_set)
    cord_tns = cord_tns * cmsk_mat.unsqueeze(dim=-1)

    return cord_tns, cmsk_mat


def cord2fram(aa_seq, cord_tns, cmsk_mat):
    """Convert per-atom 3D coordinates to per-residue local frames.

    Args:
    * aa_seq: amino-acid sequence
    * cord_tns: per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * fram_tns: per-residue local frames of size L x 4 x 3
    * fmsk_vec: per-residue local frames' validness masks of size L
    """

    # initialization
    device = cord_tns.device
    n_resds = cord_tns.shape[0]

    # gather N/CA/C-atom 3D coordinates, denoted as x0, x1, and x2, respectively
    atom_names = ['N', 'CA', 'C']
    cord_tns_sel = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)  # L x 3 x 3
    cmsk_mat_sel = ProtStruct.get_atoms(aa_seq, cmsk_mat, atom_names)  # L x 3

    # compute per-residue local frames (rotation matrix + translation vector)
    fmsk_vec = torch.prod(cmsk_mat_sel, dim=1).to(torch.int8)  # L
    rot_tns, tsl_mat = calc_rot_n_tsl_batch(cord_tns_sel)
    fram_tns = fmsk_vec.view(n_resds, 1, 1) * torch.cat([rot_tns, tsl_mat.unsqueeze(dim=1)], dim=1)
    fram_tns[fmsk_vec == 0, :3] = torch.eye(3, dtype=torch.float32, device=device)

    return fram_tns, fmsk_vec


def cord2angl(aa_seq, cord_tns, cmsk_mat):
    """Convert per-atom 3D coordinates to per-residue torsion angles.

    Args:
    * aa_seq: amino-acid sequence
    * cord_tns: per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * angl_tns: per-residue torsion angles of size L x K x 2
    * amsk_mat: per-residue torsion angles' validness masks of size L x K
    """

    # initialization
    device = cord_tns.device
    n_resds = cord_tns.shape[0]

    # build the indexing tensor for all the atoms defining torsion angles
    amsk_vec_dict = {}  # validness masks of torsion angles
    idxs_vec_dict = {}  # atom indices for each torsion angle
    for resd_name in RESD_NAMES_3C:
        atom_names_all = ATOM_NAMES_PER_RESD[resd_name]
        amsk_vec = torch.zeros((N_ANGLS_PER_RESD_MAX), dtype=torch.int8, device=device)
        idxs_mat = torch.zeros((N_ANGLS_PER_RESD_MAX, 4), dtype=torch.int64, device=device)
        for idx_rgrp, (_, _, atom_names_sel) in enumerate(ANGL_INFOS_PER_RESD[resd_name]):
            amsk_vec[idx_rgrp + 2] = 1
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                idxs_mat[idx_rgrp + 2, idx_atom_sel] = atom_names_all.index(atom_name_sel)
        amsk_vec_dict[resd_name] = amsk_vec
        idxs_vec_dict[resd_name] = idxs_mat.view(-1)

    # expand per-atom 3D coordinates & validness masks into per-angle basis
    amsk_vec_list = []
    idxs_vec_list = []
    for idx_resd, resd_name_1c in enumerate(aa_seq):
        resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
        amsk_vec_list.append(amsk_vec_dict[resd_name_3c])
        idxs_vec_list.append(idxs_vec_dict[resd_name_3c])
    amsk_mat_base = torch.stack(amsk_vec_list, dim=0)  # L x K
    idxs_mat_full = torch.stack(idxs_vec_list, dim=0)  # L x (K x 4)
    cord_tns_ext = torch.gather(
        cord_tns, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, 3),
    ).view(n_resds * N_ANGLS_PER_RESD_MAX, 4, 3)
    cmsk_tns_ext = torch.gather(cmsk_mat, 1, idxs_mat_full).view(n_resds, N_ANGLS_PER_RESD_MAX, 4)
    amsk_mat = (torch.prod(cmsk_tns_ext, dim=-1) * amsk_mat_base).to(torch.int8)

    # compute torsion angles from 3D coordinates
    angl_tns_raw = calc_dihd_angl_batch(cord_tns_ext).view(n_resds, N_ANGLS_PER_RESD_MAX)
    angl_tns = amsk_mat.unsqueeze(dim=-1) * \
        torch.stack([torch.cos(angl_tns_raw), torch.sin(angl_tns_raw)], dim=-1)

    return angl_tns, amsk_mat


def cord2fa(aa_seq, cord_tns, cmsk_mat):
    """Convert per-atom 3D coordinates to per-residue local frames & torsion angles.

    Args:
    * aa_seq: amino-acid sequence
    * cord_tns: per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * fram_tns: per-residue local frames of size L x 4 x 3
    * fmsk_vec: per-residue local frames' validness masks of size L
    * angl_tns: per-residue torsion angles of size L x K x 2
    * amsk_mat: per-residue torsion angles' validness masks of size L x K
    """

    # force to use CPU-based conversion routine (faster!)
    if cord_tns.device != torch.device('cpu'):
        device = cord_tns.device
        fram_tns, fmsk_vec, angl_tns, amsk_mat = cord2fa(aa_seq, cord_tns.cpu(), cmsk_mat.cpu())
        return fram_tns.to(device), fmsk_vec.to(device), angl_tns.to(device), amsk_mat.to(device)

    # convert per-atom 3D coordinates to per-residue local frames & torsion angles
    fram_tns, fmsk_vec = cord2fram(aa_seq, cord_tns, cmsk_mat)
    angl_tns, amsk_mat = cord2angl(aa_seq, cord_tns, cmsk_mat)

    return fram_tns, fmsk_vec, angl_tns, amsk_mat
