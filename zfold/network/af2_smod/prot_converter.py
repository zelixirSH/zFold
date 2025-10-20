"""Protein structure converter."""

import os
import torch
from collections import defaultdict

from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.dataset.tools.prot_constants import RESD_MAP_1TO3
from zfold.dataset.tools.prot_constants import RESD_NAMES_3C
from zfold.dataset.tools.prot_constants import N_ATOMS_PER_RESD_MAX
from zfold.dataset.tools.prot_constants import N_ANGLS_PER_RESD_MAX
from zfold.dataset.tools.prot_constants import ATOM_NAMES_PER_RESD
from zfold.dataset.tools.prot_constants import ATOM_INFOS_PER_RESD
from zfold.dataset.tools.prot_constants import ANGL_INFOS_PER_RESD
from zfold.network.af2_smod.utils import calc_rot_n_tsl
from zfold.network.af2_smod.utils import calc_rot_n_tsl_batch
from zfold.network.af2_smod.utils import calc_dihd_angl_batch

from zfold.network.af2_smod.utils import quat2rot as quat2rot_part

class ProtConverter():
    """Protein structure converter."""

    def __init__(self):
        """Constructor function."""

        # side-chain rigid-groups' names (padded according to ARG and LYS)
        self.rgrp_names_pad = ['omega', 'phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4']

        # convert per-atom 3D coordinates into torch.Tensor
        self.cord_dict = defaultdict(dict)
        for resd_name in RESD_NAMES_3C:
            for atom_name, _, cord_vals in ATOM_INFOS_PER_RESD[resd_name]:
                self.cord_dict[resd_name][atom_name] = torch.tensor(cord_vals, dtype=torch.float32)

        # build base transformations (angle = 0) for side-chain frames
        self.trans_dict_base = self.__build_trans_dict_base()

        # additional configurations
        self.eps = 1e-4


    def param2cord(self, aa_seq, params, atom_set='fa', rtn_cmsk=False):
        """Convert QTA parameters into per-atom 3D coordinates.

        Args:
        * aa_seq: amino-acid sequence
        * params: dict of QTA parameters (must contain 'quat', 'trsl', and 'angl')
          > quat: per-residue quaternion vectors of size L x 4 (full) / L x 3 (part)
          > trsl: per-residue translation vectors size size L x 3
          > angl: per-residue torsion angles of size L x K x 2
        * atom_set: (optional) atom set (choices: 'ca' / 'fa')

        Returns:
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M
        * fram_tns_sc: per-residue side-chain local frames of size L x K x 4 x 3

        Note:
        If <atom_set> is 'fa', then side-chain local frames will be calculated; otherwise, it will
        be set to None to keep the consistency of returned values.
        """

        # initialization
        n_resds = len(aa_seq)
        device = params['quat'].device
        n_dims_quat = params['quat'].shape[-1]
        quat2rot_fn = quat2rot_full if n_dims_quat == 4 else quat2rot_part

        # assume all the per-residue local frame & side-chain torsion angles are valid
        fmsk_mat_bb = torch.ones((n_resds, 1), dtype=torch.int8, device=device)
        amsk_mat = torch.ones((n_resds, N_ANGLS_PER_RESD_MAX), dtype=torch.int8, device=device)

        # construct per-residue backbone local frames & torsion angles
        rot_mats = quat2rot_fn(params['quat'])
        fram_tns_bb = torch.cat([rot_mats, params['trsl'].unsqueeze(dim=1)], dim=1).unsqueeze(dim=1)
        angl_tns = params['angl'] / (torch.norm(params['angl'], dim=2, keepdim=True) + self.eps)

        # calculate 3D coordinates (and side-chain local frames)
        assert atom_set in ['ca', 'fa'], 'unrecognized atom set: ' + atom_set
        if atom_set == 'ca':
            cord_tns, cmsk_mat = \
                self.fa2cord(aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat, atom_set)
            fram_tns_sc = None
        else:  # then <atom_set> must be 'fa'
            cord_tns, cmsk_mat, fram_tns_sc, _ = \
                self.fa2cord(aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat, atom_set)

        return (cord_tns, cmsk_mat, fram_tns_sc) if rtn_cmsk else (cord_tns, fram_tns_sc)


    def cord2fa(self, aa_seq, cord_tns, cmsk_mat):
        """Convert per-atom 3D coordinates to per-residue backbone local frames & torsion angles.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

        Returns:
        * fram_tns: per-residue backbone local frames of size L x 1 x 4 x 3
        * fmsk_mat: per-residue backbone local frames' validness masks of size L x 1
        * angl_tns: per-residue torsion angles of size L x K x 2
        * amsk_mat: per-residue torsion angles' validness masks of size L x K
        """

        # initialization
        device = cord_tns.device

        # force to use CPU-based conversion routine (faster!)
        if device != torch.device('cpu'):
            fram_tns, fmsk_mat, angl_tns, amsk_mat = \
                self.cord2fa(aa_seq, cord_tns.cpu(), cmsk_mat.cpu())
            return fram_tns.to(device), fmsk_mat.to(device), angl_tns.to(device), amsk_mat.to(device)

        # convert per-atom 3D coordinates to per-residue backbone local frames & torsion angles
        fram_tns, fmsk_mat = self.__cord2fram(aa_seq, cord_tns, cmsk_mat)
        angl_tns, amsk_mat = self.__cord2angl(aa_seq, cord_tns, cmsk_mat)

        return fram_tns, fmsk_mat, angl_tns, amsk_mat


    def fa2cord(self, aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat, atom_set='fa'):
        """Convert per-residue backbone local frames & torsion angles to per-atom 3D coordinates.

        Args:
        * aa_seq: amino-acid sequence
        * fram_tns_bb: per-residue backbone local frames of size L x 1 x 4 x 3
        * fmsk_mat_bb: per-residue backbone local frames' validness masks of size L x 1
        * angl_tns: per-residue torsion angles of size L x K x 2
        * amsk_mat: per-residue torsion angles' validness masks of size L x K
        * atom_set: (optional) atom set (choices: 'ca' / 'fa')

        Returns:
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M
        * fram_tns_sc: (optional) per-residue side-chain local frames of size L x K x 4 x 3
        * fmsk_mat_sc: (optional) per-residue side-chain local frames' validness masks of size L x K

        Note:
        If <atom_set> is 'fa', then both <fram_tns_sc> and <fmsk_mat_sc> will be returned.
        """

        # initialization
        device = fram_tns_bb.device
        n_resds = fram_tns_bb.shape[0]

        # validate input arguments
        assert atom_set in ['ca','bb','fa'], 'unrecognized atom set: ' + atom_set

        # take the shortcut if only CA atoms are considered
        if atom_set == 'ca':
            idx_atom_ca = 1  # CA atom is always the 2nd atom, regardless of the residue type
            cord_tns = torch.zeros((n_resds, N_ATOMS_PER_RESD_MAX, 3), dtype=torch.float32, device=device)
            cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD_MAX), dtype=torch.int8, device=device)
            cord_tns[:, idx_atom_ca] = fram_tns_bb[:, 0, 3]  # CA: backbone frame's origin point
            cmsk_mat[:, idx_atom_ca] = fmsk_mat_bb[:, 0]
            return cord_tns, cmsk_mat

        # force to use CPU-based conversion routine (faster!)
        if device != torch.device('cpu'):
            cord_tns, cmsk_mat, fram_tns_sc, fmsk_mat_sc = self.fa2cord(
                aa_seq, fram_tns_bb.cpu(), fmsk_mat_bb.cpu(), angl_tns.cpu(), amsk_mat.cpu(), atom_set)
            return cord_tns.to(device), cmsk_mat.to(device), fram_tns_sc.to(device), fmsk_mat_sc.to(device)

        # initialize 3D coordinates & validness masks
        cord_tns = torch.zeros((n_resds, N_ATOMS_PER_RESD_MAX, 3), dtype=torch.float32, device=device)
        cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD_MAX), dtype=torch.int8, device=device)
        fram_tns_sc = torch.zeros((n_resds, N_ANGLS_PER_RESD_MAX, 4, 3), dtype=torch.float32, device=device)
        fmsk_mat_sc = torch.zeros((n_resds, N_ANGLS_PER_RESD_MAX), dtype=torch.int8, device=device)

        # take the short-cut if only CA atoms are needed
        if atom_set == 'ca':
            idx_atom_ca = 1  # CA atom is always the 2nd atom, regardless of the residue type
            cord_tns[:, idx_atom_ca] = fram_tns_bb[:, 0, 3]  # CA: backbone frame's origin point
            cmsk_mat[:, idx_atom_ca] = fmsk_mat_bb[:, 0]
            return cord_tns, cmsk_mat

        # enumerate over all the residue types
        for resd_name_1c, resd_name_3c in RESD_MAP_1TO3.items():
            #print("zlz: ", aa_seq, len(aa_seq), n_resds, "resd_name_1c", resd_name_1c)
            idxs = [x for x in range(n_resds) if aa_seq[x] == resd_name_1c]
            if len(idxs) == 0:
                continue
            cord_tns[idxs], cmsk_mat[idxs], fram_tns_sc[idxs], fmsk_mat_sc[idxs] = \
                self.__fa2cord_impl(resd_name_3c, fram_tns_bb[idxs], fmsk_mat_bb[idxs], angl_tns[idxs], amsk_mat[idxs],
                                    bb_only= True if atom_set == 'bb' else False )

        return cord_tns, cmsk_mat, fram_tns_sc, fmsk_mat_sc


    def __build_trans_dict_base(self):
        """Build base transformations (angle = 0) for side-chain frames."""

        trans_dict_full = {}
        for resd_name in RESD_NAMES_3C:
            # initialization
            trans_dict = {}
            atom_infos = ATOM_INFOS_PER_RESD[resd_name]  # list of (atom name, RG index, cord.)
            angl_infos = ANGL_INFOS_PER_RESD[resd_name]  # list of (angl name, symm, atom names)
            n_angls = len(angl_infos)

            # initialize backbone atoms' 3D coordinates w.r.t. the backbone frame
            cord_dict = {}
            for atom_name, idx_rgrp, _ in atom_infos:
                if idx_rgrp == 0:  # backbone rigid-group
                    cord_dict[atom_name] = self.cord_dict[resd_name][atom_name]

            # build the pre-omega to backbone transformation
            trans_dict['omega-bb'] = (
                torch.eye(3, dtype=torch.float32),
                torch.zeros((3), dtype=torch.float32),
            )  # identity mapping

            # build the phi to backbone transformation
            x1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            x2 = cord_dict['N']
            x3 = cord_dict['CA']
            rot_mat, tsl_vec = calc_rot_n_tsl(x1, x2, x2 + (x2 - x3))
            trans_dict['phi-bb'] = (rot_mat, tsl_vec)

            # build psi & chiX to backbone transformations
            for idx_angl, (angl_name, _, atom_names_sel) in enumerate(angl_infos):
                # print(idx_angl, angl_name, atom_names_sel, cord_dict.keys())
                # calculate the rotation matrix and translation vector
                x1 = cord_dict[atom_names_sel[0]]
                x2 = cord_dict[atom_names_sel[1]]
                x3 = cord_dict[atom_names_sel[2]]
                rot_mat, tsl_vec = calc_rot_n_tsl(x1, x3, x3 + (x3 - x2))
                trans_dict['%s-bb' % angl_name] = (rot_mat, tsl_vec)

                # transform all the atoms in the current rigid-group to the backbone frame
                for atom_name, idx_rgrp, _ in atom_infos:
                    # print(atom_name, idx_rgrp)
                    if idx_rgrp == idx_angl + 3:  # 0: backbone / 1: omega / 2: phi
                        cord_dict[atom_name] = tsl_vec + torch.sum(
                            rot_mat * self.cord_dict[resd_name][atom_name].view(1, 3), dim=1)

            # build chiX+1 to chiX transformations
            for idx_angl_src in range(1, n_angls - 1):  # skip the psi angle
                idx_angl_dst = idx_angl_src + 1
                angl_name_src = angl_infos[idx_angl_src][0]
                angl_name_dst = angl_infos[idx_angl_dst][0]
                rot_mat_src, tsl_vec_src = trans_dict['%s-bb' % angl_name_src]
                rot_mat_dst, tsl_vec_dst = trans_dict['%s-bb' % angl_name_dst]
                rot_mat = torch.matmul(rot_mat_src.transpose(1, 0), rot_mat_dst)
                tsl_vec = torch.matmul(rot_mat_src.transpose(1, 0), tsl_vec_dst - tsl_vec_src)
                trans_dict['%s-%s' % (angl_name_dst, angl_name_src)] = (rot_mat, tsl_vec)

            # record the transformation dict for the current residue type
            trans_dict_full[resd_name] = trans_dict

        return trans_dict_full


    @classmethod
    def __cord2fram(cls, aa_seq, cord_tns, cmsk_mat):
        """Convert per-atom 3D coordinates to per-residue backbone local frames."""

        # initialization
        device = cord_tns.device
        n_resds = cord_tns.shape[0]

        # gather N/CA/C-atom 3D coordinates, denoted as x0, x1, and x2, respectively
        atom_names = ['N', 'CA', 'C']
        cord_tns_sel = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)  # L x 3 x 3
        cmsk_mat_sel = ProtStruct.get_atoms(aa_seq, cmsk_mat, atom_names)  # L x 3

        # compute per-residue local frames (rotation matrix + translation vector)
        fmsk_mat = torch.prod(cmsk_mat_sel, dim=1, keepdim=True).to(torch.int8)  # L x 1
        rot_tns, tsl_mat = calc_rot_n_tsl_batch(cord_tns_sel)
        fram_tns = fmsk_mat.view(-1, 1, 1) * torch.cat([rot_tns, tsl_mat.unsqueeze(dim=1)], dim=1)
        fram_tns[fmsk_mat.view(-1) == 0, :3] = torch.eye(3, dtype=torch.float32, device=device)
        fram_tns = fram_tns.view(n_resds, 1, 4, 3)  # one backbone frame per residue

        return fram_tns, fmsk_mat


    @classmethod
    def __cord2angl(cls, aa_seq, cord_tns, cmsk_mat):
        """Convert per-atom 3D coordinates to per-residue torsion angles."""

        # initialization
        device = cord_tns.device
        n_resds = cord_tns.shape[0]

        # get CA & C atoms' 3D coordinates (with 1-residue offset) to compute omega & phi angles
        cord_mat_ca = ProtStruct.get_atoms(aa_seq, cord_tns, ['CA'])
        cmsk_vec_ca = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])
        cord_mat_c = ProtStruct.get_atoms(aa_seq, cord_tns, ['C'])
        cmsk_vec_c = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['C'])

        cord_mat_cap = torch.cat([torch.zeros_like(cord_mat_ca[:1]), cord_mat_ca[:-1]], dim=0)
        cmsk_vec_cap = torch.cat([torch.zeros_like(cmsk_vec_ca[:1]), cmsk_vec_ca[:-1]], dim=0)
        cord_mat_cp = torch.cat([torch.zeros_like(cord_mat_c[:1]), cord_mat_c[:-1]], dim=0)
        cmsk_vec_cp = torch.cat([torch.zeros_like(cmsk_vec_c[:1]), cmsk_vec_c[:-1]], dim=0)


        # determine atom indices and validness masks for each residue type
        amsk_vec_dict = {}  # validness masks of torsion angles
        idxs_vec_dict = {}  # atom indices for each torsion angle
        for resd_name in RESD_NAMES_3C:
            # initialization
            atom_names_all = ATOM_NAMES_PER_RESD[resd_name]
            amsk_vec = torch.zeros((N_ANGLS_PER_RESD_MAX), dtype=torch.int8, device=device)
            idxs_mat = torch.zeros((N_ANGLS_PER_RESD_MAX, 4), dtype=torch.int64, device=device)

            # omega (CA_p - C_p - N - CA)
            amsk_vec[0] = 1  # always valid (except for the 1st residue, which will be handled later)

            idxs_mat[0, 2] = atom_names_all.index('N')
            idxs_mat[0, 3] = atom_names_all.index('CA')

            # phi (C_p - N - CA - C)
            amsk_vec[1] = 1  # always valid (except for the 1st residue, which will be handled later)

            idxs_mat[1, 1] = atom_names_all.index('N')
            idxs_mat[1, 2] = atom_names_all.index('CA')
            idxs_mat[1, 3] = atom_names_all.index('C')

            # psi, chi1, chi2, chi3, and chi4
            for idx_rgrp, (_, _, atom_names_sel) in enumerate(ANGL_INFOS_PER_RESD[resd_name]):
                amsk_vec[idx_rgrp + 2] = 1
                for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                    idxs_mat[idx_rgrp + 2, idx_atom_sel] = atom_names_all.index(atom_name_sel)

            # record atom indices and validness masks for the current residue type
            amsk_vec_dict[resd_name] = amsk_vec
            idxs_vec_dict[resd_name] = idxs_mat.view(-1)

        # expand atom indices and validness masks into the per-residue basis
        amsk_vec_list = []
        idxs_vec_list = []
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            amsk_vec_list.append(amsk_vec_dict[resd_name_3c])
            idxs_vec_list.append(idxs_vec_dict[resd_name_3c])
        amsk_mat_base = torch.stack(amsk_vec_list, dim=0)  # L x K
        idxs_mat_full = torch.stack(idxs_vec_list, dim=0)  # L x (K x 4)
        amsk_mat_base[0, :2] = 0  # 1st residue does not have omega & phi angles

        # extract 3D coordinates & validness masks for torsion angle computation
        cord_tns_ext = torch.gather(
            cord_tns, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, 3),
        ).view(n_resds, N_ANGLS_PER_RESD_MAX, 4, 3)

        cord_tns_ext[:, 0, 0] = cord_mat_cap  # omega - CA_prev
        cord_tns_ext[:, 0, 1] = cord_mat_cp   # omega - C_prev
        cord_tns_ext[:, 1, 0] = cord_mat_cp   # phi - C_prev

        cmsk_tns_ext = torch.gather(cmsk_mat, 1, idxs_mat_full).view(n_resds, N_ANGLS_PER_RESD_MAX, 4)

        cmsk_tns_ext[:, 0, 0] = cmsk_vec_cap  # omega - CA_prev
        cmsk_tns_ext[:, 0, 1] = cmsk_vec_cp   # omega - C_prev
        cmsk_tns_ext[:, 1, 0] = cmsk_vec_cp   # omega - C_prev

        amsk_mat = (torch.prod(cmsk_tns_ext, dim=-1) * amsk_mat_base).to(torch.int8)

        # compute torsion angles from 3D coordinates (as cosine & sine values)
        angl_vec = calc_dihd_angl_batch(cord_tns_ext.view(-1, 4, 3))
        angl_tns = amsk_mat.unsqueeze(dim=-1) * torch.stack(
            [torch.cos(angl_vec), torch.sin(angl_vec)], dim=-1).view(n_resds, N_ANGLS_PER_RESD_MAX, 2)

        return angl_tns, amsk_mat


    def __fa2cord_impl(self, resd_name, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat, bb_only = False):
        """Convert per-residue backbone local frames & torsion angles to per-atom 3D coordinates."""

        # initialization
        device = fram_tns_bb.device
        n_resds = fram_tns_bb.shape[0]
        atom_names_all = ATOM_NAMES_PER_RESD[resd_name]  # list of atom names
        atom_names_pad = atom_names_all + ['X'] * (N_ATOMS_PER_RESD_MAX - len(atom_names_all))
        atom_infos_all = ATOM_INFOS_PER_RESD[resd_name]  # list of (atom name, RG index, cord.) tuples
        angl_infos_all = ANGL_INFOS_PER_RESD[resd_name]  # list of (angl name, symm, atom names) tuples
        rgrp_names_all = ['omega', 'phi'] + [x[0] for x in angl_infos_all]

        # initialize side-chain local frames & 3D coordinates
        fram_mat_null = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32, device=device)
        fram_tns_dict = defaultdict(lambda: fram_mat_null.unsqueeze(dim=0).repeat(n_resds, 1, 1))
        fmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))
        cord_mat_dict = defaultdict(
            lambda: torch.zeros((n_resds, 3), dtype=torch.float32, device=device))
        cmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        # initialize the dict of rigid-body transformations
        trans_dict = {'bb': (fram_tns_bb[:, 0, :3], fram_tns_bb[:, 0, 3])}  # backbone frame

        # determine 3D coordinates of atoms belonging to the backbone rigid-group
        rot_tns_curr, tsl_mat_curr = trans_dict['bb']
        atom_names_sel = [x[0] for x in atom_infos_all if x[1] == 0]
        for atom_name_sel in atom_names_sel:
            cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)
            cord_mat_dict[atom_name_sel] = \
                tsl_mat_curr + torch.sum(rot_tns_curr * cord_vec.view(1, 1, 3), dim=2)
            cmsk_vec_dict[atom_name_sel] = fmsk_mat_bb[:, 0]

        # determine 3D coordinates of atoms belonging to side-chain rigid-groups
        if not bb_only:
            for idx_rgrp, rgrp_name_curr in enumerate(rgrp_names_all):
                # print(idx_rgrp, rgrp_name_curr)
                # determine the previous rigid-group
                if rgrp_name_curr in ['omega', 'phi', 'psi', 'chi1']:
                    rgrp_name_prev = 'bb'
                else:
                    rgrp_name_prev = 'chi%d' % (int(rgrp_name_curr[-1]) - 1)

                # obtain the relative transformation w.r.t. the previous rigid-body
                rot_tns_prev, tsl_mat_prev = trans_dict[rgrp_name_prev]
                # obtain the pre-calculated base transform
                rot_mat_base, tsl_vec_base = \
                    self.trans_dict_base[resd_name]['%s-%s' % (rgrp_name_curr, rgrp_name_prev)]
                rot_tns_base = rot_mat_base.unsqueeze(dim=0)  # will be automatically broadcasted
                tsl_mat_base = tsl_vec_base.unsqueeze(dim=0)  # will be automatically broadcasted

                rot_tns_addi, tsl_mat_addi = self.__build_trans_from_angl(angl_tns[:, idx_rgrp])
                # combine_trans previous and current
                rot_tns_curr, tsl_mat_curr = self.__combine_trans(
                    rot_tns_prev, tsl_mat_prev, rot_tns_base, tsl_mat_base, rot_tns_addi, tsl_mat_addi)
                trans_dict[rgrp_name_curr] = (rot_tns_curr, tsl_mat_curr)

                # record the current side-chain local frame
                fram_tns_dict[rgrp_name_curr] = \
                    torch.cat([rot_tns_curr, tsl_mat_curr.unsqueeze(dim=1)], dim=1)
                fmsk_vec_dict[rgrp_name_curr] = fmsk_mat_bb[:, 0] * amsk_mat[:, idx_rgrp]

                # map idealized 3D coordinates to the current rigid-group frame
                atom_names_sel = [x[0] for x in atom_infos_all if x[1] == idx_rgrp + 1]
                # print(rgrp_name_prev, rgrp_name_curr, idx_rgrp, idx_rgrp+1, atom_names_sel)
                for atom_name_sel in atom_names_sel:
                    cord_vec = self.cord_dict[resd_name][atom_name_sel].to(device)
                    cord_mat_dict[atom_name_sel] = \
                        tsl_mat_curr + torch.sum(rot_tns_curr * cord_vec.view(1, 1, 3), dim=2)
                    cmsk_vec_dict[atom_name_sel] = fmsk_vec_dict[rgrp_name_curr]

        # packing 3D coordinates & side-chain local frames into tensors
        cmsk_mat = torch.stack([cmsk_vec_dict[x] for x in atom_names_pad], dim=1)
        cord_tns = torch.stack([cord_mat_dict[x] for x in atom_names_pad], dim=1)
        fram_tns_sc = torch.stack([fram_tns_dict[x] for x in self.rgrp_names_pad], dim=1)
        fmsk_mat_sc = torch.stack([fmsk_vec_dict[x] for x in self.rgrp_names_pad], dim=1)

        return cord_tns, cmsk_mat, fram_tns_sc, fmsk_mat_sc


    @classmethod
    def __combine_trans(cls, rot_tns_1, tsl_mat_1, rot_tns_2, tsl_mat_2, *args):
        """Combine two or more transformations."""

        # combine the first two transformations
        #rot_tns = torch.bmm(rot_tns_1, rot_tns_2)  # much slower!
        rot_tns = torch.sum(rot_tns_1.unsqueeze(dim=3) * rot_tns_2.unsqueeze(dim=1), dim=2)
        tsl_mat = torch.sum(rot_tns_1 * tsl_mat_2.unsqueeze(dim=1), dim=2) + tsl_mat_1

        # recursively process remaining transformations
        if len(args) > 0:
            assert len(args) % 2 == 0, \
                'rotation matrices and translation vectors must be provided simultaneously'
            return cls.__combine_trans(rot_tns, tsl_mat, *args)

        return rot_tns, tsl_mat


    @classmethod
    def __build_trans_from_angl(cls, angl_mat):
        """Build rigid-body transformations from angles (represented as cosine & sine values)."""

        # initialization
        device = angl_mat.device
        n_resds = angl_mat.shape[0]

        # build rigid-body transformations
        one_vec = torch.ones((n_resds), dtype=torch.float32, device=device)
        zro_vec = torch.zeros((n_resds), dtype=torch.float32, device=device)
        cos_vec = angl_mat[:, 0]
        sin_vec = angl_mat[:, 1]
        rot_tns = torch.stack([
            torch.stack([one_vec, zro_vec, zro_vec], dim=1),
            torch.stack([zro_vec, cos_vec, -sin_vec], dim=1),
            torch.stack([zro_vec, sin_vec, cos_vec], dim=1),
        ], dim=1)
        tsl_mat = torch.zeros((n_resds, 3), dtype=torch.float32, device=device)

        return rot_tns, tsl_mat
