"""The helper class for calculating the loss and evaluation metrics related to <AF2SMod>.

List of loss functions:
> FAPE: frame aligned point error
> LDDT: classification loss for per-residue lDDT-Ca predictions
> QNrm: L2-norm regularization loss for quaternion vector predictions
> ANrm: L2-norm regularization loss for torsion angle matrix predictions
> Clsh: structural violation loss (steric clashes among non-bonded atoms)
"""

import os

import numpy as np
import torch
from torch import nn

from zfold.dataset.utils.math_utils import cdist
from zfold.network.af2_smod.utils import quat2rot as quat2rot_part
from zfold.network.af2_smod.utils import apply_trans
from zfold.loss.fape.constants import RESD_MAP_1TO3, ATOM_NAMES_PER_RESD, ANGL_INFOS_PER_RESD, \
    N_ATOMS_PER_RESD_MAX
from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.network.af2_smod.prot_converter import ProtConverter
from zfold.dataset.tools import LddtAssessor

def load_clsh_params(path):
    """Load steric clash check parameters for distance between non-bonded atoms."""

    params = {}
    enbl_read = False
    with open(path, 'r', encoding='UTF-8') as i_file:
        for i_line in i_file:
            if i_line.startswith('Non-bonded distance'):
                enbl_read = True
            elif i_line.startswith('-'):
                enbl_read = False
            elif enbl_read:
                atom_pair, dist_min, dist_tlr = i_line.split()
                params[atom_pair] = float(dist_min) - float(dist_tlr)

    return params


class AF2LossHelper():
    """The helper class for calculating the loss and evaluation metrics related to <AF2SMod>."""

    def __init__(
            self,
            wc_fape=1.0,       # weighting coefficient for Loss-FAPE
            wc_lddt=0.1,       # weighting coefficient for Loss-LDDT
            wc_qnrm=0.02,      # weighting coefficient for Loss-QNrm
            wc_anrm=0.02,      # weighting coefficient for Loss-ANrm
            wc_clsh=1.0,       # weighting coefficient for Loss-Clsh
            quat_type='none',  # type of quaternion vectors (choices: 'full' / 'part')
            alter_angl=True,   # whether to enable alternative torsion angles
        ):
        """Constructor function."""

        # basic configurations
        self.wc_fape = wc_fape
        self.wc_lddt = wc_lddt
        self.wc_qnrm = wc_qnrm
        self.wc_anrm = wc_anrm
        self.wc_clsh = wc_clsh
        self.quat_type = quat_type
        self.alter_angl = alter_angl

        # additional configurations
        self.eps = 1e-4
        self.dist_clamp = 10.0
        self.n_bins_lddt = 50  # number of bins for pLDDT-Ca predictions
        self.n_atoms = N_ATOMS_PER_RESD_MAX
        self.quat2rot = quat2rot_full if self.quat_type == 'full' else quat2rot_part
        # self.quat2rot = quat2rot

        # load steric clash check parameters
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        prm_fpath = os.path.join(curr_dir, 'data/stereo_chemical_props.txt')
        self.params = load_clsh_params(prm_fpath)

        # initialize the native structure's information (note: all the validness masks are shared)
        self.aa_seq = None
        self.cord_tns_ca = None  # CA-atom 3D coordinates
        self.cmsk_mat_ca = None
        self.cord_tns_fa = None  # full-atom 3D coordinates
        self.cmsk_mat_fa = None
        self.fram_tns_bb = None  # backbone local frames
        self.fmsk_mat_bb = None
        self.fram_tns_sc = None  # side-chain local frames
        self.fmsk_mat_sc = None
        self.angl_tns_bsc = None  # basic torsion angles
        self.angl_tns_alt = None  # alternative torsion angles
        self.angl_tns_opt = None  # optimal torsion angles (best fit w/ the predicted structure)
        self.amsk_mat = None
        self.dthr_tns = None  # distance threshold for steric clashes
        self.dmsk_tns = None

        # initialize the protein structure converter
        self.lddt_assessor = LddtAssessor()
        self.prot_converter = ProtConverter()


    def init(self, aa_seq, cord_tns, cmsk_mat, data_dict_addi=None):
        """Initialize the helper class w/ the native structure.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: native structure's per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M
        * data_dict_addi: (optional) additional data dict of idealized structures

        Returns: n/a
        """

        # fp32
        cord_tns = cord_tns.float()

        # centralize per-atom 3D coordinates
        cord_vec_avg = torch.sum(
            cmsk_mat.view(-1, 1) * cord_tns.view(-1, 3), dim=0) / (torch.sum(cmsk_mat) + self.eps)
        cord_tns_cen = cord_tns - cord_vec_avg.view(1, 1, 3)  # raw structure

        # calculate per-residue local frames & torsion angles and idealized 3D structure
        if data_dict_addi is None:
            fram_tns_bb, fmsk_mat_bb, angl_tns_bsc, amsk_mat = \
                self.prot_converter.cord2fa(aa_seq, cord_tns_cen, cmsk_mat)
        else:
            fram_tns_bb = data_dict_addi['fram']
            fmsk_mat_bb = data_dict_addi['fmsk']
            angl_tns_bsc = data_dict_addi['angl']
            amsk_mat = data_dict_addi['amsk']

        # reconstruct CA-atom 3D coordinates
        idx_atom_ca = 1  # CA atom is always the 2nd atom, regardless of the residue type
        cord_tns_ca = torch.zeros_like(cord_tns)
        cmsk_mat_ca = torch.zeros_like(cmsk_mat)
        cord_tns_ca[:, idx_atom_ca] = fram_tns_bb[:, 0, 3]
        cmsk_mat_ca[:, idx_atom_ca] = fmsk_mat_bb[:, 0]

        # calculate alternative torsion angles w/ 180-degree symmetry considered
        angl_tns_alt = angl_tns_bsc.detach().clone()
        for idx_resd, resd_name in enumerate(aa_seq):
            angl_infos = ANGL_INFOS_PER_RESD[RESD_MAP_1TO3[resd_name]]
            for idx_angl, (_, is_symm, _) in enumerate(angl_infos):
                if is_symm:
                    angl_tns_alt[idx_resd, idx_angl + 2] *= -1.0  # flip both cosine & sine values

        # build the distance threshold tensor and its validness masks
        dthr_tns, dmsk_tns = self.__build_dthr_n_dmsk(aa_seq, cord_tns.device)

        # record the current native structure
        self.aa_seq = aa_seq
        self.cord_tns_ca = cord_tns_ca
        self.cmsk_mat_ca = cmsk_mat_ca
        self.fram_tns_bb = fram_tns_bb
        self.fmsk_mat_bb = fmsk_mat_bb
        self.angl_tns_bsc = angl_tns_bsc
        self.angl_tns_alt = angl_tns_alt
        self.amsk_mat = amsk_mat
        self.dthr_tns = dthr_tns
        self.dmsk_tns = dmsk_tns


    def calc_loss(self, params_list, lddt_list, cord_list, fram_tns_sc, is_hr_sample = True):
        """Calculate the loss function and evaluation metrics.

        Args:
        * params_list: list of QTA parameters, one per layer
          > quat_tns: quaternion vectors of size L x 4 (full) or L x 3 (part)
          > trsl_tns: translation vectors of size L x 3
          > angl_tns: torsion angle matrices of size L x K x 2
        * lddt_list: list of per-residue & full-chain lDDT-Ca predictions, one per layer
          > plddt_mat: per-residue lDDT-Ca predictions of size L x 50
          > clddt_val: full-chain lDDT-Ca prediction (scalar)
        * cord_list: list of per-atom 3D coordinates of size L x M x 3, one per layer
        * fram_tns_sc: final layer's per-residue side-chain frames of size L x K x 4 x 3

        Returns:
        * loss: loss function
        * metrics: dict of evaluation metrics

        Note:
        In <cord_list>, only the last entry contains full-atom 3D coordinates, while all the other
          entries only contain C-Alpha atoms' 3D coordinates.
        """

        # initialization
        loss_list = []
        metrics = {}
        self.seq_len = len(self.aa_seq)

        # calculate the frame aligned point error (FAPE)
        if self.wc_fape > 0.0:
            loss_fape, metrics_fape = self.__calc_loss_fape(params_list, cord_list, fram_tns_sc)
            loss_list.append(self.wc_fape * loss_fape)
            metrics.update(**metrics_fape)

        # calculate the classification loss for per-residue lDDT-Ca predictions
        if self.wc_lddt > 0.0 and is_hr_sample:
            loss_lddt, metrics_lddt = self.__calc_loss_lddt(lddt_list, cord_list)
            loss_list.append(self.wc_lddt * loss_lddt)
            metrics.update(**metrics_lddt)

        # calculate the L2-norm regularization loss for quaternion vector predictions
        if (self.wc_qnrm > 0.0) and (self.quat_type == 'full'):
            loss_qnrm, metrics_qnrm = self.__calc_loss_qnrm(params_list)
            loss_list.append(self.wc_qnrm * loss_qnrm)
            metrics.update(**metrics_qnrm)

        # calculate the L2-norm regularization loss for torsion angle matrix predictions
        if self.wc_anrm > 0.0:
            loss_anrm, metrics_anrm = self.__calc_loss_anrm(params_list)
            loss_list.append(self.wc_anrm * loss_anrm)
            metrics.update(**metrics_anrm)

        # calculate the structural violation loss (steric clashes among non-bonded atoms)
        if self.wc_clsh > 0.0:
            loss_clsh, metrics_clsh = self.__calc_loss_clsh(cord_list[-1])
            loss_list.append(self.wc_clsh * loss_clsh)
            metrics.update(**metrics_clsh)

        # aggregate all the loss functions and evaluation metrics
        loss = torch.sum(torch.stack(loss_list))
        metrics['Loss'] = loss.item()

        # print(metrics)
        return loss, metrics


    def __calc_loss_fape(self, params_list, cord_list, fram_tns_sc):
        """Calculate the frame aligned point error (FAPE)."""

        # initialization
        metrics = {}
        # calculate the FAPE loss w/ CA-atom and backbone frames
        loss_ca_list = []
        for idx_lyr, (params, cord_tns) in enumerate(zip(params_list, cord_list)):
            loss_ca, _ = self.__calc_loss_fape_impl(params, cord_tns, atom_set='ca', fram_set='bb')
            loss_ca_list.append(loss_ca)
            metrics['dRMSD-L%d' % (idx_lyr + 1)] = self.__calc_drmsd(cord_tns, atom_set='ca')
        loss_ca = torch.mean(torch.stack(loss_ca_list))

        # calculate the FAPE loss w/ full-atom and backbone & side-chain frames
        loss_fa, metrics_fa = self.__calc_loss_fape_impl(
            params_list[-1], cord_list[-1], atom_set='fa', fram_set='bs', fram_tns_sc=fram_tns_sc)
        metrics.update(**metrics_fa)

        # calculate the overall loss function
        loss = loss_ca + loss_fa
        metrics['Loss-FAPE'] = loss.item()

        return loss, metrics


    def __calc_loss_lddt(self, lddt_list, cord_list):
        """Calculate the classification loss for per-residue lDDT-Ca predictions."""

        # calculate the classification loss for each layer's per-residue lDDT-Ca predictions
        loss_list, metrics = [], {}
        for idx_lyr, (lddt_dict, cord_tns) in enumerate(zip(lddt_list, cord_list)):
            # calculate ground-truth per-residue lDDT-Ca scores
            plddt_vec_true, plmsk_vec, clddt_val_true = \
                self.lddt_assessor.run(self.cord_tns_fa, cord_tns, self.cmsk_mat_fa, atom_set='ca')
            labl_vec = torch.clip(torch.floor(
                self.n_bins_lddt * plddt_vec_true).to(torch.int64), 0, self.n_bins_lddt - 1)

            # calculate the classification loss
            loss_vec = nn.CrossEntropyLoss(reduction='none')(lddt_dict[0].squeeze(0), labl_vec)
            loss = torch.sum(plmsk_vec * loss_vec) / (torch.sum(plmsk_vec) + self.eps)
            loss_list.append(loss)
            metrics['lDDT-L%d' % (idx_lyr + 1)] = clddt_val_true.item()

        # loss function & evaluation metrics
        loss = torch.mean(torch.stack(loss_list))
        metrics['Loss-lDDT'] = loss.item()

        return loss, metrics


    def __calc_loss_qnrm(self, params_list):
        """Calculate the L2-norm regularization loss for quaternion vector predictions."""

        # calculate the L2-norm regularization loss for each layer's quaternion vectors
        loss_list = []
        for params in params_list:
            quat_tns = params['quat-u']  # L x 4
            loss = torch.mean(torch.abs(torch.norm(quat_tns, dim=-1) - 1.0))
            loss_list.append(loss)

        # take the averaged value of all the L2-norm regularization losses
        loss = torch.mean(torch.stack(loss_list))
        metrics = {'Loss-QNrm': loss.item()}

        return loss, metrics


    def __calc_loss_anrm(self, params_list):
        """Calculate the L2-norm regularization loss for torsion angle matrix predictions."""

        # calculate the L2-norm regularization loss for each layer's torsion angle matrices
        loss_list = []
        for params in params_list:
            #TODO
            quat_tns, trsl_tns, angl_tns = params
            params = {
                'quat': quat_tns.view(len(self.aa_seq), -1).float(),
                'trsl': trsl_tns.view(len(self.aa_seq), 3).float(),
                'angl': angl_tns.view(len(self.aa_seq), -1, 2).float(),
            }

            angl_tns = params['angl']  # L x K x 2
            loss = torch.mean(torch.abs(torch.norm(angl_tns, dim=-1) - 1.0))
            loss_list.append(loss)

        # take the averaged value of all the L2-norm regularization losses
        loss = torch.mean(torch.stack(loss_list))
        metrics = {'Loss-ANrm': loss.item()}

        return loss, metrics


    def __calc_loss_clsh(self, cord_tns):
        """Calculate the structural violation loss (steric clashes among non-bonded atoms)."""

        # initialization
        n_resds = cord_tns.shape[0]

        # calculate the structural violation loss
        dist_tns = cdist(cord_tns.view(-1, 3)).view(n_resds, self.n_atoms, n_resds, self.n_atoms)
        loss_tns = self.dmsk_tns * torch.clamp(self.dthr_tns - dist_tns, min=0.0)
        loss = torch.mean(torch.sum(loss_tns, dim=(2, 3)))
        metrics = {'Loss-Clsh': loss.item()}

        return loss, metrics


    def __calc_loss_fape_impl(self, params, cord_tns, atom_set, fram_set, fram_tns_sc=None):
        """Calculate the frame aligned point error (FAPE) loss - core implementation.

        Args:
        * params: dict of QTA parameters (must contain 'quat', 'trsl', and 'angl')
          > quat: per-residue quaternion vectors of size L x 4 (full) / L x 3 (part)
          > trsl: per-residue translation vectors size size L x 3
          > angl: per-residue torsion angles of size L x K x 2
        * cord_tns: predicted per-atom 3D coordinates of size L x M x 3
        * atom_set: atom set (choices: 'ca' / 'fa')
        * fram_set: frame set (choices: 'bb' / 'bs')
        * fram_tns_sc: (optional) final layer's per-residue side-chain frames of size L x K x 4 x 3

        Returns:
        * loss: loss function
        * metrics: dict of evaluation metrics
        """

        # initialization
        wo_clamp = (atom_set == 'ca') and (np.random.uniform() <= 0.1)
        assert atom_set in ['ca', 'fa'], 'unrecognized atom set: ' + atom_set
        assert fram_set in ['bb', 'bs'], 'unrecognized frame set: ' + fram_set

        #TODO
        quat_tns, trsl_tns, angl_tns = params
        params = {
            'quat': quat_tns.view(len(self.aa_seq), -1).float(),
            'trsl': trsl_tns.view(len(self.aa_seq), 3).float(),
            'angl': angl_tns.view(len(self.aa_seq), -1, 2).float(),
        }

        # calculate the torsion angle prediction loss
        angl_tns = params['angl'] / (torch.norm(params['angl'], dim=-1, keepdim=True) + self.eps)
        if not self.alter_angl:
            self.angl_tns_opt = self.angl_tns_bsc
        else:
            with torch.no_grad():
                dist_tns_bsc = torch.norm(angl_tns - self.angl_tns_bsc, dim=-1, keepdim=True)
                dist_tns_alt = torch.norm(angl_tns - self.angl_tns_alt, dim=-1, keepdim=True)
                self.angl_tns_opt = torch.where(
                    torch.less(dist_tns_bsc, dist_tns_alt), self.angl_tns_bsc, self.angl_tns_alt)
        diff_mat = torch.norm(angl_tns - self.angl_tns_opt, dim=-1).square()
        loss_angl = torch.sum(self.amsk_mat * diff_mat) / (torch.sum(self.amsk_mat) + self.eps)

        # reconstruct the native structure's full-atom 3D coordinates & side-chain local frames
        if (atom_set == 'fa') or (fram_set == 'bs'):
            self.cord_tns_fa, self.cmsk_mat_fa, self.fram_tns_sc, self.fmsk_mat_sc = \
                self.prot_converter.fa2cord(self.aa_seq, self.fram_tns_bb, self.fmsk_mat_bb, self.angl_tns_opt, self.amsk_mat)

        # obtain 3D coordinates for the specified atom set
        if atom_set == 'ca':
            atom_names = ['CA']
            cord_mat_true = ProtStruct.get_atoms(self.aa_seq, self.cord_tns_ca, atom_names).view(-1, 3)
            cord_mat_pred = ProtStruct.get_atoms(self.aa_seq, cord_tns, atom_names).view(-1, 3)
            cmsk_vec = ProtStruct.get_atoms(self.aa_seq, self.cmsk_mat_ca, atom_names).view(-1)
        else:  # then <atom_set> must be 'fa'
            cord_mat_true = self.cord_tns_fa.view(-1, 3)  # (L x M) x 3
            cord_mat_pred = cord_tns.view(-1, 3)  # (L x M) x 3
            cmsk_vec = self.cmsk_mat_fa.view(-1)  # (L x M)

        # obtain local frames for the specified frame set
        rot_mats = self.quat2rot(params['quat'])
        fram_tns_bb = torch.cat([rot_mats, params['trsl'].unsqueeze(dim=1)], dim=1).unsqueeze(dim=1)
        if fram_set == 'bb':
            fram_tns_true = self.fram_tns_bb.view(-1, 4, 3)
            fram_tns_pred = fram_tns_bb.view(-1, 4, 3)
            fmsk_vec = self.fmsk_mat_bb.view(-1)
        else:  # then <fram_set> must be 'bs'
            assert fram_tns_sc is not None, 'side-chain local frames is not provided'
            fram_tns_true = torch.cat([self.fram_tns_bb, self.fram_tns_sc], dim=1).view(-1, 4, 3)
            fram_tns_pred = torch.cat([fram_tns_bb, fram_tns_sc], dim=1).view(-1, 4, 3)
            fmsk_vec = torch.cat([self.fmsk_mat_bb, self.fmsk_mat_sc], dim=1).view(-1)

        # decompose per-residue local frames into rotation matrices & translation vectors
        rot_tns_true, tsl_mat_true = fram_tns_true[:, :3], fram_tns_true[:, 3]
        rot_tns_pred, tsl_mat_pred = fram_tns_pred[:, :3], fram_tns_pred[:, 3]

        # align 3D coordinates under all the per-residue local frames
        n_atoms = cord_mat_true.shape[0]
        n_frams = fram_tns_true.shape[0]
        cord_tns_true_aln = apply_trans(
            cord_mat_true, rot_tns_true, tsl_mat_true, reverse=True).view(n_frams, n_atoms, 3)
        cord_tns_pred_aln = apply_trans(
            cord_mat_pred, rot_tns_pred, tsl_mat_pred, reverse=True).view(n_frams, n_atoms, 3)

        # calculate the FAPE loss
        dmsk_mat = torch.outer(fmsk_vec, cmsk_vec)
        dist_mat = torch.sqrt(
            torch.sum(torch.square(cord_tns_true_aln - cord_tns_pred_aln), dim=-1) + self.eps)
        dist_mat_clip = dist_mat if wo_clamp else torch.clip(dist_mat, 0.0, self.dist_clamp)
        dist_mat_norm = dist_mat_clip / self.dist_clamp
        loss_cord = torch.sum(dmsk_mat * dist_mat_norm) / (torch.sum(dmsk_mat) + self.eps)
        loss = loss_angl + loss_cord

        # record each loss function's value
        metrics = {
            'Loss-FAPE-Angl': loss_angl.item(),
            'Loss-FAPE-Cord': loss_cord.item(),
            'Loss-FAPE': loss.item(),
        }

        return loss, metrics


    def __calc_drmsd(self, cord_tns, atom_set):
        """Calculate the distance RMSD (root-mean-square deviation).

        Args:
        * cord_tns: predicted per-atom 3D coordinates of size L x M x 3
        * atom_set: atom set (choices: 'ca' / 'fa')

        Returns:
        * drmsd: distance RMSD (root-mean-square deviation)
        """

        # obtain 3D coordinates for the specified atom set
        if atom_set == 'ca':
            atom_names = ['CA']
            cord_tns_true = ProtStruct.get_atoms(self.aa_seq, self.cord_tns_ca, atom_names).view(-1, 3)
            cord_tns_pred = ProtStruct.get_atoms(self.aa_seq, cord_tns, atom_names).view(-1, 3)
            cmsk_vec = ProtStruct.get_atoms(self.aa_seq, self.cmsk_mat_ca, atom_names).view(-1)
        else:  # then <atom_set> must be 'fa'
            cord_tns_true = self.cord_tns_fa.view(-1, 3)
            cord_tns_pred = cord_tns.view(-1, 3)
            cmsk_vec = self.cmsk_mat_fa.view(-1)

        # calculate the distance RMSD (root-mean-square deviation)
        dist_mat_true = cdist(cord_tns_true)
        dist_mat_pred = cdist(cord_tns_pred)
        dmsk_mat = torch.outer(cmsk_vec, cmsk_vec)
        diff_mat = torch.abs(dist_mat_pred - dist_mat_true)
        drmsd = torch.sum(dmsk_mat * diff_mat) / (torch.sum(dmsk_mat) + self.eps)

        return drmsd.item()


    def __build_dthr_n_dmsk(self, aa_seq, device):
        """Build the distance threshold tensor and its validness masks.

        Note:
        Intra-residue atoms should not have any violations of distance between non-bonded atoms.
        Thus, we only need to check for inter-residue atoms (except for N-C atoms in peptide bonds).
        """

        # initialization
        n_resds = len(aa_seq)

        # find all the possible element & element-pair types
        elem_type_unk = 'X'
        elem_types = sorted(list({elem_type_unk} | {x[0] for x in self.params}))
        n_elems = len(elem_types)
        idx_elem_unk = elem_types.index(elem_type_unk)

        # convert the (pair-of-elements, dist-thres) dict into a vector
        wei_mat = torch.zeros((n_elems, n_elems), dtype=torch.float32, device=device)
        for key, val in self.params.items():
            elem_type_pri, elem_type_sec = key.split('-')
            idx_elem_pri = elem_types.index(elem_type_pri)
            idx_elem_sec = elem_types.index(elem_type_sec)
            wei_mat[idx_elem_pri, idx_elem_sec] = val
            wei_mat[idx_elem_sec, idx_elem_pri] = val
        wei_vec = wei_mat.view(-1)

        # encode atoms into one-hot encodings based on element types
        idxs_elem_mat = idx_elem_unk * torch.ones((n_resds, self.n_atoms), dtype=torch.int8, device=device)
        for idx_resd, resd_name in enumerate(aa_seq):
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            idxs_elem_mat[idx_resd, :len(atom_names)] = torch.tensor([elem_types.index(x[0]) for x in atom_names], device=device)
        idxs_elem_tns = n_elems * idxs_elem_mat.view(n_resds, self.n_atoms, 1, 1) + idxs_elem_mat.view(1, 1, n_resds, self.n_atoms)

        # build the distance threshold tensor
        onht_tns = torch.nn.functional.one_hot(idxs_elem_tns.to(torch.int64), num_classes=(n_elems ** 2))
        dthr_tns = torch.sum(wei_vec.view(1, 1, 1, 1, -1) * onht_tns, dim=-1)

        # build the distance threshold tensor' validness masks
        amsk_mat = torch.zeros((n_resds, self.n_atoms), dtype=torch.int8, device=device)
        for idx_resd, resd_name in enumerate(aa_seq):
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            amsk_mat[idx_resd, :len(atom_names)] = 1
        dmsk_tns = amsk_mat.view(n_resds, self.n_atoms, 1, 1) * amsk_mat.view(1, 1, n_resds, self.n_atoms)
        dmsk_tns *= (1 - torch.eye(n_resds, dtype=torch.int8, device=device).view(n_resds, 1, n_resds, 1))
        for idx_resd_curr in range(1, n_resds):
            idx_resd_prev = idx_resd_curr - 1
            atom_names_prev = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa_seq[idx_resd_prev]]]
            atom_names_curr = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa_seq[idx_resd_curr]]]
            idx_atom_prev = atom_names_prev.index('C')
            idx_atom_curr = atom_names_curr.index('N')
            dmsk_tns[idx_resd_prev, idx_atom_prev, idx_resd_curr, idx_atom_curr] = 0
            dmsk_tns[idx_resd_curr, idx_atom_curr, idx_resd_prev, idx_atom_prev] = 0

        return dthr_tns, dmsk_tns
