"""The helper class for calculating the frame aligned point error (FAPE) loss function."""

import logging

import torch
import numpy as np

from zfold.network.af2_smod.utils import quat2rot
from zfold.network.af2_smod.utils import rot2quat
from zfold.network.af2_smod.utils import apply_trans
from zfold.loss.fape.constants import RESD_MAP_1TO3
from zfold.loss.fape.constants import ANGL_INFOS_PER_RESD
from zfold.loss.fape.conversion import cord2fa, fa2cord
from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.dataset.tools import LddtAssessor

class FapeLossHelper():
    """The helper class for calculating the frame aligned point error (FAPE) loss function."""

    def __init__(self):
        """Constructor function."""

        # setup hyper-parameters
        self.eps = 1e-4
        self.dist_clamp = 10.0
        self.loss_w_tors = 1.00
        self.loss_w_norm = 0.02

        # additional configurations
        self.eps = 1e-4
        self.dist_clamp = 10.0
        self.n_bins_lddt = 50  # number of bins for pLDDT-Ca predictions

        # initialize the native structure's information
        self.aa_seq = None  # shared by native & predicted structures
        self.cord_tns = None
        self.cmsk_mat = None  # shared by native & predicted structures
        self.fram_tns = None
        self.fmsk_vec = None  # shared by native & predicted structures
        self.angl_tns = None
        self.amsk_mat = None  # shared by native & predicted structures
        self.params = None

        # initialize the protein structure converter
        self.lddt_assessor = LddtAssessor()


    def preprocess(self, aa_seq, cord_tns, cmsk_mat, addi_msk = None, data_dict_addi=None):
        """Pre-process per-atom 3D coordinates & validness masks.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M
        * data_dict_addi: (optional) additional data dict of idealized structures

        Returns: n/a
        """

        # centralize per-atom 3D coordinates
        cord_vec_avg = \
            torch.sum(cmsk_mat.view(-1, 1) * cord_tns.view(-1, 3), dim=0) / torch.sum(cmsk_mat)
        cord_tns_cen = cord_tns - cord_vec_avg.view(1, 1, 3)

        # generate the idealized structure
        if data_dict_addi is None:
            fram_tns, fmsk_vec, angl_tns, amsk_mat = cord2fa(aa_seq, cord_tns_cen, cmsk_mat)
            cord_tns_reco, cmsk_mat_reco = fa2cord(aa_seq, fram_tns, fmsk_vec, angl_tns, amsk_mat)
        else:
            fram_tns = data_dict_addi['fram']
            fmsk_vec = data_dict_addi['fmsk']
            angl_tns = data_dict_addi['angl']
            amsk_mat = data_dict_addi['amsk']
            cord_tns_reco = data_dict_addi['cord']
            cmsk_mat_reco = data_dict_addi['cmsk']

        # calculate alternative torsion angles w/ symmetry considered
        angl_tns_alt = angl_tns.detach().clone()
        for idx_resd, resd_name in enumerate(aa_seq):
            angl_infos = ANGL_INFOS_PER_RESD[RESD_MAP_1TO3[resd_name]]
            for idx_angl, (_, is_symm, _) in enumerate(angl_infos):
                if is_symm:
                    angl_tns_alt[idx_resd, idx_angl + 3] *= -1.0

        # calculate the averaged distance between original & reconstructed 3D coordinates
        dist_mat = torch.norm(cord_tns_reco - cord_tns_cen, dim=-1)
        dist_avg = torch.sum(cmsk_mat_reco * dist_mat) / (torch.sum(cmsk_mat) + self.eps)
        logging.debug('averaged distance before/after idealization: %.4f', dist_avg)

        # record the current native structure
        self.aa_seq = aa_seq
        self.cord_tns = cord_tns_reco
        self.cmsk_mat = cmsk_mat_reco

        self.fram_tns = fram_tns
        self.fmsk_vec = fmsk_vec
        self.angl_tns = angl_tns
        self.amsk_mat = amsk_mat

        if addi_msk is not None:
            addi_msk = addi_msk.type_as(self.amsk_mat)
            self.cmsk_mat = self.cmsk_mat * addi_msk.unsqueeze(1)
            self.fmsk_vec = self.fmsk_vec * addi_msk
            self.amsk_mat = self.amsk_mat * addi_msk.unsqueeze(1)

        self.params = {
            'quat': rot2quat(fram_tns[:, :3]),  # L x 3
            'trsl': fram_tns[:, 3],  # L x 3
            'angl': angl_tns,  # L x K x 2
            'angl-alt': angl_tns_alt,  # L x K x 2
        }


    def calc_loss(self, params, atom_set='ca', rtn_cord=False):
        """Calculate the loss function w/ QTA parameters.

        Args:
        * params: dict of QTA parameters (must contain 'quat', 'trsl', and 'angl')
        * atom_set: (optional) atom set (choices: 'ca' / 'bb' / 'fa')
        * rtn_cord: (optional) whether to return per-atom 3D coordinates

        Returns:
        * loss: loss function
        * metrics: dict of evaluation metrics
        * cord_tns: (optional) per-atom 3D coordinates of size L x M x 3
        """

        # calculate per-atom 3D coordinates from QTA parameters
        rot_mats = quat2rot(params['quat'])
        fram_tns = torch.cat([rot_mats, params['trsl'].unsqueeze(dim=1)], dim=1)  # L x 4 x 3
        angl_tns = params['angl'] / torch.norm(params['angl'], dim=2, keepdim=True)  # L x K x 2
        cord_tns, _ = fa2cord(self.aa_seq, fram_tns, self.fmsk_vec, angl_tns, self.amsk_mat, atom_set)

        # calculate the loss function & evaluation metrics
        loss_angl, metrics_angl = self.__calc_loss_angl(params['angl'])
        loss_fape, metrics_fape = self.__calc_loss_fape(cord_tns, fram_tns, atom_set)

        metrics = {**metrics_angl, **metrics_fape}

        # calculate the distance RMSD (root-mean-square deviation) for CA atoms
        # metrics['dRMSD'] = self.__calc_drmsd(cord_tns, atom_set='ca')

        return (loss_angl, loss_fape, metrics, cord_tns) if rtn_cord else (loss_angl, loss_fape, metrics)

    def calc_loss_lddt(self, lddt_list, cord_list):
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
            loss_vec = nn.CrossEntropyLoss(reduction='none')(lddt_dict['plddt'], labl_vec)
            loss = torch.sum(plmsk_vec * loss_vec) / (torch.sum(plmsk_vec) + self.eps)
            loss_list.append(loss)
            metrics['lDDT-L%d' % (idx_lyr + 1)] = clddt_val_true.item()

        # loss function & evaluation metrics
        loss = torch.mean(torch.stack(loss_list))
        metrics['Loss-lDDT'] = loss.item()

        return loss, metrics


    def __calc_loss_angl(self, angl_tns):
        """Calculate the torsion angle prediction loss.

        Args:
        * angl_tns: predicted per-residue torsion angles of size L x K x 2

        Returns:
        * loss: loss function
        * metrics: dict of evaluation metrics
        """

        # perform L2-normalization
        norm_mat = torch.norm(angl_tns, dim=-1)
        angl_tns_norm = angl_tns / (norm_mat.unsqueeze(dim=-1) + self.eps)

        # calculate the loss function - torsion
        diff_mat_std = torch.sum(torch.square(angl_tns_norm - self.params['angl']), dim=-1)
        diff_mat_alt = torch.sum(torch.square(angl_tns_norm - self.params['angl-alt']), dim=-1)
        diff_mat = torch.minimum(diff_mat_std, diff_mat_alt)
        loss_tors = torch.sum(self.amsk_mat * diff_mat) / (torch.sum(self.amsk_mat) + self.eps)

        # calculate the loss function - angle norm
        diff_mat = torch.abs(norm_mat - 1.0)
        loss_norm = torch.sum(self.amsk_mat * diff_mat) / (torch.sum(self.amsk_mat) + self.eps)

        # calculate the overall loss function
        loss = self.loss_w_tors * loss_tors + self.loss_w_norm * loss_norm
        metrics = {'Loss-Angl': loss.item()}

        return loss, metrics


    def __calc_loss_fape(self, cord_tns, fram_tns, atom_set):
        """Calculate the frame aligned point error (FAPE) loss.

        Args:
        * cord_tns: predicted per-atom 3D coordinates of size L x M x 3
        * fram_tns: predicted per-residue local frames of size L x 4 x 3
        * atom_set: atom set (choices: 'ca' / 'bb' / 'fa')

        Returns:
        * loss: loss function
        * metrics: dict of evaluation metrics
        """

        # configurations
        n_resds = len(self.aa_seq)
        wo_clamp = (atom_set == 'ca') and (np.random.uniform() <= 0.1)

        # obtain 3D coordinates for the specified atom set
        if atom_set in ['ca', 'bb']:
            atom_names = ['CA'] if atom_set == 'ca' else ['N', 'CA', 'C', 'CB']
            cord_mat_true = ProtStruct.get_atoms(self.aa_seq, self.cord_tns, atom_names).view(-1, 3)
            cord_mat_pred = ProtStruct.get_atoms(self.aa_seq, cord_tns, atom_names).view(-1, 3)
            cmsk_vec = ProtStruct.get_atoms(self.aa_seq, self.cmsk_mat, atom_names).view(-1)
        elif atom_set == 'fa':
            cord_mat_true = self.cord_tns.view(-1, 3)  # (L x M) x 3
            cord_mat_pred = cord_tns.view(-1, 3)  # (L x M) x 3
            cmsk_vec = self.cmsk_mat.view(-1)  # (L x M)
        else:
            raise ValueError('unrecognized atom set: %s' % atom_set)

        # decompose per-residue local frames into rotation matrices & translation vectors
        rot_tns_true, tsl_tns_true = self.fram_tns[:, :3], self.fram_tns[:, 3]
        rot_tns_pred, tsl_tns_pred = fram_tns[:, :3], fram_tns[:, 3]


        # align 3D coordinates under all the per-residue local frames
        n_atoms = cord_mat_true.shape[0]
        cord_tns_true_aln = apply_trans(
            cord_mat_true, rot_tns_true, tsl_tns_true, reverse=True).view(n_resds, n_atoms, 3)
        cord_tns_pred_aln = apply_trans(
            cord_mat_pred, rot_tns_pred, tsl_tns_pred, reverse=True).view(n_resds, n_atoms, 3)

        # calculate the FAPE loss
        dmsk_mat = cmsk_vec.unsqueeze(dim=0).repeat(n_resds, 1)
        dist_mat = torch.sqrt(
            torch.sum(torch.square(cord_tns_true_aln - cord_tns_pred_aln), dim=-1) + self.eps)
        dist_mat_clip = dist_mat if wo_clamp else torch.clip(dist_mat, 0.0, self.dist_clamp)
        dist_mat_norm = dist_mat_clip / self.dist_clamp
        loss = torch.sum(dmsk_mat * dist_mat_norm) / (torch.sum(dmsk_mat) + self.eps)
        metrics = {'Loss-FAPE': loss.item()}

        return loss, metrics


    def __calc_drmsd(self, cord_tns, atom_set):
        """Calculate the distance RMSD (root-mean-square deviation).

        Args:
        * cord_tns: predicted per-atom 3D coordinates of size L x M x 3
        * atom_set: atom set (choices: 'ca' / 'bb' / 'fa')

        Returns:
        * drmsd: distance RMSD (root-mean-square deviation)
        """

        # obtain 3D coordinates for the specified atom set
        if atom_set == 'ca':
            atom_names = ['CA']
            cmsk_vec = ProtStruct.get_atoms(self.aa_seq, self.cmsk_mat, atom_names).view(-1)
            cord_tns_true = ProtStruct.get_atoms(self.aa_seq, self.cord_tns, atom_names).view(1, -1, 3)
            cord_tns_pred = ProtStruct.get_atoms(self.aa_seq, cord_tns, atom_names).view(1, -1, 3)
        elif atom_set == 'bb':
            atom_names = ['N', 'CA', 'C']
            cmsk_vec = ProtStruct.get_atoms(self.aa_seq, self.cmsk_mat, atom_names).view(-1)
            cord_tns_true = ProtStruct.get_atoms(self.aa_seq, self.cord_tns, atom_names).view(1, -1, 3)
            cord_tns_pred = ProtStruct.get_atoms(self.aa_seq, cord_tns, atom_names).view(1, -1, 3)
        elif atom_set == 'fa':
            cmsk_vec = self.cmsk_mat.view(-1)
            cord_tns_true = self.cord_tns.view(1, -1, 3)
            cord_tns_pred = cord_tns.view(1, -1, 3)
        else:
            raise ValueError('unrecognized atom set: %s' % atom_set)

        # calculate the distance RMSD (root-mean-square deviation)
        dmsk_mat = torch.outer(cmsk_vec, cmsk_vec)
        dist_mat_true = torch.cdist(cord_tns_true, cord_tns_true).squeeze(dim=0)
        dist_mat_pred = torch.cdist(cord_tns_pred, cord_tns_pred).squeeze(dim=0)
        diff_mat = torch.abs(dist_mat_pred - dist_mat_true)
        drmsd = torch.sum(dmsk_mat * diff_mat) / (torch.sum(dmsk_mat) + self.eps)

        return drmsd.item()
