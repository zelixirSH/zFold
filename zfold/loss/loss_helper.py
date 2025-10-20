"""The helper class for computing various loss functions."""

import logging

import numpy as np
import torch
from torch import nn

from zfold.loss.af2_loss_helper import AF2LossHelper

class LossHelper():
    """The helper class for computing various loss functions."""

    def __init__(
            self,
            wc_da=1.0,          # weighting coefficient for inter-residue DA predictions
            wc_lm=1.0,          # weighting coefficient for masked MSA predictions
            wc_fape=1.0,        # weighting coefficient for frame aligned point error (FAPE)
            wc_lddt=0.1,        # weighting coefficient for per-residue lDDT-Ca predictions
            wc_qnrm=0.02,       # weighting coefficient for L2-norm loss on quaternion vectors
            wc_anrm=0.02,       # weighting coefficient for L2-norm loss on torsion angle metrices
            wc_clsh=1.0,        # weighting coefficient for structural violation loss
            scheme_da='multi',  # weighting scheme for inter-residue DA predictions
            crop_size=128,      # size of random crops
            quat_type='none',   # type of quaternion vectors (choices: 'full' / 'part')
            alter_angl=True,    # whether to enable alternative torsion angles
        ):
        """Constructor function."""

        # setup hyper-parameters
        self.wc_da = wc_da
        self.wc_lm = wc_lm
        self.wc_fape = wc_fape
        self.wc_lddt = wc_lddt
        self.wc_qnrm = wc_qnrm
        self.wc_anrm = wc_anrm
        self.wc_clsh = wc_clsh
        self.scheme_da = scheme_da
        self.crop_size = crop_size
        self.quat_type = quat_type
        self.alter_angl = alter_angl

        # additional configurations
        self.eps = 1e-6
        self.n_bins_lddt = 50
        self.wc_af2 = self.wc_fape + self.wc_lddt + self.wc_qnrm + self.wc_anrm + self.wc_clsh
        self.bin_vals = (torch.arange(self.n_bins_lddt) + 0.5) / self.n_bins_lddt
        self.af2_loss_helper = AF2LossHelper(
            wc_fape=self.wc_fape,
            wc_lddt=self.wc_lddt,
            wc_qnrm=self.wc_qnrm,
            wc_anrm=self.wc_anrm,
            wc_clsh=self.wc_clsh,
            quat_type=self.quat_type,
            alter_angl=self.alter_angl,
        )


    def calc_loss(self, inputs, outputs, is_hr_sample = True):
        """Calculate the loss function & evaluation metrics.

        Args:
        * inputs: dict of input tensors (see above to detailed requirements)
        * outputs: dict of output tensors (see above for detailed requirements)

        Returns:
        * loss: loss function
        * metrics: dict of evaluation metrics
        """

        # initialization
        loss_list = []
        metrics = {}

        # evaluate the loss function for inter-residue DA predictions
        if self.wc_da > 0.0:
            loss_da, metrics_da = self.__calc_loss_da(inputs['labl'], outputs)
            loss_list.append(self.wc_da * loss_da)
            metrics.update(**metrics_da)  # only use the last one

        # evaluate the loss function for masked MSA predictions
        if self.wc_lm > 0.0:
            loss_lm, metrics_lm = self.__calc_loss_lm(outputs['msa-t'], outputs['msa-m'],
                                                      outputs['lm'])
            loss_list.append(self.wc_lm * loss_lm)
            metrics.update(**metrics_lm)  # only use the last one

        # evaluate the loss function related to <AF2SMod>
        if self.wc_af2 > 0.0:
            self.af2_loss_helper.init(
                inputs['base']['seq'][0],
                inputs['base']['cord'][0],
                inputs['base']['cmsk'][0],
            )
            loss_af2, metrics_af2 = self.af2_loss_helper.calc_loss(
                params_list = outputs['af2_smod_param'],
                lddt_list = outputs['plddt'],
                cord_list = outputs['cords'],
                fram_tns_sc = outputs['fram_tns_sc'],
                is_hr_sample = is_hr_sample,
            )
            loss_list.append(loss_af2)
            metrics.update(**metrics_af2)

        # determine the overall re-weighting coefficient
        alpha = 1.0  # np.sqrt(min(seq_len, self.crop_size) / self.crop_size)
        # aggregate all the loss functions and evaluation metrics
        loss = alpha * torch.sum(torch.stack(loss_list))
        metrics['Loss'] = loss.item()

        return loss, metrics


    def __calc_loss_da(self, labl_dict, pred_dict):
        """Calculate the loss function for inter-residue DA predictions."""

        # configurations
        n_bins_pos = 12  # from bin #1 (2.0-2.5A) to bin #12 (7.5-8.0A) - zero-based indexing

        # calculate weighting coefficients for all the residue pairs
        if self.scheme_da == 'uniform':
            wc_tns_cb = labl_dict['cb-msk'] * torch.ones_like(labl_dict['cb-msk'])
            wc_tns_om = labl_dict['om-msk'] * torch.ones_like(labl_dict['om-msk'])
            wc_tns_th = labl_dict['th-msk'] * torch.ones_like(labl_dict['th-msk'])
            wc_tns_ph = labl_dict['ph-msk'] * torch.ones_like(labl_dict['ph-msk'])
        elif self.scheme_da == 'binary':
            wc_tns_cb = self.__calc_wc_tns_bc(labl_dict['cb-idx'], labl_dict['cb-msk'], n_bins_pos)
            wc_tns_om = labl_dict['om-msk'] * wc_tns_cb
            wc_tns_th = labl_dict['th-msk'] * wc_tns_cb
            wc_tns_ph = labl_dict['ph-msk'] * wc_tns_cb
        elif self.scheme_da == 'multi':
            wc_tns_cb = self.__calc_wc_tns_mc(labl_dict['cb-idx'], labl_dict['cb-msk'], n_bins=37)
            wc_tns_om = self.__calc_wc_tns_mc(labl_dict['om-idx'], labl_dict['om-msk'], n_bins=25)
            wc_tns_th = self.__calc_wc_tns_mc(labl_dict['th-idx'], labl_dict['th-msk'], n_bins=25)
            wc_tns_ph = self.__calc_wc_tns_mc(labl_dict['ph-idx'], labl_dict['ph-msk'], n_bins=25)
        else:
            raise ValueError('unrecognized scheme for inter-residue DA predictions: ' + self.scheme_da)

        # loss function - inter-residue distance predictions
        loss_tns_cb = nn.CrossEntropyLoss(reduction='none')(pred_dict['cb'], labl_dict['cb-idx'])
        loss_cb = torch.sum(wc_tns_cb * loss_tns_cb) / (torch.sum(wc_tns_cb) + self.eps)

        # loss function - inter-residue orientation predictions
        loss_tns_om = nn.CrossEntropyLoss(reduction='none')(pred_dict['om'], labl_dict['om-idx'])
        loss_tns_th = nn.CrossEntropyLoss(reduction='none')(pred_dict['th'], labl_dict['th-idx'])
        loss_tns_ph = nn.CrossEntropyLoss(reduction='none')(pred_dict['ph'], labl_dict['ph-idx'])
        loss_om = torch.sum(wc_tns_om * loss_tns_om) / (torch.sum(wc_tns_om) + self.eps)
        loss_th = torch.sum(wc_tns_th * loss_tns_th) / (torch.sum(wc_tns_th) + self.eps)
        loss_ph = torch.sum(wc_tns_ph * loss_tns_ph) / (torch.sum(wc_tns_ph) + self.eps)

        # aggregrate all the loss functions & evaluation metrics
        loss = loss_cb + (loss_om + loss_th + loss_ph) / 3.0
        metrics = {
            'Loss-CB': loss_cb.item(),
            'Loss-OM': loss_om.item(),
            'Loss-TH': loss_th.item(),
            'Loss-PH': loss_ph.item(),
        }

        return loss, metrics

    def __calc_loss_lm(self, labl_tns, mask_tns, msa_pred):
        """Calculate the loss function for masked MSA predictions."""

        # initialization
        n_smpls, msa_depth, n_resds = labl_tns.shape

        if msa_pred.shape[2] == msa_depth:
            pred_tns = msa_pred.permute(0, 3, 1, 2)  # N x C x K x L
        else:
            pred_tns = msa_pred.view(n_smpls, msa_depth, n_resds, -1).permute(0, 3, 1, 2)

        # loss function
        loss_tns = nn.CrossEntropyLoss(reduction='none')(pred_tns, labl_tns)
        loss = torch.sum(mask_tns * loss_tns) / (torch.sum(mask_tns) + self.eps)

        # evaluation metrics
        metrics = {'Loss-LM': loss.item()}

        return loss, metrics


    def __calc_wc_tns_bc(self, labl_tns, mask_tns, n_bins_pos):
        """Calculate weighting coefficients for all the residue pairs - binary-class."""

        # determine per-class weighting coefficients
        mask_tns_pos = mask_tns * (labl_tns > 0).float() * (labl_tns <= n_bins_pos).float()
        mask_tns_neg = mask_tns - mask_tns_pos
        wc_pos = torch.sum(mask_tns) / (2.0 * torch.sum(mask_tns_pos) + self.eps)
        wc_neg = torch.sum(mask_tns) / (2.0 * torch.sum(mask_tns_neg) + self.eps)

        # determine weighting coefficients for all the residue pairs
        wc_tns = wc_pos * mask_tns_pos + wc_neg * mask_tns_neg

        return wc_tns


    def __calc_wc_tns_mc(self, labl_tns, mask_tns, n_bins):
        """Calculate weighting coefficients for all the residue pairs - multi-class."""

        # determine per-class weighting coefficients
        onht_tns = mask_tns.unsqueeze(dim=-1) * nn.functional.one_hot(labl_tns, n_bins)
        cnt_vec = torch.sum(onht_tns.view(-1, n_bins), dim=0)
        n_pairs_nnz = torch.count_nonzero(mask_tns)  # number of non-zero residue pairs
        n_bins_nnz = torch.count_nonzero(cnt_vec)  # number of non-zero bins
        wei_vec = n_pairs_nnz / (n_bins_nnz * torch.clip(cnt_vec, min=0.1) + self.eps)

        # determine weighting coefficients for all the residue pairs
        wc_tns = torch.sum(wei_vec.view(1, 1, 1, -1) * onht_tns, dim=3)

        return wc_tns
