"""Loss functions."""

import numpy as np
import torch
from torch import nn


def calc_wc_tns(labl_tns, mask_tns, n_bins_pos=12):
    """Calculate weighting coefficients for contact / non-contact residue pairs."""

    # initialization
    eps = 1e-6

    # determine weighting coefficients for contact / non-contact residue pairs
    mask_tns_pos = mask_tns * (labl_tns > 0).float() * (labl_tns <= n_bins_pos).float()
    mask_tns_neg = mask_tns - mask_tns_pos
    wc_pos = torch.sum(mask_tns) / (2.0 * torch.sum(mask_tns_pos) + eps)
    wc_neg = torch.sum(mask_tns) / (2.0 * torch.sum(mask_tns_neg) + eps)
    wc_tns = wc_pos * mask_tns_pos + wc_neg * mask_tns_neg

    return wc_tns


def calc_loss_da(labl_dict, pred_dict, addi_msk = None, only_cb = False, n_bins_pos = 12):
    """Calculate the loss function for inter-residue distance & orientation predictions."""

    # initialization
    eps = 1e-6
    # n_bins_pos = 12  # from bin #1 (2.0-2.5A) to bin #12 (7.5-8.0A) - zero-based indexing

    if addi_msk is not None:
        labl_dict['cb-msk'] = labl_dict['cb-msk'] * addi_msk
        labl_dict['om-msk'] = labl_dict['cb-msk'] * addi_msk
        labl_dict['th-msk'] = labl_dict['cb-msk'] * addi_msk
        labl_dict['ph-msk'] = labl_dict['cb-msk'] * addi_msk

    # calculate weighting coefficients for contact / non-contact residue pairs
    wc_tns_cb = calc_wc_tns(labl_dict['cb-idx'], labl_dict['cb-msk'], n_bins_pos=n_bins_pos)
    wc_tns_om = wc_tns_cb * labl_dict['om-msk']
    wc_tns_th = wc_tns_cb * labl_dict['th-msk']
    wc_tns_ph = wc_tns_cb * labl_dict['ph-msk']

    # loss function - inter-residue distance predictions
    loss_tns_cb = nn.CrossEntropyLoss(reduction='none')(pred_dict['cb'], labl_dict['cb-idx'])
    loss_cb = torch.sum(wc_tns_cb * loss_tns_cb) / (torch.sum(wc_tns_cb) + eps)

    # loss function - inter-residue orientation predictions
    loss_tns_om = nn.CrossEntropyLoss(reduction='none')(pred_dict['om'], labl_dict['om-idx'])
    loss_tns_th = nn.CrossEntropyLoss(reduction='none')(pred_dict['th'], labl_dict['th-idx'])
    loss_tns_ph = nn.CrossEntropyLoss(reduction='none')(pred_dict['ph'], labl_dict['ph-idx'])
    loss_om = torch.sum(wc_tns_om * loss_tns_om) / (torch.sum(wc_tns_om) + eps)
    loss_th = torch.sum(wc_tns_th * loss_tns_th) / (torch.sum(wc_tns_th) + eps)
    loss_ph = torch.sum(wc_tns_ph * loss_tns_ph) / (torch.sum(wc_tns_ph) + eps)

    # aggregrate all the loss functions & evaluation metrics
    if only_cb:
        loss = loss_cb
    else:
        loss = loss_cb + (loss_om + loss_th + loss_ph) / 3.0

    return loss, {}


def calc_loss_lm(labl_dict, pred_dict, only_mask_loss = False):
    """Calculate the loss function for masked MSA predictions."""

    # initialization
    eps = 1e-6
    labl_tns = labl_dict['msa-t']  # N x K x L
    mask_tns = labl_dict['msa-m']  # N x K x L

    n_smpls, msa_depth, n_resds = labl_tns.shape

    if pred_dict['lm'].shape[2] == msa_depth:
        pred_tns = pred_dict['lm'].permute(0, 3, 1, 2)  # N x C x K x L
    else:
        pred_tns = pred_dict['lm'].view(n_smpls, msa_depth, n_resds, -1).permute(0, 3, 1, 2)

    # loss function
    loss_tns = nn.CrossEntropyLoss(reduction='none')(pred_tns, labl_tns)

    if only_mask_loss:
        loss = loss_tns.mean()
    else:
        loss = torch.sum(mask_tns * loss_tns) / (torch.sum(mask_tns) + eps)

    # evaluation metrics
    metrics = {'Loss-LM': loss.item()}

    return loss, metrics


def calc_loss_dm(labl_dict, pred_dict):
    """Calculate the loss function for distance matrices derived from 3D coordinate predictions."""

    # initialization
    eps = 1e-6
    dist_thres = 20.0
    n_blks = len(pred_dict['cord'])
    n_smpls, n_resds, n_atoms, _ = pred_dict['cord'][0].shape

    # compute ground-truth Euclidean distance matrices & validness masks
    cord_tns_true = labl_dict['base']['cord'].view(n_smpls, n_resds * n_atoms, 3)
    mask_tns_cord = labl_dict['base']['mask'].view(n_smpls, n_resds * n_atoms)
    dist_tns_true = torch.cdist(cord_tns_true, cord_tns_true)  # N x (L x M) x (L x M)
    mask_tns_dist = (dist_tns_true <= dist_thres).float() * \
        mask_tns_cord.unsqueeze(dim=1) * mask_tns_cord.unsqueeze(dim=2)

    # determine weighting coefficients for each block
    loss_wc_med = 1.0
    loss_wc_fnl = 2.0
    loss_wc = np.array([loss_wc_med if x != n_blks - 1 else loss_wc_fnl for x in range(n_blks)])
    loss_wc /= np.sum(loss_wc)

    # evaluate intermediate & final 3D coordinate predictions
    loss_list = []
    metrics = {}
    for idx_blk in range(n_blks):
        cord_tns_pred = pred_dict['cord'][idx_blk].view(n_smpls, n_resds * n_atoms, 3)
        dist_tns_pred = torch.cdist(cord_tns_pred, cord_tns_pred)  # N x (L x M) x (L x M)
        diff_tns = torch.abs(dist_tns_pred - dist_tns_true)  # N x (L x M) x (L x M)
        loss = torch.sum(mask_tns_dist * diff_tns) / (torch.sum(mask_tns_dist) + eps)
        loss_list.append(loss_wc[idx_blk] * loss)
        metrics['Loss-DM-%d' % (idx_blk + 1)] = loss.item()
    loss = torch.sum(torch.stack(loss_list))

    return loss, metrics

def calc_loss_hr(inputs, output):
    loss_hr_list = [nn.CrossEntropyLoss(reduction='none')(highres.permute(0, 2, 1), inputs['base']['highres'])
                    for highres in output['highres']]
    loss_hr = torch.stack(loss_hr_list).mean() if len(loss_hr_list) > 0 else 0.0
    return loss_hr

def calc_loss_dsm(labl_dict, pred_dict):
    """Calculate the loss function for denoising score matching predictions."""

    # initialization
    eps = 1e-6
    mask_tns = labl_dict['base']['mask']  # 1 x L x M
    grad_tns = labl_dict['se3f']['grad']  # N x L x M x 3
    nstd_vec = labl_dict['se3f']['nstd']  # N
    pred_tns = pred_dict['pgrd']  # N x L x M x 3
    n_smpls, _, n_atoms, _ = grad_tns.shape

    # adjust weighting coefficients for different atom types
    wc_tns = mask_tns.repeat(n_smpls, 1, 1)
    if n_atoms == 3:
        wc_tns *= torch.tensor(
            [0.1, 1.0, 0.1], dtype=torch.float32, device=wc_tns.device).view(1, 1, -1)

    # loss function - all atoms
    diff_tns = nstd_vec.view(-1, 1, 1, 1) * torch.abs(pred_tns - grad_tns)
    dnrm_tns = wc_tns * torch.square(torch.norm(diff_tns, dim=-1))
    loss = torch.sum(dnrm_tns) / (torch.sum(wc_tns) + eps)
    metrics = {'Loss-DSM': loss.item()}

    # loss function - per atom type
    if n_atoms == 3:
        loss_vec = torch.sum(dnrm_tns, dim=(0, 1)) / (torch.sum(wc_tns, dim=(0, 1)) + eps)
        loss_vec_np = loss_vec.cpu().detach().numpy()
        metrics['Loss-DSM-N'] = loss_vec_np[0]
        metrics['Loss-DSM-CA'] = loss_vec_np[1]
        metrics['Loss-DSM-C'] = loss_vec_np[2]

    return loss, metrics


def calc_loss_fape(helper, params_list, plddt_list = None):
    """Calculate the loss function."""

    # initialization
    n_lyrs = len(params_list)
    n_resds = params_list[0][0].shape[1]

    # calculate the loss function for each set of QTA parameters
    loss_fa = None
    loss_a_ca_list = []
    loss_fa_ca_list = []
    metrics_finl = {}
    for idx_lyr, (quat_tns, trsl_tns, angl_tns) in enumerate(params_list):
        # print('3d', idx_lyr)
        params = {
            'quat': quat_tns.view(n_resds, -1).float(),
            'trsl': trsl_tns.view(n_resds, 3).float(),
            'angl': angl_tns.view(n_resds, -1, 2).float(),
        }
        loss_a_ca, loss_fa_ca, metrics = helper.calc_loss(params, atom_set='ca')
        if idx_lyr == n_lyrs - 1:
            loss_a, loss_fa, metrics = helper.calc_loss(params, atom_set='fa')

        loss_a_ca_list.append(loss_a_ca)
        loss_fa_ca_list.append(loss_fa_ca)

        # metrics_finl['dRMSD-L%d' % (idx_lyr + 1)] = metrics['dRMSD']

    # calculate the overall loss function
    loss_fa_final = loss_fa + torch.mean(torch.stack(loss_fa_ca_list))
    loss_a_final = loss_a + torch.mean(torch.stack(loss_a_ca_list))

    metrics_finl['Loss-FAPE'] = loss_fa_final.item()
    metrics_finl['Loss-Angle'] = loss_a_final.item()

    return loss_a_final, loss_fa_final, metrics_finl
