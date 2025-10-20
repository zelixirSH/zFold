"""Optimize per-residue local frames & torsion angles to minimize the FAPE loss."""

import os
import logging

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from zfold.dataset.utils import zfold_init
from zfold.loss.fape.constants import N_ANGLS_PER_RESD_MAX
from zfold.loss.fape.fape_loss_helper import FapeLossHelper
from zfold.network.af2_smod.prot_struct import ProtStruct

def init_params(aa_seq):
    """Initialize QTA (Q: quaternion / T: translation / A: angle) parameters."""

    # initialization
    n_resds = len(aa_seq)

    # initialize trainable parameters
    quat_mat = torch.zeros((n_resds, 3), dtype=torch.float32, requires_grad=True)
    trsl_mat = torch.zeros((n_resds, 3), dtype=torch.float32, requires_grad=True)
    angl_tns = torch.randn(
        (n_resds, N_ANGLS_PER_RESD_MAX, 2), dtype=torch.float32, requires_grad=True)

    # pack into a dict
    params = {'quat': quat_mat, 'trsl': trsl_mat, 'angl': angl_tns}

    return params


def main():
    """Main entry."""

    # configurations
    n_iters = 500
    n_intps = 64
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    pdb_code = '5JIU'
    chain_id = 'D'
    atom_set = 'fa'
    fas_fpath = os.path.join(curr_dir, 'examples/%s_%s.fasta' % (pdb_code, chain_id))
    pdb_fpath = os.path.join(curr_dir, 'examples/%s.pdb' % pdb_code)
    out_dpath = os.path.join(curr_dir, 'outputs.%s_%s' % (pdb_code, chain_id))
    pdb_fpath_natv = os.path.join(out_dpath, 'native.pdb')
    pdb_fpath_init = os.path.join(out_dpath, 'pred_init.pdb')
    pdb_fpath_finl = os.path.join(out_dpath, 'pred_finl.pdb')

    # initialization
    zfold_init(verb_levl='DEBUG')
    helper = FapeLossHelper()
    os.makedirs(out_dpath, exist_ok=True)

    # parse the PDB file
    aa_seq, cord_tns, cmsk_mat, error_msg = \
        ProtStruct.load(pdb_fpath, fas_fpath=fas_fpath, chain_id=chain_id)
    assert error_msg is None, 'failed to parse the PDB file: ' + pdb_fpath

    # let <FapeLossHelper> pre-process the native structure
    helper.preprocess(aa_seq, cord_tns, cmsk_mat)
    ProtStruct.save(aa_seq, helper.cord_tns, helper.cmsk_mat, pdb_fpath_natv)

    # initialize QTA parameters and then export the initial predicted structure
    params_pred = init_params(aa_seq)
    loss_angl, loss_fape, _, cord_tns_pred = helper.calc_loss(params_pred, atom_set, rtn_cord=True)
    loss = loss_angl + loss_fape
    logging.info('initial FAPE loss: %.4f', loss.item())
    ProtStruct.save(aa_seq, cord_tns_pred, helper.cmsk_mat, pdb_fpath_init)

    # optimize trainable parameters to minimize the FAPE loss
    optimizer = Adam(list(params_pred.values()), lr=1e-1)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, cooldown=8, min_lr=1e-4, verbose=True)
    for idx_iter in range(n_iters):
        loss_angl, loss_fape, metrics = helper.calc_loss(params_pred, atom_set)
        loss = loss_angl + loss_fape
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss.item())
        logging.info('FAPE loss @ %d-th iteration: %.4f', idx_iter + 1, loss.item())

    # export the final predicted structure
    loss_angl, loss_fape, _, cord_tns_pred = helper.calc_loss(params_pred, atom_set, rtn_cord=True)
    loss = loss_angl + loss_fape

    logging.info('final FAPE loss: %.4f', loss.item())
    ProtStruct.save(aa_seq, cord_tns_pred, helper.cmsk_mat, pdb_fpath_finl)

    '''
    # compare per-residue QTA parameters w/ linear interpolation
    for idx_resd in range(len(aa_seq)):
        # construct reference QTA parameters w/ only one residue's QTA parameters replaced
        params_refn = {k: v.detach().clone() for k, v in params_pred.items()}
        for key in params_refn:
            params_refn[key][idx_resd] = helper.params[key][idx_resd]

        # perform linear interpolation, and then evaluate
        for idx_intp in range(n_intps):
            w_beg = 1.0 - idx_intp / (n_intps - 1)
            w_end = 1.0 - w_beg
            params_intp = {k: w_beg * params_pred[k] + w_end * params_refn[k] for k in params_pred}
            loss, _ = helper.calc_loss(params_intp, atom_set)
            logging.info('interpolated parameters #%d: %.4e', idx_intp + 1, loss.item())
    '''


if __name__ == '__main__':
    main()
