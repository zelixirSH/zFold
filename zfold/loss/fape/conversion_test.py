"""Unit-tests for structure conversion routines in AlphaFold2."""

import os
import logging
from timeit import default_timer as timer

import numpy as np
import torch

from zfold.dataset.utils import zfold_init
from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.loss.fape.constants import RESD_MAP_1TO3
from zfold.loss.fape.constants import ATOM_NAMES_PER_RESD
from zfold.loss.fape.conversion import cord2fa
from zfold.loss.fape.conversion import fa2cord


def visualize_masks(aa_seq, mask_tns):
    """Visualize 1D / 2D masks."""

    assert mask_tns.ndim in [1, 2]
    for idx in range(len(aa_seq)):
        resd_name = RESD_MAP_1TO3[aa_seq[idx]]
        mask_str = ''.join(['*' if x == 1 else '.'  for x in np.nditer(mask_tns[idx])])
        logging.info('%s - %s', resd_name, mask_str)


def compare_cords(aa_seq, cord_tns_base, mask_mat_base, cord_tns_reco, mask_mat_reco):
    """Compare 3D coordinates between base & reconstructed structures."""

    # initialization
    eps = 1e-8
    n_resds, n_atoms = mask_mat_base.shape

    # compare 3D coordinates between base & reconstructed structures
    assert cord_tns_base.shape == cord_tns_reco.shape
    assert mask_mat_base.shape == mask_mat_reco.shape
    for idx_resd in range(n_resds):
        resd_name = RESD_MAP_1TO3[aa_seq[idx_resd]]
        for idx_atom in range(n_atoms):
            if mask_mat_base[idx_resd, idx_atom] == 0:
                continue
            atom_name = ATOM_NAMES_PER_RESD[resd_name][idx_atom]
            if mask_mat_reco[idx_resd, idx_atom] == 0:
                logging.warning('%s|%s - MISSING!', resd_name, atom_name)
                continue
            cord_vec_base = cord_tns_base[idx_resd, idx_atom]
            cord_vec_reco = cord_tns_reco[idx_resd, idx_atom]
            error = torch.norm(cord_vec_reco - cord_vec_base)
            logging.info('%s|%s - %.4f', resd_name, atom_name, error)

    # calculate the overall coordinate RMSD
    dist_mat = torch.norm(cord_tns_reco - cord_tns_base, dim=-1)
    mask_mat = mask_mat_base * mask_mat_reco
    rmsd = torch.sum(mask_mat * dist_mat) / (torch.sum(mask_mat) + eps)
    logging.info('coordinate RMSD: %.4f', rmsd.item())


def main():
    """Main entry."""

    # configurations
    pdb_code = '5JIU'
    chain_id = 'D'
    #device = torch.device('cpu')
    device = torch.device('cuda:0')
    n_repts = 16  # number of repeated runs for measuring the time consumption
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fas_fpath = os.path.join(curr_dir, 'examples/%s_%s.fasta' % (pdb_code, chain_id))
    pdb_fpath = os.path.join(curr_dir, 'examples/%s_%s.pdb' % (pdb_code, chain_id))

    # initialization
    zfold_init(verb_levl='DEBUG')

    # parse the PDB file
    aa_seq, cord_tns, cmsk_mat, error_msg = \
        ProtStruct.load(pdb_fpath, fas_fpath=fas_fpath)

    cord_tns = cord_tns.to(device)
    cmsk_mat = cmsk_mat.to(device)
    logging.info('=== cmsk_mat ===')
    visualize_masks(aa_seq, cmsk_mat.detach().cpu().numpy())

    # convert per-atom 3D coordinates to per-residue local frames & torsion angles
    fram_tns, fmsk_vec, angl_tns, amsk_mat = cord2fa(aa_seq, cord_tns, cmsk_mat)
    logging.info('fram_tns: %s / %s', fram_tns.shape, fram_tns.dtype)
    logging.info('fmsk_vec: %s / %s', fmsk_vec.shape, fmsk_vec.dtype)
    logging.info('angl_tns: %s / %s', angl_tns.shape, angl_tns.dtype)
    logging.info('amsk_mat: %s / %s', amsk_mat.shape, amsk_mat.dtype)
    logging.info('=== fmsk_vec ===')
    visualize_masks(aa_seq, fmsk_vec.detach().cpu().numpy())
    logging.info('=== amsk_mat ===')
    visualize_masks(aa_seq, amsk_mat.detach().cpu().numpy())

    # convert per-residue local frames & torsion angles to per-atom 3D coordinates
    cord_tns_reco, cmsk_mat_reco = fa2cord(aa_seq, fram_tns, fmsk_vec, angl_tns, amsk_mat)
    logging.info('cord_tns_reco: %s / %s', cord_tns_reco.shape, cord_tns_reco.dtype)
    logging.info('cmsk_mat_reco: %s / %s', cmsk_mat_reco.shape, cmsk_mat_reco.dtype)
    logging.info('=== cmsk_mat_reco ===')
    visualize_masks(aa_seq, cmsk_mat_reco.detach().cpu().numpy())

    # compare per-atom 3D coordinates between original & reconstructed structures
    compare_cords(aa_seq, cord_tns, cmsk_mat, cord_tns_reco, cmsk_mat_reco)

    # measure the time consumption of <cord2fa>
    time_vec = np.zeros((n_repts), dtype=np.float32)
    for idx_rept in range(n_repts):
        time_beg = timer()
        fram_tns, fmsk_vec, angl_tns, amsk_mat = cord2fa(aa_seq, cord_tns, cmsk_mat)
        time_vec[idx_rept] = 1000.0 * (timer() - time_beg)

    logging.info('cord2fa: %.2f +/- %.2f (ms)' % (np.mean(time_vec), np.std(time_vec)))

    # measure the time consumption of <fa2cord>
    time_vec = np.zeros((n_repts), dtype=np.float32)
    for idx_rept in range(n_repts):
        time_beg = timer()
        cord_tns_reco, cmsk_mat_reco = fa2cord(aa_seq, fram_tns, fmsk_vec, angl_tns, amsk_mat)
        time_vec[idx_rept] = 1000.0 * (timer() - time_beg)
    logging.info('fa2cord: %.2f +/- %.2f (ms)' % (np.mean(time_vec), np.std(time_vec)))


if __name__ == '__main__':
    main()
