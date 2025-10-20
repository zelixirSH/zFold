"""Unit-tests for the <ProtConverter> class."""

import os
import logging
from timeit import default_timer as timer

import torch
import numpy as np
from zfold.dataset.utils import zfold_init
from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.network.af2_smod.prot_converter import ProtConverter

def backbone():
    """Main entry."""

    # configurations
    eps = 1e-6
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
    aa_seq, cord_tns_base, cmsk_mat_base, error_msg = \
        ProtStruct.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, 'failed to parse the PDB file: ' + pdb_fpath
    cord_tns_base = cord_tns_base.to(device)
    cmsk_mat_base = cmsk_mat_base.to(device)

    # test w/ <ProtConverter>
    converter = ProtConverter()
    fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat = \
        converter.cord2fa(aa_seq, cord_tns_base, cmsk_mat_base)

    cord_tns_reco, cmsk_mat_reco, _, _ = \
        converter.fa2cord(aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat,
                          atom_set='fa')

    # calculate the overall coordinate RMSD
    dist_mat = torch.norm(cord_tns_reco - cord_tns_base, dim=-1)
    cmsk_mat = cmsk_mat_base * cmsk_mat_reco
    rmsd = torch.sum(cmsk_mat * dist_mat) / (torch.sum(cmsk_mat) + eps)
    logging.info('coordinate RMSD: %.4f', rmsd.item())


def main():
    """Main entry."""

    # configurations
    eps = 1e-6
    # pdb_code = '5JIU'
    # chain_id = 'D'

    pdb_code = '3MLS'
    chain_id = 'P'

    device = torch.device('cpu')
    # device = torch.device('cuda:0')
    n_repts = 16  # number of repeated runs for measuring the time consumption
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fas_fpath = os.path.join(curr_dir, 'examples/%s_%s.fasta' % (pdb_code, chain_id))
    pdb_fpath = os.path.join(curr_dir, 'examples/%s_%s.pdb' % (pdb_code, chain_id))

    # initialization
    zfold_init(verb_levl='DEBUG')

    # parse the PDB file
    aa_seq, cord_tns_base, cmsk_mat_base, error_msg = \
        ProtStruct.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, 'failed to parse the PDB file: ' + pdb_fpath
    cord_tns_base = cord_tns_base.to(device)
    cmsk_mat_base = cmsk_mat_base.to(device)

    # test w/ <ProtConverter>
    converter = ProtConverter()
    fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat = \
        converter.cord2fa(aa_seq, cord_tns_base, cmsk_mat_base)

    cord_tns_reco, cmsk_mat_reco, fram_tns_sc, fmsk_mat_sc = \
        converter.fa2cord(aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat, atom_set='fa')

    # calculate the overall coordinate RMSD
    print(cord_tns_reco.shape, cord_tns_base.shape)
    dist_mat = torch.norm(cord_tns_reco - cord_tns_base, dim=-1)
    cmsk_mat = cmsk_mat_base * cmsk_mat_reco
    rmsd = torch.sum(cmsk_mat * dist_mat) / (torch.sum(cmsk_mat) + eps)
    logging.info('coordinate RMSD: %.4f', rmsd.item())

    # # measure the time consumption of <cord2fa>
    # time_vec = np.zeros((n_repts), dtype=np.float32)
    # for idx_rept in range(n_repts):
    #     time_beg = timer()
    #     fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat = \
    #         converter.cord2fa(aa_seq, cord_tns_base, cmsk_mat_base)
    #     time_vec[idx_rept] = 1000.0 * (timer() - time_beg)
    # logging.info('cord2fa: %.2f +/- %.2f (ms)' % (np.mean(time_vec), np.std(time_vec)))
    #
    # # measure the time consumption of <fa2cord>
    # time_vec = np.zeros((n_repts), dtype=np.float32)
    # for idx_rept in range(n_repts):
    #     time_beg = timer()
    #     cord_tns_reco, cmsk_mat_reco, fram_tns_sc, fmsk_mat_sc = \
    #         converter.fa2cord(aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat, atom_set='fa')
    #     time_vec[idx_rept] = 1000.0 * (timer() - time_beg)
    # logging.info('fa2cord: %.2f +/- %.2f (ms)' % (np.mean(time_vec), np.std(time_vec)))


if __name__ == '__main__':
    main()
    # backbone()
