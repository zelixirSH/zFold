"""Prune clusters w/o any valid PDB structures.

Since sequence clustering results are obtained on the latest RCSB-PDB database, it is possible that
some clusters may be empty for a early snapshot of RCSB-PDB dataset. This script removes those
clusters to build a reduced set of sequence clustering results.
"""

import os
import logging

from rosetta_fold.utils import tfold_init
from rosetta_fold.datasets.rcsb_pdb.utils import load_chn_ids_grp
from rosetta_fold.datasets.rcsb_pdb.utils import save_chn_ids_grp


def get_pdb_codes(pdb_dpath_root):
    """Get all the PDB codes."""

    pdb_codes = set()
    for subdir_name in os.listdir(pdb_dpath_root):
        pdb_dpath = os.path.join(pdb_dpath_root, subdir_name)
        pdb_codes_new = {x[3:7].upper() for x in os.listdir(pdb_dpath)}
        pdb_codes.update(pdb_codes_new)

    return pdb_codes


def prune_chn_ids(chn_ids_src, pdb_codes):
    """Prune groups of PDB chain IDs, which has no valid PDB structure."""

    chn_ids_dst = []
    for chn_ids_raw in chn_ids_src:
        chn_ids_prn = [x for x in chn_ids_raw if x[:4] in pdb_codes]
        if len(chn_ids_prn) > 0:
            chn_ids_dst.append(chn_ids_prn)

    return chn_ids_dst


def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, '../../../../data')
    pdb_dpath = os.path.join(data_dir, 'RCSB-PDB-20200101')
    cid_fpath_src = os.path.join(data_dir, 'bc-30.out')
    cid_fpath_dst = os.path.join(data_dir, 'bc-30-pruned.txt')

    # initialization
    tfold_init()

    # get all the PDB codes
    pdb_codes = get_pdb_codes(pdb_dpath)
    logging.info('# of PDB codes: %d', len(pdb_codes))

    # get sequence clustering results, and then prune empty ones
    chn_ids_src = load_chn_ids_grp(cid_fpath_src)
    chn_ids_dst = prune_chn_ids(chn_ids_src, pdb_codes)
    save_chn_ids_grp(chn_ids_dst, cid_fpath_dst)
    n_chns_src = sum([len(x) for x in chn_ids_src])
    n_chns_dst = sum([len(x) for x in chn_ids_dst])
    logging.info('# of PDB chain IDs: %d (original) / %d (pruned)', n_chns_src, n_chns_dst)
    logging.info('# of groups of PDB chain IDs: %d (original) / %d (pruned)',
                 len(chn_ids_src), len(chn_ids_dst))


if __name__ == '__main__':
    main()
