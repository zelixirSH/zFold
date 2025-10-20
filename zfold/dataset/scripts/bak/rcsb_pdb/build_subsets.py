"""Build subsets of various sizes from the full RCSB-PDB database.

For RCSB-PDB-1k and RCSB-PDB-10k, PDB chain IDs with MSA data available are selected.
For RCSB-PDB-100k, there is no such restraint, since only ~35k PDB chain IDs have MSA data.
"""

import os
import random
import shutil
import logging
from multiprocessing import Manager, Pool

from rosetta_fold.utils import tfold_init
from rosetta_fold.utils import get_num_threads
from rosetta_fold.utils import recreate_directory
from rosetta_fold.utils import parse_fas_file
from rosetta_fold.datasets.rcsb_pdb.utils import load_chn_ids_raw
from rosetta_fold.datasets.rcsb_pdb.utils import load_chn_ids_grp
from rosetta_fold.datasets.rcsb_pdb.utils import save_chn_ids_raw


def check_msa_data(fas_fpath_dbs, msa_dpath, postfix, chn_id_dbs, result_dict):
    """Check whether the specified PDB chain IDs has pre-extracted MSA data available."""

    chn_id_msa = chn_id_dbs[:4].lower() + chn_id_dbs[-1]
    msa_dpath_sel = os.path.join(msa_dpath, '%s.fas' % chn_id_msa)
    fas_fpath_msa = os.path.join(msa_dpath_sel, '%s_%s.fasta' % (chn_id_msa, postfix))
    a3m_fpath_msa = os.path.join(msa_dpath_sel, '%s_%s.a3m' % (chn_id_msa, postfix))
    if os.path.exists(fas_fpath_msa) and os.path.exists(a3m_fpath_msa):
        _, aa_seq_dbs = parse_fas_file(fas_fpath_dbs)
        _, aa_seq_msa = parse_fas_file(fas_fpath_msa)
        if aa_seq_dbs == aa_seq_msa:
            result_dict[chn_id_dbs] = True
        else:
            result_dict[chn_id_dbs] = False
            n_chns_w_msa = sum([1 if x else 0 for x in result_dict.values()])
            n_chns_wo_msa = len(result_dict) - n_chns_w_msa
            logging.warning('inconsistent amino-acid sequences detected for %s', chn_id_dbs)
            logging.warning('Ref: %s', aa_seq_dbs)
            logging.warning('MSA: %s', aa_seq_msa)
            logging.info('# of chain IDs: %d (w/ MSA) / %d (w/o MSA)', n_chns_w_msa, n_chns_wo_msa)
    else:
        result_dict[chn_id_dbs] = False


def build_subset(dbs_dpath_src, dbs_dpath_dst, chn_ids, msa_dpath=None, postfix=None):
    """Build a subset of RCSB-PDB database with specified PDB chain IDs."""

    # setup directory paths
    fas_dpath_src = os.path.join(dbs_dpath_src, 'fasta.files')
    pdb_dpath_src = os.path.join(dbs_dpath_src, 'pdb.files.native')
    cid_fpath_dst = os.path.join(dbs_dpath_dst, 'chain_ids.txt')
    fas_dpath_dst = os.path.join(dbs_dpath_dst, 'fasta.files')
    pdb_dpath_dst = os.path.join(dbs_dpath_dst, 'pdb.files.native')

    # export PDB chain IDs
    save_chn_ids_raw(chn_ids, cid_fpath_dst)

    # copy FASTA & native PDB files
    recreate_directory(fas_dpath_dst)
    recreate_directory(pdb_dpath_dst)
    for chn_id in chn_ids:
        fas_fpath_src = os.path.join(fas_dpath_src, '%s.fasta' % chn_id)
        fas_fpath_dst = os.path.join(fas_dpath_dst, '%s.fasta' % chn_id)
        pdb_fpath_src = os.path.join(pdb_dpath_src, '%s.pdb' % chn_id)
        pdb_fpath_dst = os.path.join(pdb_dpath_dst, '%s.pdb' % chn_id)
        shutil.copyfile(fas_fpath_src, fas_fpath_dst)
        shutil.copyfile(pdb_fpath_src, pdb_fpath_dst)

    # copy pre-extracted MSA files
    if (msa_dpath is not None) and (postfix is not None):
        a3m_dpath_dst = os.path.join(dbs_dpath_dst, 'a3m.files.nr')
        recreate_directory(a3m_dpath_dst)
        for chn_id in chn_ids:
            chn_id_msa = chn_id[:4].lower() + chn_id[-1]
            msa_dpath_sel = os.path.join(msa_dpath, '%s.fas' % chn_id_msa)
            a3m_fpath_src = os.path.join(msa_dpath_sel, '%s_%s.a3m' % (chn_id_msa, postfix))
            a3m_fpath_dst = os.path.join(a3m_dpath_dst, '%s.a3m' % chn_id)
            shutil.copyfile(a3m_fpath_src, a3m_fpath_dst)


def main():
    """Main entry."""

    # configurations
    n_threads = get_num_threads()
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, '../../../../data')
    dbs_dpath_full = os.path.join(data_dir, 'RCSB-PDB-328k')
    dbs_dpath_nord = os.path.join(data_dir, 'RCSB-PDB-27k')  # one entry per cluster
    dbs_dpath_tiny = os.path.join(data_dir, 'RCSB-PDB-1k')  # 1k sequences in total
    cid_fpath_raw = os.path.join(dbs_dpath_full, 'chain_ids.txt')
    cid_fpath_grp = os.path.join(dbs_dpath_full, 'chain_ids_bc30.txt')
    n_chns_tiny = 1000
    msa_dpath = '/mnt/superCeph2/private/user/shentao/database/BC40_20200509/mrf_NR'
    postfix = 'NR_1e-3_5_-1'

    # initialization
    tfold_init()

    # get PDB chain IDs
    chn_ids_raw = load_chn_ids_raw(cid_fpath_raw)
    chn_ids_grp = load_chn_ids_grp(cid_fpath_grp)

    # find PDB chain IDs with pre-extracted MSA data available
    args_list = []
    result_dict = Manager().dict()
    for chn_id in chn_ids_raw:
        fas_fpath = os.path.join(dbs_dpath_full, 'fasta.files', '%s.fasta' % chn_id)
        args_list.append((fas_fpath, msa_dpath, postfix, chn_id, result_dict))
    with Pool(processes=n_threads) as pool:
        pool.starmap(check_msa_data, args_list)
    chn_ids_msa = [x for x in chn_ids_raw if result_dict[x]]

    # build the non-redundant subset (one entry per cluster)
    logging.info('building the non-redundant RCSB-PDB subset ...')
    chn_ids_nord = [random.sample(x, 1)[0] for x in chn_ids_grp]
    build_subset(dbs_dpath_full, dbs_dpath_nord, chn_ids_nord)

    # build the tiny subset (1k sequences in total)
    logging.info('building the tiny RCSB-PDB subset ...')
    chn_ids_tiny = random.sample(chn_ids_msa, n_chns_tiny)
    build_subset(dbs_dpath_full, dbs_dpath_tiny, chn_ids_tiny, msa_dpath, postfix)


if __name__ == '__main__':
    main()
