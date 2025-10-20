"""Build per-chain FASTA and native PDB files."""

import os
import logging
import itertools
from collections import defaultdict
from multiprocessing import Manager, Pool

from rosetta_fold.utils import tfold_init
from rosetta_fold.utils import get_num_threads
from rosetta_fold.utils import export_fas_file
from rosetta_fold.utils import export_pdb_file
from rosetta_fold.tools import PdbParser
from rosetta_fold.datasets.rcsb_pdb.utils import load_chn_ids_grp
from rosetta_fold.datasets.rcsb_pdb.utils import save_chn_ids_raw
from rosetta_fold.datasets.rcsb_pdb.utils import save_chn_ids_grp


def build_fas_n_pdb(pgz_fpath, chn_ids, fas_dpath, pdb_dpath, error_dict):
    """Build per-chain FASTA and native PDB files."""

    # configurations
    seq_len_max = 1000

    # bild per-chain FASTA and native PDB files
    parser = PdbParser()
    structure = None
    for chn_id in chn_ids:
        # setup file paths
        fas_fpath = os.path.join(fas_dpath, '%s.fasta' % chn_id)
        pdb_fpath = os.path.join(pdb_dpath, '%s.pdb' % chn_id)
        if os.path.exists(fas_fpath) and os.path.exists(pdb_fpath):
            continue
        if chn_id in error_dict:
            continue

        # parse the source GZIP-compressed PDB file
        aa_seq, atom_cords, atom_masks, structure, error_msg = \
            parser.run(pgz_fpath, structure=structure, chain_id=chn_id[-1])
        if error_msg is not None:
            error_dict[chn_id] = error_msg
            logging.warning('failed to parse the PDB file for %s', chn_id)
            continue
        if len(aa_seq) > seq_len_max:
            error_dict[chn_id] = 'SEQUENCE_TOO_LONG'
            logging.warning('maximal sequence length exceeded (%s - %d AAs)', chn_id, len(aa_seq))
            continue

        # build per-chain FASTA and native PDB files
        export_fas_file(chn_id, aa_seq, fas_fpath)
        export_pdb_file(aa_seq, atom_cords, pdb_fpath, atom_masks)
        logging.info('FASTA & native PDB files generated for %s', chn_id)


def main():
    """Main entry."""

    # configurations
    n_threads = get_num_threads()
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, '../../../../data')
    pgz_dpath = os.path.join(data_dir, 'RCSB-PDB-20200101')
    cid_fpath_src = os.path.join(data_dir, 'bc-30-pruned.txt')
    dbs_dpath = os.path.join(data_dir, 'RCSB-PDB-328k')
    fas_dpath = os.path.join(dbs_dpath, 'fasta.files')
    pdb_dpath = os.path.join(dbs_dpath, 'pdb.files.native')
    cid_fpath_raw = os.path.join(dbs_dpath, 'chain_ids.txt')
    cid_fpath_grp = os.path.join(dbs_dpath, 'chain_ids_bc30.txt')
    err_fpath = os.path.join(dbs_dpath, 'pdb_errors.txt')

    # initialization
    tfold_init()

    # load PDB chain IDs in the grouped format, and then re-group them by PDB codes
    chn_ids_src = load_chn_ids_grp(cid_fpath_src)
    chn_ids_dict = defaultdict(list)
    for chn_id in itertools.chain(*chn_ids_src):
        pdb_code = chn_id[:4].lower()
        chn_ids_dict[pdb_code].append(chn_id)

    # initialize the error dict
    error_dict = Manager().dict()
    if os.path.exists(err_fpath):
        with open(err_fpath, 'r') as i_file:
            for i_line in i_file:
                chn_id, error_msg = i_line.strip().split()
                error_dict[chn_id] = error_msg

    # build per-chain FASTA and native PDB files
    args_list = []
    os.makedirs(fas_dpath, exist_ok=True)
    os.makedirs(pdb_dpath, exist_ok=True)
    for pdb_code, chn_ids in chn_ids_dict.items():
        pgz_fpath = os.path.join(pgz_dpath, pdb_code[1:3], 'pdb%s.ent.gz' % pdb_code)
        args_list.append((pgz_fpath, chn_ids, fas_dpath, pdb_dpath, error_dict))
    with Pool(processes=n_threads) as pool:
        pool.starmap(build_fas_n_pdb, args_list)

    # export PDB chain IDs
    chn_ids_raw = {x.replace('.fasta', '') for x in os.listdir(fas_dpath)}
    chn_ids_grp = []
    for chn_ids_sel in chn_ids_src:
        chn_ids_prn = [x for x in chn_ids_sel if x in chn_ids_raw]
        if len(chn_ids_prn) > 0:
            chn_ids_grp.append(chn_ids_prn)
    save_chn_ids_raw(chn_ids_raw, cid_fpath_raw)
    save_chn_ids_grp(chn_ids_grp, cid_fpath_grp)

    # record errors encountered in parsing PDB files
    with open(err_fpath, 'w') as o_file:
        o_file.write('\n'.join(['%s %s' % (k, v) for k, v in error_dict.items()]) + '\n')


if __name__ == '__main__':
    main()
