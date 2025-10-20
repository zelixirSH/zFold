"""Build HDF5 files for train/valid/test subsets."""

import os
import random
import logging
import itertools
from multiprocessing import Pool

import h5py

from xfold.dataset.tools import tfold_init
from xfold.dataset.tools import get_num_threads
from xfold.dataset.tools import recreate_directory

from xfold.dataset.tools import PdbParser
from xfold.dataset.scripts.rcsb_pdb.utils import load_chn_ids_raw
from xfold.dataset.scripts.rcsb_pdb.utils import load_chn_ids_grp


def build_hdf5_file(fas_dpath, pdb_dpath, chn_ids, hdf_fpath):
    """Build a HDF5 file with specified PDB chain IDs."""

    # show the greeting message
    logging.info('building the HDF5 file: %s', hdf_fpath)

    # obtain the FASTA sequence & atom coordinates (w/ masks) for each chain ID
    prot_data = {}
    parser = PdbParser()
    for chn_id in chn_ids:
        fas_fpath = os.path.join(fas_dpath, '%s.fasta' % chn_id)
        pdb_fpath = os.path.join(pdb_dpath, '%s.pdb' % chn_id)
        aa_seq, atom_cords, atom_masks, _, error = parser.run(pdb_fpath, fas_fpath=fas_fpath)
        assert error is None, 'failed to parse the PDB file: %s (error: %s)' % (pdb_fpath, error)
        prot_data[chn_id] = (aa_seq, atom_cords, atom_masks)

    # export the PDB data to a HDF5 file
    os.makedirs(os.path.dirname(os.path.realpath(hdf_fpath)), exist_ok=True)
    with h5py.File(hdf_fpath, 'w') as o_file:
        for chn_id, (aa_seq, atom_cords, atom_masks) in prot_data.items():
            group = o_file.create_group(chn_id)
            group.create_dataset('aa_seq', data=aa_seq)
            group.create_dataset('atom_cords', data=atom_cords)
            group.create_dataset('atom_masks', data=atom_masks)


def build_hdf5_files_raw(dbs_dpath, n_chns_per_file, n_threads):
    """Build HDF5 files for the data split based on PDB chain IDs."""

    # initialization
    fas_dpath = os.path.join(dbs_dpath, 'fasta.files')
    pdb_dpath = os.path.join(dbs_dpath, 'pdb.files.native')
    hdf_dpath = os.path.join(dbs_dpath, 'hdf5.files')
    cid_fpath_dict = {
        'trn': os.path.join(dbs_dpath, 'chain_ids_trn.txt'),
        'val': os.path.join(dbs_dpath, 'chain_ids_val.txt'),
        'tst': os.path.join(dbs_dpath, 'chain_ids_tst.txt'),
    }

    # prepare input arguments for building HDF5 files
    args_list = []
    for subset, cid_fpath in cid_fpath_dict.items():
        chn_ids = load_chn_ids_raw(cid_fpath)
        random.shuffle(chn_ids)
        n_chns = len(chn_ids)
        n_files = (n_chns + n_chns_per_file - 1) // n_chns_per_file
        for idx_file in range(n_files):
            idx_chn_beg = n_chns_per_file * idx_file
            idx_chn_end = min(n_chns, idx_chn_beg + n_chns_per_file)
            chn_ids_sel = chn_ids[idx_chn_beg:idx_chn_end]
            hdf_fpath = os.path.join(
                hdf_dpath, '%s-%04d-of-%04d.hdf5' % (subset, idx_file, n_files))
            args_list.append((fas_dpath, pdb_dpath, chn_ids_sel, hdf_fpath))

    # build HDF5 files with multi-threading enabled
    recreate_directory(hdf_dpath)
    with Pool(processes=n_threads) as pool:
        pool.starmap(build_hdf5_file, args_list)


def allocate_chn_ids(chn_ids_grp, n_bcks):
    """Allocate PDB chain IDs w/ balanced bucket size."""

    chn_ids_grp.sort(key=len, reverse=True)  # sort in the descending order of cluster sizes
    chn_ids_bck = [[] for _ in range(n_bcks)]
    for chn_ids in chn_ids_grp:
        idx_bck_min = 0
        for idx_bck in range(1, n_bcks):
            if len(chn_ids_bck[idx_bck_min]) > len(chn_ids_bck[idx_bck]):
                idx_bck_min = idx_bck
        chn_ids_bck[idx_bck_min].extend(chn_ids)
    for idx_bck in range(n_bcks):
        logging.info('bucket #%d: %d chains', idx_bck + 1, len(chn_ids_bck[idx_bck]))

    return chn_ids_bck


def build_hdf5_files_grp(dbs_dpath, n_chns_per_file, n_threads):
    """Build HDF5 files for the data split based on groups of PDB chain IDs."""

    # initialization
    fas_dpath = os.path.join(dbs_dpath, 'fasta.files')
    pdb_dpath = os.path.join(dbs_dpath, 'pdb.files.native')
    hdf_dpath = os.path.join(dbs_dpath, 'hdf5.files')
    cid_fpath_dict = {
        'trn': os.path.join(dbs_dpath, 'chain_ids_bc30_trn.txt'),
        'val': os.path.join(dbs_dpath, 'chain_ids_bc30_val.txt'),
        'tst': os.path.join(dbs_dpath, 'chain_ids_bc30_tst.txt'),
    }

    # prepare input arguments for building HDF5 files
    args_list = []
    for subset, cid_fpath in cid_fpath_dict.items():
        chn_ids_grp = load_chn_ids_grp(cid_fpath)
        n_chns = len(list(itertools.chain(*chn_ids_grp)))
        n_files = (n_chns + n_chns_per_file - 1) // n_chns_per_file
        chn_ids_bck = allocate_chn_ids(chn_ids_grp, n_files)
        for idx_file, chn_ids_sel in enumerate(chn_ids_bck):
            hdf_fpath = os.path.join(
                hdf_dpath, '%s-%04d-of-%04d.hdf5' % (subset, idx_file, n_files))
            args_list.append((fas_dpath, pdb_dpath, chn_ids_sel, hdf_fpath))

    # build HDF5 files with multi-threading enabled
    recreate_directory(hdf_dpath)
    with Pool(processes=n_threads) as pool:
        pool.starmap(build_hdf5_file, args_list)


def main():
    """Main entry."""

    # configurations
    n_chns_per_file = 4096
    n_threads = get_num_threads()
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, '../../../../data')
    suffix_list = ['1k', '27k', '328k']

    # initialization
    tfold_init()

    # split the dataset into train/valid/test subsets
    for suffix in suffix_list:
        dbs_dpath = os.path.join(data_dir, 'RCSB-PDB-' + suffix)
        if suffix in ['1k', '27k']:  # data split based on PDB chain IDs
            build_hdf5_files_raw(dbs_dpath, n_chns_per_file, n_threads)
        elif suffix == '328k':  # data split based on groups of PDB chain IDs
            build_hdf5_files_grp(dbs_dpath, n_chns_per_file, n_threads)


if __name__ == '__main__':
    main()
