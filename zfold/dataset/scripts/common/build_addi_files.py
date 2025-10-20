"""Build additional HDF5 & NPZ files."""

import os
import logging
from multiprocessing import Manager, Pool

import h5py
import torch
import numpy as np
from scipy.spatial.distance import cdist

from zfold.dataset.utils import zfold_init
from zfold.dataset.utils import get_md5sum
from zfold.dataset.utils import parse_fas_file
from zfold.dataset.utils import calc_plane_angle
from zfold.dataset.utils import calc_dihedral_angle
from zfold.network.af2_smod.prot_struct import ProtStruct

from tqdm import tqdm
import shutil
import random

def _move(a, b):
    if os.path.exists(a):
        try:
            shutil.move(a, b)
        except:
            return

def gunzip(file):
    if os.path.exists(file) and not os.path.exists(file.replace('.gz','')):
        os.system(f'gunzip {file}')

def get_prot_data(fas_fpath, pdb_fpath, prot_data):
    """Get the protein data (amino-acid sequence & full-atom 3D coordinates."""

    chn_id, _ = parse_fas_file(fas_fpath)

    chn_id = os.path.basename(fas_fpath).replace('.fasta','')

    aa_seq, cord_tns, mask_mat, error = ProtStruct.load(pdb_fpath, fas_fpath=fas_fpath)

    # assert error is None, 'failed to parse the PDB file: %s (error: %s)' % (pdb_fpath, error)

    if error is None:
        prot_data[chn_id] = (aa_seq, cord_tns, mask_mat)
    else:
        print('failed to parse the PDB file: %s (error: %s)' % (pdb_fpath, error))


def build_hdf_file(prot_data, chn_ids, hdf_fpath):
    """Build a HDF5 file w/ amino-acid sequences and full-atom 3D coordinates."""

    os.makedirs(os.path.dirname(os.path.realpath(hdf_fpath)), exist_ok=True)

    if os.path.exists(hdf_fpath):
        return

    with h5py.File(hdf_fpath, 'w') as o_file:
        for chn_id in chn_ids:
            if chn_id in prot_data:
                aa_seq, cord_tns, mask_mat = prot_data[chn_id]
                group = o_file.create_group(chn_id)
                group.create_dataset('seq', data=aa_seq)
                group.create_dataset('cord', data=cord_tns)
                group.create_dataset('mask', data=mask_mat)

    logging.info('HDF5 file built: %s', hdf_fpath)


def build_npz_file(aa_seq, cord_tns, mask_mat, npz_fpath):
    """Build a NPZ file w/ ground-truth labels for inter-residue distance & orientation."""

    if os.path.exists(npz_fpath):
        logging.info('NPZ file exists: %s', npz_fpath)
        return

    # initialization
    n_resds = len(aa_seq)
    atom_names = ['N', 'CA', 'CB']

    # obtain 3D coordinates for N/CA/CB atoms
    cord_tns_sel = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)
    mask_mat_sel = ProtStruct.get_atoms(aa_seq, mask_mat, atom_names)
    x_n, x_ca, x_cb = [x.squeeze_().numpy() for x in torch.split(cord_tns_sel, 1, dim=1)]
    m_n, m_ca, m_cb = [x.squeeze_().numpy() for x in torch.split(mask_mat_sel, 1, dim=1)]

    # use GLY's CA atom as the replacement for its missing CB atom
    is_gly = np.array([1 if aa_seq[x] == 'G' else 0 for x in range(n_resds)], dtype=np.int8)
    x_cab = is_gly[:, None] * x_ca + (1 - is_gly[:, None]) * x_cb
    m_cab = is_gly * m_ca + (1 - is_gly) * m_cb

    # build ground-truth labels
    labl_data = {}
    labl_data['cb-val'] = cdist(x_cab, x_cab).astype(np.float16)
    labl_data['cb-msk'] = np.outer(m_cab, m_cab).astype(np.int8)
    labl_data['om-val'] = np.zeros((n_resds, n_resds), dtype=np.float16)
    labl_data['om-msk'] = np.zeros((n_resds, n_resds), dtype=np.int8)
    labl_data['th-val'] = np.zeros((n_resds, n_resds), dtype=np.float16)
    labl_data['th-msk'] = np.zeros((n_resds, n_resds), dtype=np.int8)
    labl_data['ph-val'] = np.zeros((n_resds, n_resds), dtype=np.float16)
    labl_data['ph-msk'] = np.zeros((n_resds, n_resds), dtype=np.int8)
    for ir in range(n_resds):
        for ic in range(n_resds):
            labl_data['om-val'][ir, ic] = calc_dihedral_angle(x_ca[ir], x_cb[ir], x_cb[ic], x_ca[ic])
            labl_data['th-val'][ir, ic] = calc_dihedral_angle(x_n[ir], x_ca[ir], x_cb[ir], x_cb[ic])
            labl_data['ph-val'][ir, ic] = calc_plane_angle(x_ca[ir], x_cb[ir], x_cb[ic])
            labl_data['om-msk'][ir, ic] = m_ca[ir] * m_cb[ir] * m_cb[ic] * m_ca[ic]
            labl_data['th-msk'][ir, ic] = m_n[ir] * m_ca[ir] * m_cb[ir] * m_cb[ic]
            labl_data['ph-msk'][ir, ic] = m_ca[ir] * m_cb[ir] * m_cb[ic]

    # build a NPZ file
    os.makedirs(os.path.dirname(os.path.realpath(npz_fpath)), exist_ok=True)
    np.savez(npz_fpath, **labl_data)
    logging.info('NPZ file built: %s', npz_fpath)

def main(db):
    """Main entry."""
    # configurations
    n_threads = 64
    n_chns_per_file = 1024

    if db == 'msa_800k_part1':
        # MSA-DATA
        data_dir = '/mnt/superCeph2/private/user/shentao/MSA-Trans-DATA/part1'
        cid_fpath_all = '/mnt/superCeph2/private/user/shentao/tfold-self-distilation/version1/prot_ids.txt'

        hdf_dpath = '/mnt/superCeph2/private/user/shentao/tfold-self-distilation/version1/af2.model_1.hdf5.files'
        fas_dpath = os.path.join(data_dir, 'msa_800k_part1.fas')
        pdb_dpath = os.path.join(data_dir, 'files.af2.model_1.pdb')
        npz_dpath = os.path.join(data_dir, 'npz.files.da_labl')

        cid_fpath_split = '/mnt/superCeph2/private/user/shentao/tfold-self-distilation/version1/splits'
        os.makedirs(cid_fpath_split, exist_ok=True)

    elif db == 'trrosetta':
        # trrosetta
        data_dir = '/mnt/superCeph2/private/user/shentao/database/trRosetta_DATA/trrosetta/'
        cid_fpath_all = f'{data_dir}/list15051.txt'

        hdf_dpath = os.path.join(data_dir, 'hdf5.files')
        fas_dpath = os.path.join(data_dir, 'fas')
        pdb_dpath = os.path.join(data_dir, 'pdb')
        npz_dpath = os.path.join(data_dir, 'npz.files.da_labl')

        cid_fpath_split = f'{data_dir}/splits'
        os.makedirs(cid_fpath_split, exist_ok=True)

    elif db == 'cameo3m':
        # CAMEO3M
        data_dir = '/data1/protein/zyf/cameo3m-seq'
        fas_dpath = os.path.join(data_dir, '20210710-20211002.targets.fasta')
        pdb_dpath = os.path.join(data_dir, '20210710-20211002.targets.pdb')
        cid_fpath_all = os.path.join(data_dir, '20210710-20211002.targets.svr98.ids.txt')
        hdf_dpath = os.path.join(data_dir, 'hdf5.files')
        npz_dpath = os.path.join(data_dir, 'npz.files.da_labl')

        cid_fpath_split = f'{data_dir}/splits'
        os.makedirs(cid_fpath_split, exist_ok=True)

    elif db == 'casp14':
        # CASP14
        data_dir = '/data1/protein/zyf/casp14'
        fas_dpath = os.path.join(data_dir, 'fasta.files')
        pdb_dpath = os.path.join(data_dir, 'pdb.files.native')
        cid_fpath_all = os.path.join(data_dir, 'domain_ids.txt')

        hdf_dpath = os.path.join(data_dir, 'hdf5.files_new')
        npz_dpath = os.path.join(data_dir, 'npz.files.da_labl_new')
        cid_fpath_split = f'{data_dir}/split_new'
        os.makedirs(cid_fpath_split, exist_ok=True)

    elif db == 'RCSB':
        # RCSB-PDB-28k
        data_dir = '/apdcephfs/share_1436367/jonathanwu/Datasets/RCSB-PDB-28k'
        fas_dpath = os.path.join(data_dir, 'fasta.files')
        pdb_dpath = os.path.join(data_dir, 'pdb.files.native')
        cid_fpath_all = os.path.join(data_dir, 'chain_ids.txt')
        cid_fpath_dict = {
            'trn': os.path.join(data_dir, 'chain_ids_trn.txt'),
            'val': os.path.join(data_dir, 'chain_ids_val.txt'),
            'tst': os.path.join(data_dir, 'chain_ids_tst.txt'),
        }
        save_dir = '/mnt/superCeph2/private/user/shentao/database/tmp'
        hdf_dpath = os.path.join(save_dir, 'hdf5.files')
        npz_dpath = os.path.join(save_dir, 'npz.files.da_labl')

    else:
        raise NotImplementedError

    f = open(cid_fpath_all, 'r')
    lines = f.readlines()
    f.close()

    inter = 10000
    cid_fpath_dict = {}
    for split in range(len(lines)//inter +1):
        split_path = f'{cid_fpath_split}/chain_ids_split-{split}.txt'
        if not os.path.exists(split_path):
            f = open(split_path, 'w')
            f.writelines(lines[split*inter: (split+1)*inter])
            f.close()
            print(f'{cid_fpath_split}/chain_ids_split-{split}.txt')

        cid_fpath_dict[str(split)] = f'{cid_fpath_split}/chain_ids_split-{split}.txt'

    # initialization
    tfold_init()

    items = list(cid_fpath_dict.items())
    random.shuffle(items)

    for subset, cid_fpath in items:
        with open(cid_fpath, 'r') as i_file:
            chn_ids = [i_line.strip() for i_line in i_file]

        prot_data = Manager().dict()

        args_list = []
        for chn_id in tqdm(chn_ids):
            fas_fpath = os.path.join(fas_dpath, '%s.fasta' % chn_id)
            pdb_fpath = os.path.join(pdb_dpath, '%s.pdb' % chn_id)
            args_list.append((fas_fpath, pdb_fpath, prot_data))

        with Pool(processes=n_threads) as pool:
            pool.starmap(get_prot_data, args_list)

        # build HDF5 files
        args_list = []
        chn_ids.sort(key=get_md5sum)
        n_chns = len(chn_ids)
        n_files = (n_chns + n_chns_per_file - 1) // n_chns_per_file

        for idx_file in range(n_files):
            idx_chn_beg = n_chns_per_file * idx_file
            idx_chn_end = min(n_chns, idx_chn_beg + n_chns_per_file)
            chn_ids_sel = sorted(chn_ids[idx_chn_beg:idx_chn_end])
            hdf_fpath = os.path.join(hdf_dpath, '%s-%04d-of-%04d.hdf5' % (subset, idx_file, n_files))
            args_list.append((prot_data, chn_ids_sel, hdf_fpath))

        with Pool(processes=n_threads) as pool:
            pool.starmap(build_hdf_file, args_list)

        # build NPZ files
        os.makedirs(npz_dpath, exist_ok = True)
        args_list = []
        random.shuffle(chn_ids)
        for chn_id in chn_ids:
            if chn_id in prot_data:
                aa_seq, cord_tns, mask_mat = prot_data[chn_id]
                npz_fpath = os.path.join(npz_dpath, '%s.npz' % chn_id)
                args_list.append((aa_seq, cord_tns, mask_mat, npz_fpath))

        with Pool(processes=n_threads) as pool:
            pool.starmap(build_npz_file, args_list)

if __name__ == '__main__':
    # main(db = 'cameo3m')
    main(db = 'casp14')
