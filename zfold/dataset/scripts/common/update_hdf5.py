"""Update HDF5 files to include per-residue local frames & torsion angles."""

import os
import logging
from multiprocessing import Manager, Pool

import h5py
import torch

from zfold.dataset.utils import zfold_init
from zfold.loss.fape.fape_loss_helper import FapeLossHelper

def build_fa_data(prot_id, aa_seq, cord_tns, cmsk_mat, prot_data):
    """Build per-residue local frames & torsion angles."""

    # initialize the <FapeLossHelper>
    helper = FapeLossHelper()
    helper.preprocess(aa_seq, cord_tns, cmsk_mat)

    # record per-residue local frames & torsion angles
    prot_data[prot_id] = {
        'seq': aa_seq,
        'cord-o': cord_tns.detach().cpu().numpy(),  # original
        'cmsk-o': cmsk_mat.detach().cpu().numpy(),
        'cord-r': helper.cord_tns.detach().cpu().numpy(),  # reconstructed
        'cmsk-r': helper.cmsk_mat.detach().cpu().numpy(),
        'fram': helper.fram_tns.detach().cpu().numpy(),
        'fmsk': helper.fmsk_vec.detach().cpu().numpy(),
        'angl': helper.angl_tns.detach().cpu().numpy(),
        'amsk': helper.amsk_mat.detach().cpu().numpy(),
    }


def update_hdf5(path_src, path_dst):
    """Update the HDF5 file to include per-residue local frames & torsion angles."""

    # parse the HDF5 file
    args_list = []
    prot_data = {}
    with h5py.File(path_src, 'r', driver='core') as i_file:
        for prot_id in i_file:
            aa_seq = i_file[prot_id]['seq'][()]#.decode('utf-8')
            cord_tns = torch.tensor(i_file[prot_id]['cord'][()], dtype=torch.float32)
            cmsk_mat = torch.tensor(i_file[prot_id]['mask'][()], dtype=torch.int8)
            args_list.append((prot_id, aa_seq, cord_tns, cmsk_mat, prot_data))

    # build per-residue local frames & torsion angles
    for args in args_list:
        build_fa_data(*args)

    # save per-residue local frames & torsion angles into the HDF5 file
    os.makedirs(os.path.dirname(os.path.realpath(path_dst)), exist_ok=True)
    with h5py.File(path_dst, 'w') as o_file:
        for prot_id in prot_data:
            group = o_file.create_group(prot_id)
            group.create_dataset('seq', data=prot_data[prot_id]['seq'])
            group.create_dataset('cord-o', data=prot_data[prot_id]['cord-o'])
            group.create_dataset('cmsk-o', data=prot_data[prot_id]['cmsk-o'])
            group.create_dataset('cord-r', data=prot_data[prot_id]['cord-r'])
            group.create_dataset('cmsk-r', data=prot_data[prot_id]['cmsk-r'])
            group.create_dataset('fram', data=prot_data[prot_id]['fram'])
            group.create_dataset('fmsk', data=prot_data[prot_id]['fmsk'])
            group.create_dataset('angl', data=prot_data[prot_id]['angl'])
            group.create_dataset('amsk', data=prot_data[prot_id]['amsk'])


def main():
    """Main entry."""

    # configurations
    n_threads = 16
    data_dir = '/data1/protein/zyf/casp14'
    hdf_dpath_src = f'{data_dir}/hdf5.files_new'
    hdf_dpath_dst = f'{data_dir}/hdf5.files_new.fa'

    # initialization
    tfold_init()

    # append per-residue local frames & torsion angles into HDF5 files
    args_list = []
    for hdf_fname in os.listdir(hdf_dpath_src):
        hdf_fpath_src = os.path.join(hdf_dpath_src, hdf_fname)
        hdf_fpath_dst = os.path.join(hdf_dpath_dst, hdf_fname)
        if not os.path.exists(hdf_fpath_dst):
            print(hdf_fpath_dst)
            args_list.append((hdf_fpath_src, hdf_fpath_dst))

    with Pool(processes=n_threads) as pool:
        pool.starmap(update_hdf5, args_list)


if __name__ == '__main__':
    main()
