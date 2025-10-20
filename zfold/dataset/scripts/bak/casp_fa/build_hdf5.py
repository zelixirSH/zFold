"""Build a HDF5 file for CASP13 / CASP14 targets."""

import os
import logging

import h5py

from zfold.dataset.utils import tfold_init
from zfold.dataset.utils import parse_fas_file
from zfold.network.af2_smod.prot_struct import ProtStruct

def get_prot_data(fas_fpath, pdb_fpath, prot_data):
    """Get the protein data (amino-acid sequence & full-atom 3D coordinates."""

    chn_id, _ = parse_fas_file(fas_fpath)
    aa_seq, cord_tns, mask_mat, error = ProtStruct.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error is None, 'failed to parse the PDB file: %s (error: %s)' % (pdb_fpath, error)
    prot_data[chn_id] = (aa_seq, cord_tns, mask_mat)


def build_hdf_file(prot_data, chn_ids, hdf_fpath):
    """Build a HDF5 file w/ amino-acid sequences and full-atom 3D coordinates."""

    os.makedirs(os.path.dirname(os.path.realpath(hdf_fpath)), exist_ok=True)
    with h5py.File(hdf_fpath, 'w') as o_file:
        for chn_id in chn_ids:
            aa_seq, cord_tns, mask_mat = prot_data[chn_id]
            group = o_file.create_group(chn_id)
            group.create_dataset('seq', data=aa_seq)
            group.create_dataset('cord', data=cord_tns)
            group.create_dataset('mask', data=mask_mat)
    logging.info('HDF5 file built: %s', hdf_fpath)


def main():
    """Main entry."""

    # configurations
    data_dir = '/data1/protein/zyf/casp14'
    fas_dpath = os.path.join(data_dir, 'fasta.files')
    pdb_dpath = os.path.join(data_dir, 'pdb.files.native')
    did_fpath = os.path.join(data_dir, 'domain_ids.txt')
    hdf_fpath = os.path.join(data_dir, 'hdf5.files_fa/casp14.hdf5')

    # initialization
    tfold_init()

    # get the protein data
    with open(did_fpath, 'r') as i_file:
        prot_ids = [i_line.strip() for i_line in i_file]

    prot_data = {}
    for prot_id in prot_ids:
        fas_fpath = os.path.join(fas_dpath, '%s.fasta' % prot_id)
        pdb_fpath = os.path.join(pdb_dpath, '%s.pdb' % prot_id)
        get_prot_data(fas_fpath, pdb_fpath, prot_data)

    # build the HDF5 file
    build_hdf_file(prot_data, prot_ids, hdf_fpath)


if __name__ == '__main__':
    main()
