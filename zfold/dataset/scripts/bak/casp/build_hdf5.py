"""Build HDF5 files for train/valid/test subsets.

CASP13:
> IMPROPER_CA_CA_DISTANCE: T0970-D1, T0984-D2, T0987-D2, T1013-D1
> INSUFFICIENT_ATOMS_W_COORDINATES: T0981-D2

CASP14:
> IMPROPER_CA_CA_DISTANCE: T1037-D1, T1042-D1
"""

import os
import logging

import h5py

from zfold.dataset.utils import tfold_init
from zfold.dataset.tools import PdbParser


def build_hdf5_file(fas_dpath, pdb_dpath, prot_ids, hdf_fpath):
    """Build a HDF5 file with specified protein IDs."""

    # show the greeting message
    logging.info('building the HDF5 file: %s', hdf_fpath)

    # obtain the FASTA sequence & atom coordinates (w/ masks)
    prot_data = {}
    parser = PdbParser(check_mode='lenient')  # skip non-fatal errors
    for prot_id in prot_ids:
        fas_fpath = os.path.join(fas_dpath, '%s.fasta' % prot_id)
        pdb_fpath = os.path.join(pdb_dpath, '%s.pdb' % prot_id)
        aa_seq, atom_cords, atom_masks, _, error = parser.run(pdb_fpath, fas_fpath=fas_fpath)
        assert error is None, 'failed to parse the PDB file: %s (error: %s)' % (pdb_fpath, error)
        prot_data[prot_id] = (aa_seq, atom_cords, atom_masks)

    # export the PDB data to a HDF5 file
    os.makedirs(os.path.dirname(os.path.realpath(hdf_fpath)), exist_ok=True)
    with h5py.File(hdf_fpath, 'w') as o_file:
        for prot_id, (aa_seq, atom_cords, atom_masks) in prot_data.items():
            group = o_file.create_group(prot_id)
            group.create_dataset('aa_seq', data=aa_seq)
            group.create_dataset('atom_cords', data=atom_cords)
            group.create_dataset('atom_masks', data=atom_masks)


def main():
    """Main entry."""

    # configurations
    data_dir = '/data1/protein/zyf/casp14'
    fas_dpath = os.path.join(data_dir, 'fasta.files')
    pdb_dpath = os.path.join(data_dir, 'pdb.files.native')
    did_fpath = os.path.join(data_dir, 'domain_ids.txt')
    hdf_fpath = os.path.join(data_dir, 'hdf5.files/casp14.hdf5')

    # initialization
    tfold_init()

    # build the HDF5 file
    with open(did_fpath, 'r') as i_file:
        prot_ids = [i_line.strip() for i_line in i_file]
    build_hdf5_file(fas_dpath, pdb_dpath, prot_ids, hdf_fpath)


if __name__ == '__main__':
    main()
