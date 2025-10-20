"""Unit-tests for the <PdbParser> class."""

import os
import logging

from zfold.dataset.utils import zfold_init
from zfold.dataset.tools import PdbParser


def check_outputs(aa_seq, atom_cords, atom_masks, structure, error_msg):
    """Check <PdbParser>'s outputs."""

    if error_msg is not None:
        logging.warning('error message: %s', error_msg)
    else:  # no error is detected
        logging.info('amino-acid sequence: %s (%d residues)', aa_seq, len(aa_seq))
        logging.info('atom coordinates: %s', str(atom_cords.shape))
        logging.info('atom masks: %s', str(atom_masks.shape))
        logging.info('BioPython structure: %s', str(structure))


def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    pdb_fpath = os.path.join(curr_dir, 'examples/1a34.pdb')
    pgz_fpath = os.path.join(curr_dir, 'examples/pdb1a34.ent.gz')
    chain_id = 'A'

    # initialization
    zfold_init()
    parser = PdbParser()

    # parse the PDB file
    logging.info('parsing the PDB file: %s', pdb_fpath)
    aa_seq, atom_cords, atom_masks, structure, error_msg = parser.run(pdb_fpath, chain_id=chain_id)
    check_outputs(aa_seq, atom_cords, atom_masks, structure, error_msg)

    # parse the GZIP-compressed PDB file
    logging.info('parsing the GZIP-compressed PDB file: %s', pgz_fpath)
    aa_seq, atom_cords, atom_masks, structure, error_msg = parser.run(pgz_fpath, chain_id=chain_id)
    check_outputs(aa_seq, atom_cords, atom_masks, structure, error_msg)


if __name__ == '__main__':
    main()
