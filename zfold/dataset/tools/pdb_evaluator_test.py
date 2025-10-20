"""Unit-tests for the <PdbEvaluator> class."""

import os
import logging

from zfold.dataset.utils import zfold_init
from zfold.dataset.tools import PdbEvaluator


def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    pdb_fpath_natv = os.path.join(curr_dir, 'examples/1ahoA00_native.pdb')
    pdb_fpath_decy = os.path.join(curr_dir, 'examples/1ahoA00_decoy.pdb')
    metrics = ['gdt_ts', 'tm_scr', 'lddt_ca', 'lddt_bb']

    # initialization
    zfold_init()
    evaluator = PdbEvaluator()

    # parse the PDB file
    logging.info('native PDB file: %s', pdb_fpath_natv)
    logging.info(' decoy PDB file: %s', pdb_fpath_decy)
    for metric in metrics:
        score = evaluator.run(pdb_fpath_decy, pdb_fpath_natv, metric)
        logging.info('%s => %.4f', metric, score)


if __name__ == '__main__':
    main()
