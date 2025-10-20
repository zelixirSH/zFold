"""Unit-tests for the <StructChecker> class."""

import os

from zfold.dataset.tools.struct_checker import StructChecker

def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    dat_dpath = os.path.join(curr_dir, 'examples')
    pdb_fpath_natv = os.path.join(dat_dpath, '1jjdA00_native.pdb')
    pdb_fpath_clsh = os.path.join(dat_dpath, '1jjdA00_clash.pdb')
    pdb_fpath_flip = os.path.join(dat_dpath, '1jjdA00_flipped.pdb')

    # test <StructChecker> with native & perturbed PDB files
    checker = StructChecker()
    print('=== steric clash check ===')
    for pdb_fpath in [pdb_fpath_natv, pdb_fpath_clsh, pdb_fpath_flip]:
        is_valid = checker.check_pdb_file(pdb_fpath, task='clash')
        print('{} => {}'.format(pdb_fpath, is_valid))
    print('=== handedness check ===')
    for pdb_fpath in [pdb_fpath_natv, pdb_fpath_clsh, pdb_fpath_flip]:
        is_valid = checker.check_pdb_file(pdb_fpath, task='handedness')
        print('{} => {}'.format(pdb_fpath, is_valid))


if __name__ == '__main__':
    main()
