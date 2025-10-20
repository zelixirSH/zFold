"""Utility functions for loading/saving PDB chain IDs."""

import os


def load_chn_ids_raw(path):
    """Load PDB chain IDs in the raw format."""

    with open(path, 'r') as i_file:
        chn_ids = [i_line.strip() for i_line in i_file]

    return chn_ids


def load_chn_ids_grp(path):
    """Load PDB chain IDs in the grouped format."""

    with open(path, 'r') as i_file:
        chn_ids = [i_line.strip().split() for i_line in i_file]

    return chn_ids


def save_chn_ids_raw(chn_ids, path):
    """Save PDB chain IDs in the raw format."""

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        o_file.write('\n'.join(sorted(list(chn_ids))) + '\n')


def save_chn_ids_grp(chn_ids, path):
    """Save PDB chain IDs in the grouped format."""

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        o_file.write('\n'.join([' '.join(sorted(list(x))) for x in chn_ids]) + '\n')
