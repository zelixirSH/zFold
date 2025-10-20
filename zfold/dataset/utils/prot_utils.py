"""Protein-related utility functions."""

import os
import shutil
import subprocess

import numpy as np

from zfold.dataset.utils.comm_utils import get_rand_str
from zfold.dataset.utils.file_utils import get_tmp_dpath

# constants
AA_NAMES_DICT_1TO3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS',
    'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN',
    'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}
AA_NAMES_DICT_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_NAMES_1CHAR = 'ACDEFGHIKLMNPQRSTVWY'


def parse_fas_file(path):
    """Parse the FASTA file.

    Args:
    * path: path to the FASTA file

    Returns:
    * prot_id: protein ID (as in the commentary line)
    * aa_seq: amino-acid sequence
    """

    assert os.path.exists(path), 'FASTA file does not exist: ' + path
    with open(path, 'r') as i_file:
        i_lines = [i_line.strip() for i_line in i_file]
        prot_id = i_lines[0][1:]
        aa_seq = ''.join(i_lines[1:])

    return prot_id, aa_seq


def parse_pdb_file(path, atom_name='CA'):
    """Parse the PDB file to obtain atom coordinates.

    Args:
    * path: path to the PDB file

    Returns:
    * cord_mat: atoms' 3D coordinates (N x 3)
    * mask_vec: atoms' validness masks (N)
    """

    # check whether the PDB file exists
    assert os.path.exists(path), 'PDB file does not exist: ' + path

    # parse the PDB file
    idxs_resd = set()
    atom_dict = {}
    with open(path, 'r') as i_file:
        for i_line in i_file:
            if not i_line.startswith('ATOM'):
                continue
            idx_resd = int(i_line[22:26])
            idxs_resd.add(idx_resd)
            if i_line[12:16].strip() == atom_name:
                cord_x = float(i_line[30:38])
                cord_y = float(i_line[38:46])
                cord_z = float(i_line[46:54])
                atom_dict[idx_resd] = np.array([cord_x, cord_y, cord_z], dtype=np.float32)

    # extract the coordinate matrix & mask vector
    idxs_resd = sorted(list(idxs_resd))
    cord_mat = np.stack([
        atom_dict.get(x, np.zeros((3), dtype=np.float32)) for x in idxs_resd
    ], axis=0)
    mask_vec = np.array([1 if x in atom_dict else 0 for x in idxs_resd], dtype=np.int8)

    return cord_mat, mask_vec


def export_fas_file(prot_id, aa_seq, path):
    """Export the amino-acid sequence to a FASTA file.

    Args:
    * prot_id: protein ID (as in the commentary line)
    * aa_seq: amino-acid sequence
    * path: path to the FASTA file

    Returns: n/a
    """

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        o_file.write('>%s\n%s\n' % (prot_id, aa_seq))


def export_pdb_file(aa_seq, atom_cords, path, atom_masks=None):
    """Export the 3D structure to a PDB file.

    Args:
    * aa_seq: amino-acid sequence
    * atom_cords: atom coordinates (['CA']: L x 3 / ['N', 'CA', 'C']: L x 3 x 3)
    * path: path to the PDB file
    * (optional) atom_masks: atom masks (['CA']: L / ['N', 'CA', 'C']: L x 3)

    Returns: n/a
    """

    # configurations
    alt_loc = ' '
    chain_id = 'A'
    i_code = ' '
    occupancy = 1.0
    temp_factor = 1.0
    charge = ' '
    cord_min = -999.999
    cord_max = 9999.999
    seq_len = len(aa_seq)

    # take all the atom coordinates as valid, if not specified
    if atom_masks is None:
        atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

    # determine the set of atom names (per residue)
    assert (atom_cords.size == seq_len * 3) or (atom_cords.size == seq_len * 3 * 3)
    atom_cords_ext = np.reshape(atom_cords, [seq_len, -1, 3])
    atom_masks_ext = np.reshape(atom_masks, [seq_len, -1])
    atom_names = ['CA'] if atom_cords_ext.shape[1] == 1 else ['N', 'CA', 'C']

    # reset invalid values in atom coordinates
    atom_cords_ext = np.clip(atom_cords_ext, cord_min, cord_max)
    atom_cords_ext[np.isnan(atom_cords_ext)] = 0.0
    atom_cords_ext[np.isinf(atom_cords_ext)] = 0.0

    # export the 3D structure to a PDB file
    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        n_atoms = 0
        for idx_resd, resd_name in enumerate(aa_seq):
            for idx_atom, atom_name in enumerate(atom_names):
                if atom_masks_ext[idx_resd, idx_atom] == 0:
                    continue
                n_atoms += 1
                line_str = ''.join([
                    'ATOM  ',
                    '%5d' % n_atoms,
                    '  ' + atom_name + ' ' * (3 - len(atom_name)),
                    alt_loc,
                    '%3s' % AA_NAMES_DICT_1TO3[resd_name],
                    ' %s' % chain_id,
                    '%4d' % (idx_resd + 1),
                    '%s   ' % i_code,
                    '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 0],
                    '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 1],
                    '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 2],
                    '%6.2f' % occupancy,
                    '%6.2f' % temp_factor,
                    ' ' * 10,
                    '%2s' % atom_name[0],
                    '%2s' % charge,
                ])
                assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                o_file.write(line_str + '\n')


def calc_gdt_ts(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the GDT-TS score.

    Args:
    * pdb_fpath_mod: path to the PDB file (modelled structure)
    * pdb_fpath_ref: path to the PDB file (reference structure)

    Returns:
    * score: evaluation result
    """

    cmd_out = subprocess.check_output(['DeepScore', pdb_fpath_mod, pdb_fpath_ref, '-P 0 -n -2'])
    line_str = cmd_out.decode('utf-8')
    score = float(line_str.split()[14])

    return score


def calc_tm_scr(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the TM-score.

    Args:
    * pdb_fpath_mod: path to the PDB file (modelled structure)
    * pdb_fpath_ref: path to the PDB file (reference structure)

    Returns:
    * score: evaluation result
    """

    cmd_out = subprocess.check_output(['DeepScore', pdb_fpath_mod, pdb_fpath_ref, '-P 0 -n -2'])
    line_str = cmd_out.decode('utf-8')
    score = float(line_str.split()[11])

    return score


def calc_lddt_ca(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the lDDT-Ca score.

    Args:
    * pdb_fpath_mod: path to the PDB file (modelled structure)
    * pdb_fpath_ref: path to the PDB file (reference structure)

    Returns:
    * score: evaluation result
    """

    cmd_out = subprocess.check_output(['lddt', '-c', pdb_fpath_mod, pdb_fpath_ref])
    line_strs = cmd_out.decode('utf-8').split('\n')
    for line_str in line_strs:
        if line_str.startswith('Global LDDT score'):
            score = float(line_str.split()[-1])
            break

    return score


def calc_lddt_bb(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the lDDT score for backbone (N-CA-C) atoms.

    Args:
    * pdb_fpath_mod: path to the PDB file (modelled structure)
    * pdb_fpath_ref: path to the PDB file (reference structure)

    Returns:
    * score: evaluation result
    """

    def _build_pdb_file_bb(path_src, path_dst):
        os.makedirs(os.path.dirname(os.path.realpath(path_dst)), exist_ok=True)
        with open(path_src, 'r') as i_file, open(path_dst, 'w') as o_file:
            for i_line in i_file:
                if i_line.startswith('ATOM') and i_line[12:16].strip() in ['N', 'CA', 'C']:
                    o_file.write(i_line)

    # generate PDB files w/ backbone atoms only
    tmp_dpath = get_tmp_dpath()
    pdb_fpath_mod_bb = os.path.join(tmp_dpath, '%s.pdb' % get_rand_str())
    pdb_fpath_ref_bb = os.path.join(tmp_dpath, '%s.pdb' % get_rand_str())
    _build_pdb_file_bb(pdb_fpath_mod, pdb_fpath_mod_bb)
    _build_pdb_file_bb(pdb_fpath_ref, pdb_fpath_ref_bb)

    # calculatethe the lDDT score for backbone atoms
    cmd_out = subprocess.check_output(['lddt', pdb_fpath_mod_bb, pdb_fpath_ref_bb])
    line_strs = cmd_out.decode('utf-8').split('\n')
    for line_str in line_strs:
        if line_str.startswith('Global LDDT score'):
            score = float(line_str.split()[-1])
            break

    # clean-up
    shutil.rmtree(tmp_dpath)

    return score


def eval_pdb_file(pdb_fpath_mod, pdb_fpath_ref, metric, result_dict=None):
    """Evaluate the PDB file w/ specified metric.

    Args:
    * pdb_fpath_mod: path to the PDB file (modelled structure)
    * pdb_fpath_ref: path to the PDB file (reference structure)
    * metric: evaluation metric
    * (optional) result_dict: dict of evaluation results

    Returns:
    * score: evaluation result
    """

    if metric == 'gdt_ts':
        score = calc_gdt_ts(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'tm_scr':
        score = calc_tm_scr(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'lddt_ca':
        score = calc_lddt_ca(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'lddt_bb':
        score = calc_lddt_bb(pdb_fpath_mod, pdb_fpath_ref)
    else:
        raise ValueError('unrecognized evaluation metric: ' + metric)

    if result_dict is not None:
        result_dict[(pdb_fpath_mod, metric)] = score

    return score
