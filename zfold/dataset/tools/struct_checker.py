"""Check for potential issues in the 3D structure."""

import os
import tempfile
import subprocess

import numpy as np

from zfold.dataset.utils import parse_pdb_file
from zfold.dataset.utils import export_pdb_file
from zfold.dataset.utils import calc_dihedral_angle

class StructChecker():
    """Check for potential issues in the 3D structure."""

    def __init__(self):
        """Constructor function."""

        self.eps = 1e-6

        # 1. handedness
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.stt_fpath = os.path.join(curr_dir, 'ca_dihd_stats.txt')
        assert os.path.exists(self.stt_fpath), \
            'CA atoms\' dihedral angle statistics file does not exist' + self.stt_fpath
        self.dihd_stats = self.__restore_dihd_stats(self.stt_fpath)

        # 2. steric clashes
        cmd_out = subprocess.check_output(['which', 'lddt'])
        self.bin_fpath = cmd_out.decode('utf-8').strip()
        self.prm_fpath = os.path.join(os.path.dirname(self.bin_fpath), 'stereo_chemical_props.txt')
        assert os.path.exists(self.bin_fpath), 'binary file does not exist: ' + self.bin_fpath
        assert os.path.exists(self.prm_fpath), 'parameter file does not exist: ' + self.prm_fpath


    def check_pdb_file(self, path, task='clash', result_dict=None):
        """Check for structural issues from the PDB file."""

        # run the corresponding task
        if task == 'clash':
            is_valid = self.__check_clash(path)
        elif task == 'handedness':
            cord_mat, mask_vec = parse_pdb_file(path, atom_name='CA')
            is_valid = self.__check_handedness(cord_mat, mask_vec)
        else:
            raise ValueError('unrecognized task name: ' + task)

        # record results in the dict
        if result_dict is not None:
            result_dict[path] = is_valid

        return is_valid


    def check_cord_mat(self, cord_mat, mask_vec=None, task='clash', key=None, result_dict=None):
        """Check for structural issues from the 3D coordinate matrix."""

        # if <mask_vec> is not provided, then all the 3D coordinates are treated as valid
        if mask_vec is None:
            mask_vec = np.ones((cord_mat.shape[0]), dtype=np.int8)

        # run the corresponding task
        if task == 'clash':
            with tempfile.NamedTemporaryFile(delete=False) as o_file:
                pdb_fpath = o_file.name
            aa_seq = 'A' * cord_mat.shape[0]
            export_pdb_file(aa_seq, cord_mat, pdb_fpath, atom_masks=mask_vec)
            is_valid = self.__check_clash(pdb_fpath)
            os.remove(pdb_fpath)
        elif task == 'handedness':
            is_valid = self.__check_handedness(cord_mat, mask_vec)
        else:
            raise ValueError('unrecognized task name: ' + task)

        # record results in the dict
        if (key is not None) and (result_dict is not None):
            result_dict[key] = is_valid

        return is_valid


    @classmethod
    def __restore_dihd_stats(cls, path):
        """Restore CA atoms' dihedral angle statistics."""

        hist_range = None
        prob_vec_raw = []
        with open(path, 'r') as i_file:
            for i_line in i_file:
                ubnd, lbnd, prob = [float(x) for x in i_line.split()]
                if hist_range is None:
                    hist_range = (ubnd, lbnd)
                else:
                    hist_range = (min(ubnd, hist_range[0]), max(lbnd, hist_range[1]))
                prob_vec_raw.append(prob)
        prob_vec = np.array(prob_vec_raw, dtype=np.float32)

        return {'n_bins': prob_vec.size, 'range': hist_range, 'prob': prob_vec}


    def __check_clash(self, pdb_fpath):
        """Check for the steric clash issue in the 3D structure."""

        # run <lddt> with structural check enabled
        cmd_out = subprocess.check_output(
            [self.bin_fpath, pdb_fpath, pdb_fpath, '-f', '-p', self.prm_fpath])
        line_strs = cmd_out.decode('utf-8').split('\n')

        # parse line strings to determine whether there exists any steric clashes
        for line_str in line_strs:
            if line_str.startswith('Global LDDT score'):
                is_valid = float(line_str.split()[-1]) == 1.0
                break

        return is_valid


    def __check_handedness(self, cord_mat_raw, mask_vec):
        """Check for the handedness issue in the 3D structure."""

        # obtain mirrored CA atoms' coordinates
        cord_mat_mrr = cord_mat_raw * np.array([-1, 1, 1], dtype=np.float32)[None, :]

        # calculate CA atoms' dihedral angles for both standard & reversed coordinates
        dihd_angls_raw = self.__calc_dihd_angls(cord_mat_raw, mask_vec)
        dihd_angls_mrr = self.__calc_dihd_angls(cord_mat_mrr, mask_vec)

        # calculate the KL-divergence between CA atoms' dihedral angle distributions
        kl_div_raw = self.__calc_kl_div(dihd_angls_raw)
        kl_div_mrr = self.__calc_kl_div(dihd_angls_mrr)
        is_valid = (kl_div_raw < kl_div_mrr)

        return is_valid


    @classmethod
    def __calc_dihd_angls(cls, cord_mat, mask_vec):
        """Calculate CA atoms' dihedral angles."""

        dihd_angls_raw = []
        seq_len = cord_mat.shape[0]
        for idx in range(seq_len - 4):
            if mask_vec[idx:idx + 4].min() != 1:
                continue
            dihd_angls_raw.append(calc_dihedral_angle(
                cord_mat[idx], cord_mat[idx + 1], cord_mat[idx + 2], cord_mat[idx + 3]))
        dihd_angls = np.array(dihd_angls_raw, dtype=np.float32)

        return dihd_angls


    def __calc_kl_div(self, dihd_angls):
        """Calculate the KL-divergence between CA atoms' dihedral angle distributions."""

        hist, _ = np.histogram(
            dihd_angls, bins=self.dihd_stats['n_bins'], range=self.dihd_stats['range'])
        prob_qry = hist / (np.sum(hist) + self.eps)
        prob_ref = self.dihd_stats['prob']
        kl_div = np.sum(prob_qry * np.log((prob_qry + self.eps) / prob_ref))

        return kl_div
