"""The evaluator for PDB files."""

import os
import shutil
import subprocess

from zfold.dataset.utils import get_rand_str
from zfold.dataset.utils import get_tmp_dpath


class PdbEvaluator():
    """The evaluator for PDB files."""

    def __init__(self):
        """Constructor function."""

        self.tmp_dpath = get_tmp_dpath()


    def run(self, pdb_fpath_mod, pdb_fpath_ref, metric, result_dict=None):
        """Evaluate the PDB file w/ specified metric."""

        if metric == 'gdt_ts':
            score = self.__calc_gdt_ts(pdb_fpath_mod, pdb_fpath_ref)
        elif metric == 'tm_scr':
            score = self.__calc_tm_scr(pdb_fpath_mod, pdb_fpath_ref)
        elif metric == 'lddt_ca':
            score = self.__calc_lddt_ca(pdb_fpath_mod, pdb_fpath_ref)
        elif metric == 'lddt':
            score = self.__calc_lddt(pdb_fpath_mod, pdb_fpath_ref)
        elif metric == 'lddt_bb':
            score = self.__calc_lddt_bb(pdb_fpath_mod, pdb_fpath_ref)
        else:
            raise ValueError('unrecognized evaluation metric: ' + metric)

        if result_dict is not None:
            result_dict[(pdb_fpath_mod, metric)] = score

        return score


    @classmethod
    def __calc_gdt_ts(cls, pdb_fpath_mod, pdb_fpath_ref):
        """Calculate the GDT-TS score."""

        cmd_out = subprocess.check_output(['DeepScore', pdb_fpath_mod, pdb_fpath_ref, '-P 0 -n -2'])
        line_str = cmd_out.decode('utf-8')
        gdt_ts = float(line_str.split()[14])

        return gdt_ts


    @classmethod
    def __calc_tm_scr(cls, pdb_fpath_mod, pdb_fpath_ref):
        """Calculate the TM-score."""

        cmd_out = subprocess.check_output(['DeepScore', pdb_fpath_mod, pdb_fpath_ref, '-P 0 -n -2'])
        line_str = cmd_out.decode('utf-8')
        tm_scr = float(line_str.split()[11])

        return tm_scr


    @classmethod
    def __calc_lddt_ca(cls, pdb_fpath_mod, pdb_fpath_ref):
        """Calculate the lDDT-Ca score."""

        cmd_out = subprocess.check_output(['lddt', '-c', pdb_fpath_mod, pdb_fpath_ref])
        line_strs = cmd_out.decode('utf-8').split('\n')
        for line_str in line_strs:
            if line_str.startswith('Global LDDT score'):
                lddt_ca = float(line_str.split()[-1])

        return lddt_ca

    @classmethod
    def __calc_lddt(cls, pdb_fpath_mod, pdb_fpath_ref):
        """Calculate the lDDT-Ca score."""

        cmd_out = subprocess.check_output(['lddt', pdb_fpath_mod, pdb_fpath_ref])
        line_strs = cmd_out.decode('utf-8').split('\n')
        for line_str in line_strs:
            if line_str.startswith('Global LDDT score'):
                lddt_ca = float(line_str.split()[-1])

        return lddt_ca


    def __calc_lddt_bb(self, pdb_fpath_mod, pdb_fpath_ref):
        """Calculate the lDDT score for backbone (N-CA-C-O) atoms."""

        # generate PDB files w/ backbone atoms only
        pdb_fpath_mod_bb = os.path.join(self.tmp_dpath, '%s.pdb' % get_rand_str())
        pdb_fpath_ref_bb = os.path.join(self.tmp_dpath, '%s.pdb' % get_rand_str())
        self.__build_pdb_file_bb(pdb_fpath_mod, pdb_fpath_mod_bb)
        self.__build_pdb_file_bb(pdb_fpath_ref, pdb_fpath_ref_bb)

        # calculatethe the lDDT score for backbone atoms
        cmd_out = subprocess.check_output(['lddt', pdb_fpath_mod_bb, pdb_fpath_ref_bb])
        line_strs = cmd_out.decode('utf-8').split('\n')
        for line_str in line_strs:
            if line_str.startswith('Global LDDT score'):
                lddt_bb = float(line_str.split()[-1])

        # clean-up
        os.remove(pdb_fpath_mod_bb)
        os.remove(pdb_fpath_ref_bb)

        return lddt_bb


    @classmethod
    def __build_pdb_file_bb(cls, path_src, path_dst):
        """Build a PDB file with backbone atoms only."""

        os.makedirs(os.path.dirname(os.path.realpath(path_dst)), exist_ok=True)
        with open(path_src, 'r') as i_file, open(path_dst, 'w') as o_file:
            for i_line in i_file:
                if i_line.startswith('ATOM') and i_line[12:16].strip() in ['N', 'CA', 'C', 'O']:
                    o_file.write(i_line)
