"""Utility functions."""
import fire
import os
import tempfile
import subprocess

def eval_pdb_file(pdb_fpath_mod, pdb_fpath_ref, metric, result_dict=None):
    """Evaluate the PDB file w/ specified metric."""

    if metric == 'gdt_ts':
        score = calc_gdt_ts(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'tm_scr':
        score = calc_tm_scr(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'lddt':
        score = calc_lddt(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'lddt_ca':
        score = calc_lddt_ca(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'lddt_bb':
        score = calc_lddt_bb(pdb_fpath_mod, pdb_fpath_ref)
    elif metric == 'lddt_check':
        score = calc_lddt_check(pdb_fpath_mod, pdb_fpath_ref)
    else:
        raise ValueError('unrecognized evaluation metric: ' + metric)

    if result_dict is not None:
        result_dict[(pdb_fpath_mod, metric)] = score

    return score

def calc_gdt_ts(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the GDT-TS score."""

    cmd_out = subprocess.check_output(
        ['DeepScore', pdb_fpath_mod, pdb_fpath_ref, '-P 0 -n -2'])
    line_str = cmd_out.decode('utf-8')
    gdt_ts = float(line_str.split()[14])

    return gdt_ts


def calc_tm_scr(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the TM-score."""

    cmd_out = subprocess.check_output(
        ['DeepScore', pdb_fpath_mod, pdb_fpath_ref, '-P 0 -n -2'])
    line_str = cmd_out.decode('utf-8')
    tm_scr = float(line_str.split()[11])

    return tm_scr

def calc_lddt(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the lDDT score."""

    cmd_out = subprocess.check_output(
        ['lddt',  pdb_fpath_mod, pdb_fpath_ref])
    line_strs = cmd_out.decode('utf-8').split('\n')
    for line_str in line_strs:
        if line_str.startswith('Global LDDT score'):
            lddt = float(line_str.split()[-1])
    return lddt

def calc_lddt_check(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the lDDT score."""

    cmd_out = subprocess.check_output(
        ['lddt','-f', '-p', './stereo_chemical_props.txt', pdb_fpath_mod, pdb_fpath_ref])
    line_strs = cmd_out.decode('utf-8').split('\n')
    for line_str in line_strs:
        if line_str.startswith('Global LDDT score'):
            lddt = float(line_str.split()[-1])
    return lddt


def calc_lddt_ca(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the lDDT-Ca score."""

    cmd_out = subprocess.check_output(
        ['lddt', '-c', pdb_fpath_mod, pdb_fpath_ref])
    line_strs = cmd_out.decode('utf-8').split('\n')
    for line_str in line_strs:
        if line_str.startswith('Global LDDT score'):
            lddt_ca = float(line_str.split()[-1])
    return lddt_ca


def calc_lddt_bb(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the lDDT score for backbone (N-CA-C-O) atoms."""
    def _build_pdb_file_bb(path_src, path_dst):
        os.makedirs(os.path.dirname(os.path.realpath(path_dst)), exist_ok=True)
        with open(path_src, 'r') as i_file, open(path_dst, 'w') as o_file:
            for i_line in i_file:
                if i_line.startswith('ATOM') \
                  and i_line[12:16].strip() in ['N', 'CA', 'C', 'O']:
                    o_file.write(i_line)

    # generate PDB files w/ backbone atoms only
    pdb_fname_ref, pdb_fname_mod = map(
        lambda fpath: os.path.splitext(os.path.basename(fpath))[0],
        [pdb_fpath_ref, pdb_fpath_mod])

    tmp_prefix = os.path.join(tempfile.gettempdir(),
                              f'{pdb_fname_ref}-{pdb_fname_mod}-evaluate-')

    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp_dpath:
        pdb_fpath_ref_bb = os.path.join(tmp_dpath,
                                        f'{pdb_fname_ref}-reference.pdb')
        pdb_fpath_mod_bb = os.path.join(tmp_dpath,
                                        f'{pdb_fname_mod}-model.pdb')
        _build_pdb_file_bb(pdb_fpath_ref, pdb_fpath_ref_bb)
        _build_pdb_file_bb(pdb_fpath_mod, pdb_fpath_mod_bb)

        # calculatethe the lDDT score for backbone atoms
        cmd_out = subprocess.check_output(
            ['lddt', pdb_fpath_mod_bb, pdb_fpath_ref_bb])
        line_strs = cmd_out.decode('utf-8').split('\n')
        for line_str in line_strs:
            if line_str.startswith('Global LDDT score'):
                lddt_bb = float(line_str.split()[-1])

    return lddt_bb

