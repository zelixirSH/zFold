"""Utility functions."""

import os
import numpy as np
import torch
import logging
import subprocess
from collections import defaultdict

from zfold.dataset.utils import AA_NAMES_1CHAR
from zfold.dataset.utils import AA_NAMES_DICT_1TO3
from zfold.dataset.utils import cvt_to_one_hot

def send_to_device(data_dict_src, device):
    """Send the data dict to the specified device."""

    data_dict_dst = {}
    for key, val in data_dict_src.items():
        if isinstance(val, str):
            data_dict_dst[key] = val
        elif isinstance(val, torch.Tensor):
            data_dict_dst[key] = val.to(device)
        elif isinstance(val, dict):
            data_dict_dst[key] = send_to_device(val, device)
        elif isinstance(val, list):
            if isinstance(val[0], str):
                data_dict_dst[key] = val
            else:
                assert isinstance(val[0], dict)
                data_dict_dst[key] = [send_to_device(x, device) for x in val]
        else:
            raise ValueError('unrecognized value type: %s' % type(val))

    return data_dict_dst


def inspect_data_dict(data_dict, prefix=''):
    """Inspect the data dict."""

    for key, val in data_dict.items():
        if isinstance(val, dict):
            inspect_data_dict(val, prefix=(prefix + key + '/'))
        elif isinstance(val, (torch.Tensor, np.ndarray)):
            logging.info('%s: %s / %s', prefix + key, val.shape, val.dtype)
        else:
            logging.info('%s: %s', prefix + key, val)


def setup_path_dict(config, name=None, subset=None):
    """Setup the dict of file & directory paths."""

    # determine the file name to protein IDs
    if subset is None:
        assert name in ['casp13', 'casp14', 'cameo']
        pid_fname = config['pid_fname']
    elif name == 'pdb28k':
        assert subset in ['trn', 'val', 'tst']
        pid_fname = config['pid_fname_' + subset]
    elif name == 'semi':
        pid_fname = config['pid_fname']
    elif name == 'semi_msa_100k':
        pid_fname = config['pid_fname']
    elif name == 'semi_msa_800k_part1':
        assert subset in ['0','5']
        pid_fname = config['pid_fname']
    elif name == 'semi_af2db':
        assert subset in ['0','5']
        pid_fname = config['pid_fname']
    else:
        raise ValueError('unrecognized dataset name: %s' % name)

    # setup the dict of file & directory paths
    path_dict = defaultdict(lambda: None)
    path_dict['pid_fpath'] = os.path.join(config['root_dir'], pid_fname)
    path_dict['hdf_dpath'] = os.path.join(config['root_dir'], config['hdf_dname'])

    if ',' in config['a3m_dname']:
        path_dict['a3m_dpath'] = []
        a3m_dnames = config['a3m_dname'].split(',')
        for a3m_dname in a3m_dnames:
            if os.path.exists(a3m_dname):
                path_dict['a3m_dpath'].append(a3m_dname)
            else:
                path_dict['a3m_dpath'].append(os.path.join(config['root_dir'], a3m_dname))
    else:
        path_dict['a3m_dpath'] = os.path.join(config['root_dir'], config['a3m_dname'])

    path_dict['lbl_dpath'] = os.path.join(config['root_dir'], config['lbl_dname'])

    if config['tpl_dname'] != 'NOT_PROVIDED':
        path_dict['tpl_dpath'] = os.path.join(config['root_dir'], config['tpl_dname'])
    if config['npz_dname'] != 'NOT_PROVIDED':
        path_dict['npz_dpath'] = os.path.join(config['root_dir'], config['npz_dname'])

    if 'plddt_fname' in config:
        path_dict['plddt_fpath'] = os.path.join(config['root_dir'], config['plddt_fname'])

    path_dict['a3m_format'] = config['a3m_format']

    return path_dict

def setup_path_dict_no_root(config, name=None, subset=None):
    """Setup the dict of file & directory paths."""

    # determine the file name to protein IDs
    if subset is None:
        assert name in ['casp13', 'casp14', 'cameo', 'trrosetta']
        pid_fname = config['pid_fname']
    elif name == 'pdb28k':
        assert subset in ['trn', 'val', 'tst']
        pid_fname = config['pid_fname_' + subset]
    elif name == 'semi':
        pid_fname = config['pid_fname']
    else:
        raise ValueError('unrecognized dataset name: %s' % name)

    # setup the dict of file & directory paths
    path_dict = defaultdict(lambda: None)
    path_dict['pid_fpath'] = pid_fname
    path_dict['hdf_dpath'] = config['hdf_dname']

    if ',' in config['a3m_dname']:
        path_dict['a3m_dpath'] = []
        a3m_dnames = config['a3m_dname'].split(',')
        for a3m_dname in a3m_dnames:
            if os.path.exists(a3m_dname):
                path_dict['a3m_dpath'].append(a3m_dname)
    else:
        path_dict['a3m_dpath'] = config['a3m_dname']

    path_dict['lbl_dpath'] = config['lbl_dname']

    if config['tpl_dname'] != 'NOT_PROVIDED':
        path_dict['tpl_dpath'] = config['tpl_dname']

    if config['npz_dname'] != 'NOT_PROVIDED':
        path_dict['npz_dpath'] = config['npz_dname']

    if 'plddt_fname' in config:
        path_dict['plddt_fpath'] = config['plddt_fname']
    if config['highres_fpath'] != 'NOT_PROVIDED':
        path_dict['highres_fpath'] = config['highres_fpath']

    return path_dict


def calc_nois_stds(nois_std_beg, nois_std_end, n_nois_levls):
    """Calculate a series of random noise's standard deviations."""

    base = 2.0  # equivalent for any values larger than 1
    nois_std_beg_log = np.log(nois_std_beg) / np.log(base)
    nois_std_end_log = np.log(nois_std_end) / np.log(base)
    nois_stds = np.logspace(
        nois_std_beg_log, nois_std_end_log, num=n_nois_levls, base=base, dtype=np.float32)

    return nois_stds


def build_onht_tns(aa_seqs):
    """Build one-hot encodings of amino-acid sequences."""

    aa_idxs = np.array([[AA_NAMES_1CHAR.index(z) for z in x] for x in aa_seqs], dtype=np.int32)
    onht_tns = cvt_to_one_hot(aa_idxs, len(AA_NAMES_1CHAR))  # N x L x 20

    return torch.tensor(onht_tns, dtype=torch.float32)


def build_ssep_tns(aa_seqs, n_dims=20, div_fctr=1000):
    """Build sequence separations of amino-acid sequences."""

    seq_len = len(aa_seqs[0])
    n_levls = n_dims // 2
    div_fctrs = np.power(div_fctr, np.arange(n_levls) / (n_levls - 1))
    sgap_mat = np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]
    ssep_tns = np.repeat(np.concatenate([
        np.sin(sgap_mat[:, :, None] / div_fctrs[None, None, :]),
        np.cos(sgap_mat[:, :, None] / div_fctrs[None, None, :]),
    ], axis=2)[None, :, :, :], len(aa_seqs), axis=0)  # N x L x L x 20

    return torch.tensor(ssep_tns, dtype=torch.float32)


def export_pdb_file(aa_seq, cord_tns, path, mask_tns=None):
    """Export the 3D structure to a PDB file."""

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

    # pre-processing
    atom_names = ['CA'] if cord_tns.shape[1] == 1 else ['N', 'CA', 'C', 'O']
    cord_tns = np.clip(cord_tns, cord_min, cord_max)
    cord_tns[np.isnan(cord_tns)] = 0.0
    cord_tns[np.isinf(cord_tns)] = 0.0
    if mask_tns is None:
        mask_tns = np.ones(cord_tns.shape[:-1], dtype=np.int8)

    # export the 3D structure to a PDB file
    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        n_atoms = 0
        for idx_resd, resd_name in enumerate(aa_seq):
            for idx_atom, atom_name in enumerate(atom_names):
                if mask_tns[idx_resd, idx_atom] == 0:
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
                    '%8.3f' % cord_tns[idx_resd, idx_atom, 0],
                    '%8.3f' % cord_tns[idx_resd, idx_atom, 1],
                    '%8.3f' % cord_tns[idx_resd, idx_atom, 2],
                    '%6.2f' % occupancy,
                    '%6.2f' % temp_factor,
                    ' ' * 10,
                    '%2s' % atom_name[0],
                    '%2s' % charge,
                ])
                assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                o_file.write(line_str + '\n')
