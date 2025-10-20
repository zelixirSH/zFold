"""The protein dataset.

Symbols:
- N_b: number of random crops - base elements
- N_c: number of random crops - 3D coordinates for SE(3)-Fold
- L: sequence length
- M: number of atoms per residue ('ca': M = 1 / 'bb': M = 3 / 'fa': M = 14)
- K: number of sequences in the multiple sequence alignment

Data Table:
- base
  - id: protein ID (string)
  - seq: list of cropped AA sequences of size N_b
  - cord: ground-truth 3D coordinates of size N_b x L x M x 3
  - cmsk: ground-truth 3D coordinates' validness masks of size N_b x L x M
  - fram: ground-truth per-residue local frames of size N_b x L x 4 x 3
  - fmsk: ground-truth per-residue local frames' validness masks of size N_b x L
  - angl: ground-truth per-residue torsion angles of size N_b x L x 7 x 2
  - amsk: ground-truth per-residue torsion angles' validness masks of size N_b x L x 7
- se3f:
  - cord: perturbed 3D coordinates of size N_c x L x M x 3
  - nstd: random noise's standard deviations of size N_c
  - grad: ground-truth gradients over perturbed 3D coordinates of size N_c x L x M x 3
- feat:
  - msa-t: ground-truth MSA tokens of size N_b x K x L
  - msa-p: perturbed MSA tokens of size N_b x K x L
  - msa-m: perturbed MSA tokens' validness masks of size N_b x K x L
  - t1ds: 1D features from structural templates of size N_b x M x L x 23
  - t2ds: 2D features from structural templates of size N_b x M x L x L x 1
  - dist: inter-residue distance predictions of size N_b x L x L x 37
  - angl: inter-residue orientation predictions of size N_b x L x L x 75
  - mfea: MSA features of size N_b x K x L x 384
  - pfea: pair features of size N_b x 256 x L x L
- labl:
  - X-idx: classification label indices of size N_b x L x L (X: 'cb' / 'om' / th' / 'ph')
  - X-msk: classification validness masks of size N_b x L x L (X: 'cb' / 'om' / th' / 'ph')
"""

import os
import string
import random
import logging
import itertools
from typing import List, Tuple
from collections import defaultdict

import pickle
import h5py
import numpy as np
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
from zfold.utils.gen_fea import parse_tpl_file
from zfold.network.esm.data import Alphabet
from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.dataset.dbutils import calc_nois_stds
from zfold.utils.gen_fea import parse_msanpz

def parse_hdf_file(path, prot_id):
    """Parse the HDF5 file."""

    # parse the HDF5 file
    with h5py.File(path, 'r', driver='core') as i_file:
        aa_seq = i_file[prot_id]['seq'][()]#.decode('utf-8')
        cord_tns_orig = i_file[prot_id]['cord-o'][()]
        cmsk_mat_orig = i_file[prot_id]['cmsk-o'][()]
        cord_tns_reco = i_file[prot_id]['cord-r'][()]
        cmsk_mat_reco = i_file[prot_id]['cmsk-r'][()]
        fram_tns = i_file[prot_id]['fram'][()]
        fmsk_vec = i_file[prot_id]['fmsk'][()]
        angl_tns = i_file[prot_id]['angl'][()]
        amsk_mat = i_file[prot_id]['amsk'][()]

    # pack into a dict
    data_dict = {
        'id': prot_id,
        'seq': aa_seq,
        'cord-o': torch.tensor(cord_tns_orig, dtype=torch.float32),  # L x 14 x 3
        'cmsk-o': torch.tensor(cmsk_mat_orig, dtype=torch.int8),  # L x 14
        'cord-r': torch.tensor(cord_tns_reco, dtype=torch.float32),  # L x 14 x 3
        'cmsk-r': torch.tensor(cmsk_mat_reco, dtype=torch.int8),  # L x 14
        'fram': torch.tensor(fram_tns, dtype=torch.float32),  # L x 4 x 3
        'fmsk': torch.tensor(fmsk_vec, dtype=torch.int8),  # L
        'angl': torch.tensor(angl_tns, dtype=torch.float32),  # L x 7 x 2
        'amsk': torch.tensor(amsk_mat, dtype=torch.int8),  # L x 7
    }

    return data_dict


def parse_a3m_file(path, alphabet, msa_depth, is_train):
    """Parse the A3M file."""

    # === ESM pre-processing - BELOW ===
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    def read_sequence(filename: str) -> Tuple[str, str]:
        """Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))
        return record.description, str(record.seq)

    def remove_insertions(sequence: str) -> str:
        """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
        return sequence.translate(translation)

    def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
        """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, remove_insertions(str(record.seq)))
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

    def read_msa_v2(filename: str, nseq: int, mode: str) -> List[Tuple[str, str]]:
        """Read sequences from an MSA file, automatically removes insertions - v2."""
        records_raw = list(SeqIO.parse(filename, "fasta"))
        records = [(x.description, remove_insertions(str(x.seq))) for x in records_raw]
        if len(records) <= nseq:
            return records
        elif mode == 'topk':
            return records[:nseq]
        else:  # then <mode> must be 'sample'
            return [records[0]] + random.sample(records, nseq - 1)

    # === ESM pre-processing - ABOVE ===
    # parse the A3M file
    converter = alphabet.get_batch_converter()
    msa_data = [read_msa_v2(path, msa_depth, mode=('sample' if is_train else 'topk'))]
    _, _, msa_tokens = converter(msa_data)
    msa_tokens_true = msa_tokens.squeeze(0).data.cpu().numpy()[:, 1:]

    return msa_tokens_true


def parse_npz_file(path):
    """Parse the NPZ file containing intermediate embeddings."""

    # parse the NPZ file
    with np.load(path) as npz_data:
        dist_tns = npz_data['dist']
        angl_tns = np.concatenate([npz_data['omega'], npz_data['theta'], npz_data['phi']], axis=-1)
        mfea_tns = npz_data['msa_feats'][0, 0]  # old format
        #mfea_tns = npz_data['msa_feats'][0]  # new format
        pfea_tns = npz_data['pair_feats'][0]

    # pack into a dict
    data_dict = {
        'dist': torch.tensor(dist_tns, dtype=torch.float32),  # L x L x 37
        'angl': torch.tensor(angl_tns, dtype=torch.float32),  # L x L x 75
        'mfea': torch.tensor(mfea_tns, dtype=torch.float32).permute(1, 0, 2),  # K x L x D_m
        'pfea': torch.tensor(pfea_tns, dtype=torch.float32).permute(2, 0, 1),  # D_p x L x L
    }

    return data_dict


def parse_lbl_file(path, nctc_pos='first', da_bins = [37,25,25,25] ):
    """Parse the NPZ file containing GT-labels for inter-residue distance & orientation."""

    # configurations
    n_bins_cb = da_bins[0]
    n_bins_om = da_bins[1]
    n_bins_th = da_bins[2]
    n_bins_ph = da_bins[3]

    # functions for building classification labels
    def _build_dist_idxs(dist_mat, n_bins, dist_min=2.0, dist_max=20.0):
        bin_wid = (dist_max - dist_min) / (n_bins - 1)
        idxs = np.clip(np.floor((dist_mat - dist_min) / bin_wid).astype(np.int64), 0, n_bins - 1)
        nctc_mat = (idxs == n_bins - 1).astype(np.int8)
        return idxs, nctc_mat

    def _build_dihd_idxs(angl_mat, nctc_mat, n_bins, angl_min=-np.pi, angl_max=np.pi):
        bin_wid = (angl_max - angl_min) / (n_bins - 1)
        idxs = np.clip(np.floor((angl_mat - angl_min) / bin_wid).astype(np.int64), 0, n_bins - 2)
        idxs[nctc_mat == 1] = n_bins - 1
        return idxs

    def _build_plan_idxs(angl_mat, nctc_mat, n_bins, angl_min=-np.pi, angl_max=np.pi):
        bin_wid = (angl_max - angl_min) / (n_bins - 1)
        idxs = np.clip(np.floor((angl_mat - angl_min) / bin_wid).astype(np.int64), 0, n_bins - 2)
        idxs[nctc_mat == 1] = n_bins - 1
        return idxs

    # parse the NPZ file
    with np.load(path) as npz_data:
        # build classification labels (non-contact bin is the last one)
        idxs_cb, nctc_mat = _build_dist_idxs(npz_data['cb-val'], n_bins_cb)
        idxs_om = _build_dihd_idxs(npz_data['om-val'], nctc_mat, n_bins_om)
        idxs_th = _build_dihd_idxs(npz_data['th-val'], nctc_mat, n_bins_th)
        idxs_ph = _build_plan_idxs(npz_data['ph-val'], nctc_mat, n_bins_ph)

        # move the non-contact bin to the first one if needed
        assert nctc_pos in ['first', 'last'], 'unrecognized <nctc_pos>: ' + nctc_pos
        if nctc_pos == 'first':
            idxs_cb = (idxs_cb + 1) % n_bins_cb  # move the non-contact bin to the first one
            idxs_om = (idxs_om + 1) % n_bins_om
            idxs_th = (idxs_th + 1) % n_bins_th
            idxs_ph = (idxs_ph + 1) % n_bins_ph

        # pack into a dict
        data_dict = {
            'cb-idx': torch.tensor(idxs_cb, dtype=torch.int64),  # L x L
            'cb-msk': torch.tensor(npz_data['cb-msk'], dtype=torch.int8),  # L x L
            'om-idx': torch.tensor(idxs_om, dtype=torch.int64),  # L x L
            'om-msk': torch.tensor(npz_data['om-msk'], dtype=torch.int8),  # L x L
            'th-idx': torch.tensor(idxs_th, dtype=torch.int64),  # L x L
            'th-msk': torch.tensor(npz_data['th-msk'], dtype=torch.int8),  # L x L
            'ph-idx': torch.tensor(idxs_ph, dtype=torch.int64),  # L x L
            'ph-msk': torch.tensor(npz_data['ph-msk'], dtype=torch.int8),  # L x L
        }

    return data_dict


class ProtDatasetConfig():
    """Configurations for the <ProtDataset> class."""

    def __init__(
            self,
            pid_fpath=None,     # file path to protein IDs
            hdf_dpath=None,     # directory path to HDF5 files
            a3m_dpath=None,     # directory path to A3M files
            a3m_format='.a3m',
            tpl_dpath=None,     # directory path to TPL files
            tpl_mode='v1',
            npz_dpath=None,     # directory path to NPZ files for intermediate embeddings
            msanpz_dpath=None,
            tplnpz_dpath=None,
            lbl_dpath=None,     # directory path to NPZ files for ground-truth labels
            plddt_fpath=None,   # file path to pkl file for plddt score of pseudo labels
            highres_fpath=None,
            msa_depth=128,      # maximal number of sequences in the MSA
            tpl_topk=4,         # number of top-ranked structural templates
            resd_frmt='ca',     # residue format ('ca' / 'bb' / 'fa')
            n_crops_base=1,     # number of random crops - base elements
            n_crops_se3f=16,    # number of random crops - 3D coordinates for SE(3)-Fold
            crop_size=128,      # size of random crops
            nois_std_max=10.0,  # random noise's maximal standard deviation
            nois_std_min=0.01,  # random noise's minimal standard deviation
            n_nois_levls=61,    # number of random noise's standard deviation levels
            is_train=True,      # whether the training mode is enabled
            da_bins = [37.25,25,25],
        ):
        """Constructor function."""

        # setup configurations
        self.pid_fpath = pid_fpath
        self.hdf_dpath = hdf_dpath
        self.a3m_dpath = a3m_dpath
        self.a3m_format = a3m_format
        self.tpl_dpath = tpl_dpath
        self.npz_dpath = npz_dpath
        self.msanpz_dpath = msanpz_dpath
        self.tplnpz_dpath = tplnpz_dpath
        self.lbl_dpath = lbl_dpath
        self.plddt_fpath = plddt_fpath
        self.highres_fpath = highres_fpath
        self.is_train = is_train
        self.msa_depth = msa_depth
        self.tpl_topk = tpl_topk
        self.resd_frmt = resd_frmt
        self.n_crops_base = n_crops_base
        self.n_crops_se3f = n_crops_se3f
        self.crop_size = crop_size
        self.nois_std_max = nois_std_max
        self.nois_std_min = nois_std_min
        self.n_nois_levls = n_nois_levls
        self.da_bins = da_bins
        self.tpl_mode = tpl_mode
        self.mask_prob = 0.15

        # determine standard deviations of all the random noise levels
        self.nois_stds = None if self.n_nois_levls == 0 else \
            calc_nois_stds(self.nois_std_max, self.nois_std_min, self.n_nois_levls)

        # over-ride the size of random crops if needed
        if not self.is_train:
            self.crop_size = -1

    def show(self):
        """Show detailed configurations."""

        logging.info('=== ProtDatasetConfig - Start ===')
        logging.info('pid_fpath: %s', self.pid_fpath)
        logging.info('hdf_dpath: %s', self.hdf_dpath)
        logging.info('a3m_dpath: %s', self.a3m_dpath)
        logging.info('a3m_format: %s', self.a3m_format)
        logging.info('tpl_dpath: %s', self.tpl_dpath)
        logging.info('npz_dpath: %s', self.npz_dpath)
        logging.info(f'tplnpz_dpath: {self.tplnpz_dpath}')
        logging.info('lbl_dpath: %s', self.lbl_dpath)
        logging.info('msa_depth: %d', self.msa_depth)

        if self.plddt_fpath is not None:
            logging.info('plddt_depth: %d', self.plddt_fpath)

        logging.info('tpl_topk: %d', self.tpl_topk)
        logging.info('resd_frmt: %s', self.resd_frmt)
        logging.info('n_crops_base: %d', self.n_crops_base)
        logging.info('n_crops_se3f: %d', self.n_crops_se3f)
        logging.info('crop_size: %d', self.crop_size)
        logging.info('nois_std_max: %.4f', self.nois_std_max)
        logging.info('nois_std_min: %.4f', self.nois_std_min)
        logging.info('n_nois_levls: %d', self.n_nois_levls)
        logging.info('is_train: %s', self.is_train)
        logging.info('da bins: %s', self.da_bins)
        logging.info('=== ProtDatasetConfig - Finish ===')


class ProtDatasetSemi(Dataset):
    """The protein dataset."""

    def __init__(self, config, semi_config, ratio = 0.75):
        """Constructor function."""

        super().__init__()

        self.ratio = ratio

        if isinstance(config, list):
            all_dataset = [ProtDataset(config_) for config_ in config]
            self.dataset = torch.utils.data.ConcatDataset(all_dataset)
            crop_size = all_dataset[0].config.crop_size
        else:
            self.dataset = ProtDataset(config=config)
            crop_size = self.dataset.config.crop_size

        self.semi_dataset = None
        if self.ratio > 0 and semi_config is not None:
            self.semi_dataset = ProtDataset(config=semi_config, is_semi = True)
            assert crop_size == self.semi_dataset.config.crop_size

        self.num = int(len(self.dataset) / (1 - ratio))

        # TODO: for fairseq training
        self.sizes = np.asarray([crop_size for i in range(self.num)])

    def __len__(self):
        """Get the number of elements in the dataset."""

        return self.num

    def __getitem__(self, idx):

        if self.ratio <= 0 or self.semi_dataset is None:
            return self.dataset.__getitem__(random.randint(0, len(self.dataset)-1))

        if random.uniform(0,1) < self.ratio:
            return self.semi_dataset.__getitem__(random.randint(0, len(self.semi_dataset)-1))
        else:
            return self.dataset.__getitem__(random.randint(0, len(self.dataset)-1))

class ProtDataset(Dataset):
    """The protein dataset."""

    def __init__(self, config, is_semi = False):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.config = config
        self.config.show()
        self.is_semi = is_semi

        # additional configurations
        self.n_atoms_dict = {'ca': 1, 'bb': 3, 'fa': 14}
        self.n_atoms = self.n_atoms_dict[self.config.resd_frmt]  # number of atoms per residue

        # initialize the dataset
        self.__build_i2f_map()
        self.msa_alphabet = Alphabet.from_architecture('MSA Transformer')


    def __len__(self):
        """Get the number of elements in the dataset."""

        return len(self.prot_ids)


    def __getitem__(self, idx):
        """Get the i-th element in the dataset."""

        # get file paths
        prot_id = self.prot_ids[idx]
        hdf_fpath, a3m_fpath, tpl_fpath, npz_fpath, tplnpz_fpath, lbl_fpath, seq_len =\
            self.i2f_map[prot_id]

        # parse files into different data dicts
        data_dict_full = defaultdict(dict)
        hdf_data = parse_hdf_file(hdf_fpath, prot_id)
        data_dict_full['base'] = self.__build_data_dict_base(hdf_data)
        data_dict_full['se3f'] = self.__build_data_dict_se3f(data_dict_full['base']['cord-o'])

        if a3m_fpath is not None:
            msa_dict = {}

            if isinstance(a3m_fpath, list):
                random.shuffle(a3m_fpath)
                for a3m_fpath_ in a3m_fpath:
                    a3m_fpath = a3m_fpath_
                    if os.path.exists(a3m_fpath_):
                        break

            if not os.path.exists(a3m_fpath):
                idx = random.randint(0, self.__len__() - 1)
                print(f'{a3m_fpath} does not exists, return sample {idx}')
                return self.__getitem__(idx)

            if a3m_fpath.endswith('.a3m'):
                try:
                    msa_tokens_true = parse_a3m_file(
                        a3m_fpath, self.msa_alphabet, self.config.msa_depth, self.config.is_train)
                except:
                    idx = random.randint(0, self.__len__() - 1)
                    print(f'sth is wrong in process {prot_id} msa, return sample {idx}')
                    return self.__getitem__(idx)

            else:
                raise NotImplementedError

            '''
                # elif a3m_fpath.endswith('.npz'):
                #     try:
                #         feature_dict = parse_msanpz(a3m_fpath)
                #     except:
                #         idx = random.randint(0, self.__len__() - 1)
                #         print(f'sth is wrong in process {prot_id} msanpz, return sample {idx}')
                #         # os.remove(a3m_fpath)
                #         # print(f'remove {a3m_fpath}')
                #         return self.__getitem__(idx)
                #
                #     # msa
                #     msa_tokens_true = feature_dict['true_msa_esm_tokens'].data.cpu().numpy()
                #
                #     # subsample msa
                #     msa_index = [i + 1 for i in range(msa_tokens_true.shape[0] - 1)]
                #     if self.config.is_train:
                #         random.shuffle(msa_index)
                #     msa_index = np.asarray([0] + msa_index[:self.config.msa_depth - 1])
                #     msa_tokens_true = msa_tokens_true[msa_index, ...]
                #
                #     # extra msa
                #     msa_tokens_extra = feature_dict['extra_msa_esm_tokens'].data.cpu().numpy()
                #     # subsample extra msa & self.config.msa_depth * 2
                #     msa_index_extra = [i + 1 for i in range(msa_tokens_extra.shape[0] - 1)]
                #     if self.config.is_train:
                #         random.shuffle(msa_index_extra)
                #     msa_index_extra = np.asarray([0] + msa_index_extra[:self.config.msa_depth*2 - 1])
                #     msa_tokens_extra = msa_tokens_extra[msa_index_extra, ...]
                #
                #     msa_dict['extra-msa-t'] = torch.tensor(msa_tokens_extra, dtype=torch.int64)  # K x L
            '''

            # apply random masks on MSA tokens
            msa_masks = (np.random.uniform(size=msa_tokens_true.shape) <= self.config.mask_prob).astype(np.int8)
            msa_tokens_pert = np.copy(msa_tokens_true)

            if self.config.is_train:
                msa_tokens_pert[msa_masks == 1] = self.msa_alphabet.mask_idx

            # pack into a dict
            msa_dict['msa-t'] = torch.tensor(msa_tokens_true, dtype=torch.int64)  # K x L
            msa_dict['msa-p'] = torch.tensor(msa_tokens_pert, dtype=torch.int64)  # K x L
            msa_dict['msa-m'] = torch.tensor(msa_masks, dtype=torch.int8)  # K x L
            # K: number of homologous sequences

            data_dict_full['feat'].update(msa_dict)  # MSA tokens

        if True:  # always executed, even if <tpl_fpath> is None

            n_resds = len(hdf_data['seq'])

            tpl_inputs = [tpl_fpath, None] if os.path.exists(tpl_fpath) else None
            if tpl_inputs is None:
                print(f'{tpl_fpath} does not exists')

            try:
                feat_dict = parse_tpl_file(tpl_inputs, n_resds, self.config.tpl_topk, self.config.is_train,
                                           mode=self.config.tpl_mode)
            except:
                idx = random.randint(0, self.__len__() - 1)
                print(f'sth is wrong in process {prot_id} tpl features, return sample {idx}')
                return self.__getitem__(idx)

            data_dict_full['feat'].update(feat_dict)  # structural templates

        if npz_fpath is not None:
            feat_dict = parse_npz_file(npz_fpath)
            data_dict_full['feat'].update(feat_dict)  # inter-residue geometry predictions

        if lbl_fpath is not None:
            data_dict_full['labl'] = parse_lbl_file(lbl_fpath, da_bins=self.config.da_bins)  # ground-truth labels

        # high_resolution
        hr_score = torch.zeros([1, seq_len], dtype=torch.int64)
        if not self.is_semi and \
            self.prot_high_resolution is not None and \
                prot_id.split('_')[0].upper() in self.prot_high_resolution:
            hr_score = torch.ones([1, seq_len], dtype=torch.int64)

        # plddt_score
        plddt_score = torch.ones([1, seq_len])
        if prot_id in self.prot_plddts:
            _, plddt_score = self.prot_plddts[prot_id]
            plddt_score = np.asarray(plddt_score).reshape(1, seq_len)
            plddt_score /= 100

        elif self.is_semi:
            # TODO
            plddt_score = torch.ones([1, seq_len]) * 0.80

        elif self.config.plddt_fpath is not None:
            raise FileNotFoundError

        feat_dict = {'plddt': plddt_score,
                     'highres': hr_score}
        data_dict_full['base'].update(feat_dict)

        # apply random cropping
        data_dict_crop = self.__crop_data_dict(data_dict_full)

        return data_dict_crop

    def __build_i2f_map(self):
        """Build the mapping from protein IDs to HDF5 file paths."""

        # get protein IDs
        assert os.path.exists(self.config.pid_fpath), 'file not found: ' + self.config.pid_fpath
        with open(self.config.pid_fpath, 'r') as i_file:
            prot_ids = {i_line.strip() for i_line in i_file}

        # build the mapping from protein IDs to HDF5 file paths
        self.i2f_map = {}
        for hdf_fname in os.listdir(self.config.hdf_dpath):
            hdf_fpath = os.path.join(self.config.hdf_dpath, hdf_fname)
            logging.info('inspecting the HDF5 file: %s', hdf_fpath)
            with h5py.File(hdf_fpath, 'r', driver='core') as i_file:

                for prot_id in i_file:
                    if prot_id not in prot_ids:
                        continue
                    aa_seq = i_file[prot_id]['seq'][()]#.decode('utf-8')

                    if (not self.config.is_train) and (len(aa_seq) > 512): #512
                        logging.info('skipping <%s> (%d amino-acids)', prot_id, len(aa_seq))
                        continue

                    if isinstance(self.config.a3m_dpath, list):
                        a3m_fpath = []
                        for a3m_ in self.config.a3m_dpath:
                            a3m_fpath.append(os.path.join(a3m_,f'{prot_id}{self.config.a3m_format}'))
                    else:
                        a3m_fpath = None if self.config.a3m_dpath is None else \
                            os.path.join(self.config.a3m_dpath, f'{prot_id}{self.config.a3m_format}')

                    tpl_fpath = None if self.config.tpl_dpath is None else \
                        os.path.join(self.config.tpl_dpath, '%s.pkl' % prot_id)
                    npz_fpath = None if self.config.npz_dpath is None else \
                        os.path.join(self.config.npz_dpath, '%s.npz' % prot_id)
                    # tplnpz_fpath = None if self.config.tplnpz_dpath is None else \
                    #     os.path.join(self.config.tplnpz_dpath, '%s.npz' % prot_id)
                    lbl_fpath = None if self.config.lbl_dpath is None else \
                        os.path.join(self.config.lbl_dpath, '%s.npz' % prot_id)

                    self.i2f_map[prot_id] = (hdf_fpath, a3m_fpath, tpl_fpath, npz_fpath, None, lbl_fpath, len(aa_seq))

        self.prot_plddts = {}
        if self.config.plddt_fpath is not None:
            with open(self.config.plddt_fpath, "rb") as fp:   #Pickling
                self.prot_plddts = pickle.load(fp)

        self.prot_high_resolution = None
        if not self.is_semi and self.config.highres_fpath is not None:
            with open(self.config.highres_fpath, "r") as f:   #Pickling
                self.prot_high_resolution = f.readlines()
                self.prot_high_resolution = set([l.strip().upper() for l in self.prot_high_resolution])

        # obtain a sorted list of protein IDs
        self.prot_ids = sorted(list(self.i2f_map.keys()))

        # TODO for fairseq training
        self.sizes = np.asarray([ self.i2f_map[prot_id][-1] for prot_id in self.prot_ids])
        logging.info('# of protein IDs: %d', len(self.prot_ids))


    def __build_data_dict_base(self, hdf_data):
        """Build the data dict of base elements."""

        # initialization
        n_resds = len(hdf_data['seq'])

        # build ground-truth 3D coordinates & validness masks
        if self.config.resd_frmt in ['ca', 'bb']:
            atom_names = ['CA'] if self.config.resd_frmt == 'ca' else ['N', 'CA', 'C']
            cord_tns_orig = ProtStruct.get_atoms(aa_seq, hdf_data['cord-o'], atom_names)
            cmsk_mat_orig = ProtStruct.get_atoms(aa_seq, hdf_data['cmsk-o'], atom_names)
            cord_tns_reco = ProtStruct.get_atoms(aa_seq, hdf_data['cord-r'], atom_names)
            cmsk_mat_reco = ProtStruct.get_atoms(aa_seq, hdf_data['cmsk-r'], atom_names)
        elif self.config.resd_frmt == 'fa':
            cord_tns_orig, cmsk_mat_orig = hdf_data['cord-o'], hdf_data['cmsk-o']
            cord_tns_reco, cmsk_mat_reco = hdf_data['cord-r'], hdf_data['cmsk-r']
        else:
            raise ValueError('unrecognized residue format: %s', self.config.resd_frmt)

        # pack into a dict
        data_dict = {
            'id': hdf_data['id'],
            'seq': hdf_data['seq'],
            'cord-o': cord_tns_orig.view(1, n_resds, self.n_atoms, 3),  # 1 x L x M x 3
            'cmsk-o': cmsk_mat_orig.view(1, n_resds, self.n_atoms),  # 1 x L x M
            'cord-r': cord_tns_reco.view(1, n_resds, self.n_atoms, 3),  # 1 x L x M x 3
            'cmsk-r': cmsk_mat_reco.view(1, n_resds, self.n_atoms),  # 1 x L x M
            'fram': hdf_data['fram'].view(1, n_resds, 4, 3),  # 1 x L x 4 x 3
            'fmsk': hdf_data['fmsk'].view(1, n_resds),  # 1 x L
            'angl': hdf_data['angl'].view(1, n_resds, 7, 2),  # 1 x L x 7 x 2
            'amsk': hdf_data['amsk'].view(1, n_resds, 7),  # 1 x L x 7
        }

        return data_dict


    def __build_data_dict_se3f(self, cord_tns_true):
        """Build the data dict for the <SE3FoldNet> network."""

        # initialization
        n_smpls = self.config.n_crops_se3f
        n_resds = cord_tns_true.shape[1]

        # perturb ground-truth 3D coordinates
        cord_tns_pert, nstd_vec, grad_tns = None, None, None
        if self.config.nois_stds is not None:
            if self.config.is_train:
                idxs_vec = torch.randint(self.config.n_nois_levls, size=(n_smpls,))
                nstd_vec = torch.tensor(self.config.nois_stds[idxs_vec], dtype=torch.float32)
                nois_tns = nstd_vec.view(-1, 1, 1, 1) * \
                    torch.randn((n_smpls, n_resds, self.n_atoms, 3), dtype=torch.float32)
                cord_tns_pert = cord_tns_true + nois_tns
                grad_tns = -nois_tns / torch.square(nstd_vec.view(-1, 1, 1, 1))
            else:
                cord_tns_pert = 0.1 * self.config.nois_std_max * \
                    torch.randn((n_smpls, n_resds, self.n_atoms, 3), dtype=torch.float32)

        # pack into the dict
        data_dict = {
            'cord': cord_tns_pert,  # None / N x L x M x 3
            'nstd': nstd_vec,  # None / N
            'grad': grad_tns,  # None / N x L x M x 3
        }
        data_dict = {k: v for k, v in data_dict.items() if v is not None}

        return data_dict


    def __crop_data_dict(self, data_dict_src):
        """Apply random cropping on the data dict."""

        # initialization
        n_resds = len(data_dict_src['base']['seq'])
        crop_size = self.config.crop_size if self.config.is_train else n_resds

        # get residue indices' boundaries for random cropping
        if (crop_size == -1) or (n_resds <= crop_size):
            crop_bnds_base = [[0, n_resds] for _ in range(self.config.n_crops_base)]
        else:
            idxs = np.random.randint(n_resds - crop_size + 1, size=(self.config.n_crops_base))
            crop_bnds_base = [[x, x + crop_size] for x in np.nditer(idxs)]
        crop_bnds_se3f = crop_bnds_base * (self.config.n_crops_se3f // self.config.n_crops_base)

        # apply random cropping on base elements, additional features, and ground-truth labels
        data_dict_dst = {}
        data_dict_dst['base'] = self.__crop_data_dict_base(data_dict_src['base'], crop_bnds_base)
        data_dict_dst['se3f'] = self.__crop_data_dict_se3f(data_dict_src['se3f'], crop_bnds_se3f)
        data_dict_dst['feat'] = self.__crop_data_dict_feat(data_dict_src['feat'], crop_bnds_base)
        data_dict_dst['labl'] = self.__crop_data_dict_labl(data_dict_src['labl'], crop_bnds_base)

        return data_dict_dst


    @classmethod
    def __crop_data_dict_base(cls, data_dict_src, crop_bnds):
        """Apply random cropping on base elements."""

        data_dict_dst = {
            'id': data_dict_src['id'],
            'seq': [data_dict_src['seq'][ib:ie] for ib, ie in crop_bnds],
            'cord': torch.stack([data_dict_src['cord-o'][0, ib:ie] for ib, ie in crop_bnds]),
            'cmsk': torch.stack([data_dict_src['cmsk-o'][0, ib:ie] for ib, ie in crop_bnds]),
            'plddt': torch.stack([data_dict_src['plddt'][0, ib:ie] for ib, ie in crop_bnds]),
            'highres': torch.stack([data_dict_src['highres'][0, ib:ie] for ib, ie in crop_bnds]),
            'addi': [{
                'fram': data_dict_src['fram'][0, ib:ie],
                'fmsk': data_dict_src['fmsk'][0, ib:ie],
                'angl': data_dict_src['angl'][0, ib:ie],
                'amsk': data_dict_src['amsk'][0, ib:ie],
                'cord': data_dict_src['cord-r'][0, ib:ie],
                'cmsk': data_dict_src['cmsk-r'][0, ib:ie],
            } for ib, ie in crop_bnds],
        }

        return data_dict_dst


    @classmethod
    def __crop_data_dict_se3f(cls, data_dict_src, crop_bnds):
        """Apply random cropping on 3D coordinates for SE(3)-Fold."""

        data_dict_dst = {}
        if 'cord' in data_dict_src:
            data_dict_dst['cord'] = torch.stack(
                [data_dict_src['cord'][ix, ib:ie] for ix, (ib, ie) in enumerate(crop_bnds)])
        if 'nstd' in data_dict_src:
            data_dict_dst['nstd'] = data_dict_src['nstd']
        if 'grad' in data_dict_src:
            data_dict_dst['grad'] = torch.stack(
                [data_dict_src['grad'][ix, ib:ie] for ix, (ib, ie) in enumerate(crop_bnds)])

        return data_dict_dst


    @classmethod
    def __crop_data_dict_feat(cls, data_dict_src, crop_bnds):
        """Apply random cropping on additional features."""

        data_dict_dst = {}

        # MSA tokens (GT & masked)
        if 'msa-t' in data_dict_src:
            data_dict_dst['msa-t'] = torch.stack(
                [data_dict_src['msa-t'][:, ib:ie] for ib, ie in crop_bnds])
        if 'extra-msa-t' in data_dict_src:
            data_dict_dst['extra-msa-t'] = torch.stack(
                [data_dict_src['extra-msa-t'][:, ib:ie] for ib, ie in crop_bnds])
        if 'msa-p' in data_dict_src:
            data_dict_dst['msa-p'] = torch.stack(
                [data_dict_src['msa-p'][:, ib:ie] for ib, ie in crop_bnds])
        if 'msa-m' in data_dict_src:
            data_dict_dst['msa-m'] = torch.stack(
                [data_dict_src['msa-m'][:, ib:ie] for ib, ie in crop_bnds])

        # structural templates
        if 't1ds' in data_dict_src:
            data_dict_dst['t1ds'] = torch.stack(
                [data_dict_src['t1ds'][:, ib:ie] for ib, ie in crop_bnds])
        if 't2ds' in data_dict_src:
            data_dict_dst['t2ds'] = torch.stack(
                [data_dict_src['t2ds'][:, ib:ie, ib:ie] for ib, ie in crop_bnds])

        # pre-computed intermediate features
        if 'dist' in data_dict_src:
            data_dict_dst['dist'] = torch.stack(
                [data_dict_src['dist'][ib:ie, ib:ie] for ib, ie in crop_bnds])
        if 'angl' in data_dict_src:
            data_dict_dst['angl'] = torch.stack(
                [data_dict_src['angl'][ib:ie, ib:ie] for ib, ie in crop_bnds])
        if 'mfea' in data_dict_src:
            data_dict_dst['mfea'] = torch.stack(
                [data_dict_src['mfea'][:, ib:ie] for ib, ie in crop_bnds])
        if 'pfea' in data_dict_src:
            data_dict_dst['pfea'] = torch.stack(
                [data_dict_src['pfea'][:, ib:ie, ib:ie] for ib, ie in crop_bnds])
        return data_dict_dst


    @classmethod
    def __crop_data_dict_labl(cls, data_dict_src, crop_bnds):
        """Apply random cropping on ground-truth labels."""

        data_dict_dst = {}
        for key, val in data_dict_src.items():
            data_dict_dst[key] = torch.stack([val[ib:ie, ib:ie] for ib, ie in crop_bnds])

        return data_dict_dst

