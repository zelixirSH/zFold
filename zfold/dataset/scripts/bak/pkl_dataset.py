import os
import subprocess
import threading
from pathlib import Path
import numpy as np
import torch
import random
import pickle
import logging

logger = logging.getLogger(__name__)

from zfold.utils.gen_fea import *
from openfold.config import model_config
from openfold.data import feature_pipeline
from zfold.utils.gen_lbl import convert_DisOri_label_v2
from tqdm import tqdm

# Input
#   - t1d: 1D template info (B, T, L, 2)
#   - t2d: 2D template info (B, T, L, L, 10)

def gen_fea(pdb_id, pkl, lbl, feature_processor, bins, mode):

    feature_dict = load_obj(pkl)
    feature_dict['template_all_atom_mask'] = feature_dict['template_all_atom_masks']
    processed_feature_dict = feature_processor.process_features(feature_dict, mode=mode)

    label = np.load(lbl)
    lbl, agnle0, angle1, angle2 = convert_DisOri_label_v2(label, np.ones_like(label), bins=bins)
    lbls = torch.cat([torch.LongTensor(lbl).unsqueeze(0),
                      torch.LongTensor(agnle0).unsqueeze(0),
                      torch.LongTensor(angle1).unsqueeze(0),
                      torch.LongTensor(angle2).unsqueeze(0)], dim=0)

    processed_feature_dict['lbl'] = lbls
    processed_feature_dict = dict(processed_feature_dict)
    return processed_feature_dict

def crop_feature_dict(processed_feature_dict, crop_size, is_train = True):

    seq_len = processed_feature_dict['aatype'].shape[0]

    if seq_len > crop_size and crop_size > 0:

        crop_i = np.random.randint(0, seq_len - crop_size) if is_train else 0

        for key in crop_index_dict.keys():

            if key not in processed_feature_dict:
                continue

            for index in crop_index_dict[key]:
                if index == 0:
                    processed_feature_dict[key] = processed_feature_dict[key][crop_i:crop_i + crop_size, ...]
                elif index == 1:
                    processed_feature_dict[key] = processed_feature_dict[key][:, crop_i:crop_i + crop_size, ...]
                elif index == 2:
                    processed_feature_dict[key] = processed_feature_dict[key][:, :, crop_i:crop_i + crop_size, ...]
                else:
                    raise NotImplementedError

    return processed_feature_dict


class PKLDataset(torch.utils.data.Dataset):
    """
    For loading protein sequence datasets in the common FASTA data format
    """
    def __init__(self, txt, model_name = 'model_1', bins = [37, 25, 25, 25],
                 crop_size = 128,
                 extra_msa_depth = 256,
                 save_dir = None,
                 is_train = False):

        self.tpl_mode = 'v1'
        self.extra_msa_depth = extra_msa_depth
        self.crop_size = crop_size
        self.dataset = []

        if not isinstance(txt, list):
            f = open(txt, 'r')
            lines = f.readlines()
            f.close()
            self.dataset.extend([line.strip().split(' ') for line in lines])
        else:
            for t in txt:
                f = open(t, 'r')
                lines = f.readlines()
                f.close()
                self.dataset.extend([line.strip().split(' ') for line in lines])

        config = model_config(model_name)
        self.feature_processor = feature_pipeline.FeaturePipeline(config.data)
        self.num = len(self.dataset)
        self.bins = bins
        self.sizes = np.asarray([crop_size for i in range(len(self.dataset))])
        print(self.num)
        self.save_dir = save_dir
        self.is_train = is_train

    def __getitem__(self, idx):

        try:
            pdb_id, pkl, lbl = self.dataset[idx]
            npz = f'{self.save_dir}/{pdb_id}.npz'
            if not os.path.exists(npz):
                print(f'{npz} not exists, return sample 0')
                return self.__getitem__(random.randint(0, self.num - 1))

            processed_feature_dict = parse_msanpz(npz)

            seq_len = processed_feature_dict['aatype'].shape[0]

            processed_feature_dict['extra_msa'] = processed_feature_dict['extra_msa'][:self.extra_msa_depth, :, :]
            processed_feature_dict['extra_msa_mask'] = processed_feature_dict['extra_msa_mask'][:self.extra_msa_depth, :, :]
            processed_feature_dict['extra_msa_row_mask'] = processed_feature_dict['extra_msa_row_mask'][:self.extra_msa_depth,:]
            processed_feature_dict['extra_has_deletion'] = processed_feature_dict['extra_has_deletion'][:self.extra_msa_depth,:, :]
            processed_feature_dict['extra_deletion_value'] = processed_feature_dict['extra_deletion_value'][:self.extra_msa_depth, :, :]

            feat_dict = parse_tpl_file(path = [pkl, None],
                                       n_resds = seq_len,
                                       tpl_topk = 4,
                                       is_train = self.is_train,
                                       mode = self.tpl_mode)

            processed_feature_dict.update(feat_dict)  # structural templates

            processed_feature_dict = crop_feature_dict(processed_feature_dict, self.crop_size)

            return processed_feature_dict

        except:
            print('sth is wrong, return sample 0')
            return self.__getitem__(0)

    def __len__(self):
        return self.num

def gen_txt(save_file, pkl_dir, lbl_dir):
    f = open(save_file, 'w')
    for tgt in tqdm(os.listdir(pkl_dir)):
        pkl = f'{pkl_dir}/{tgt}/features.pkl'
        lbl = f'{lbl_dir}/{tgt}.npy'
        if os.path.exists(pkl) and os.path.exists(lbl):
            f.write(f'{tgt} {pkl} {lbl}\n')
    f.close()

def get_crop_index():
    f = open('crop_index','r')
    lines = f.readlines()
    f.close()
    crop_index_dict = {}
    for line in lines:
        tmp = line.strip().split()
        crop_index_dict[tmp[1]] = tmp[2].replace('(','').replace(')','').split(',')
    return crop_index_dict
