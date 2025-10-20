import os
import string
import random
import logging
import itertools
from typing import List, Tuple
from collections import defaultdict

from box import Box
from multiprocessing import Manager, Pool
from zfold.loss.losses import calc_loss_da
from zfold.loss.losses import calc_loss_lm
from zfold.loss.losses import calc_loss_fape
from zfold.dataset.dbutils import send_to_device
from zfold.dataset.dbutils import setup_path_dict_no_root
from zfold.dataset.tools.pdb_evaluator import PdbEvaluator
from zfold.network.af2_smod.prot_struct import *
from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.dataset.prot_dataset import *

if __name__ == '__main__':
    data_yaml = '/home/seutao/Projects/CODE/Z-Fold/configs/data.yaml'
    data_config = Box.from_yaml(filename=data_yaml)

    name, subset, is_train = 'trrosetta', None, True
    # name, subset, is_train = 'pdb28k', 'trn', True
    # name, subset, is_train = 'casp14', None, False
    path_dict = setup_path_dict_no_root(data_config['data'][name], name, subset)
    ds_config = ProtDatasetConfig(**path_dict, **data_config['data']['base'], is_train=is_train)
    print(ds_config)

    ds_config_semi = None

    # name, subset, is_train = 'semi', '0', True
    # path_dict = setup_path_dict_no_root(data_config['data'][name], name, subset)
    # ds_config_semi = ProtDatasetConfig(**path_dict, **data_config['data']['base'], is_train=is_train)
    # print(ds_config)

    dataset = ProtDatasetSemi(ds_config, ds_config_semi, ratio=0.75)
    print(len(dataset))

    for e in range(1):
        for i in range(len(dataset)):
            print(i)
            inputs = send_to_device(dataset[i], device='cuda:0')
            # build input features from MSA & structral templates
            msa_tokens = inputs['feat']['msa-p']
            t1ds_tns = inputs['feat']['t1ds']
            t2ds_tns = inputs['feat']['t2ds']

            print(msa_tokens.shape,
                  t1ds_tns.shape,
                  t2ds_tns.shape)


