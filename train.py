import os
import torch
import logging
from box import Box
from zfold.dataset.utils import zfold_init
from zfold.zfoldnet_e2e import XFold
from zfold.utils import load_pretrain
from zfold.config import update_config
from zfold.dataset.prot_dataset import ProtDataset
from zfold.dataset.prot_dataset import ProtDatasetConfig
from zfold.loss.losses import calc_loss_da, calc_loss_lm
from zfold.loss.loss_helper import LossHelper
from zfold.dataset.dbutils import send_to_device
from zfold.dataset.dbutils import setup_path_dict_no_root

import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, step_size, iter_max, warmup_iter, power=1.0, last_epoch=-1):
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        self.warmup_iter = warmup_iter

        if self.warmup_iter > 0:
            self.warmup_factor = 1.0 / self.warmup_iter
        else:
            self.warmup_factor = 1

        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):

        if self.warmup_iter > 0 and self.last_epoch <= self.warmup_iter:
            self.warmup_factor = self.last_epoch / float(self.warmup_iter)
            return self.warmup_factor * lr
        else:
            return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]


def main():
    # setup configurations
    esm_root = './facebook_esm_checkpoints'
    root = './m1-384_256_lm4_lp4_md128_mp0.15_gr1_bs128_pld0.3-MSATrans'

    data_yaml = './configs/data.yaml'
    config_yaml = f'{root}/model.yaml'

    zfold_init()

    is_cuda = True
    data_config = Box.from_yaml(filename=data_yaml)

    name, subset, is_train = 'trrosetta', None, True
    path_dict = setup_path_dict_no_root(data_config['data'][name], name, subset)
    ds_config = ProtDatasetConfig(**path_dict, **data_config['data']['base'], is_train=is_train)
    dataset = ProtDataset(ds_config)

    loss_helper = LossHelper(
        wc_da=1.0,     # weighting coefficient for inter-residue DA predictions
        wc_lm=1.0,     # weighting coefficient for masked MSA predictions
        wc_fape=1.0,   # weighting coefficient for frame aligned point error (FAPE)
        wc_lddt=0.1,   # weighting coefficient for per-residue lDDT-Ca predictions
        wc_qnrm=0.02,  # weighting coefficient for L2-norm loss on quaternion vectors
        wc_anrm=0.02,  # weighting coefficient for L2-norm loss on torsion angle metrices
        wc_clsh=0.0,   # weighting coefficient for structural violation loss
    )

    MODEL_PARAM = Box.from_yaml(filename=config_yaml)
    MODEL_PARAM = update_config(MODEL_PARAM)
    MODEL_PARAM.msa_bert.msa_bert_config.model_yaml = f'{esm_root}/msa_trans_official.yaml'
    MODEL_PARAM.msa_bert.msa_bert_config.model_weight = f'{esm_root}/msa_trans_official.pt'

    model = XFold(MODEL_PARAM)
    model = model.cuda() if is_cuda else model

    n_iters = 125 * 1000
    warmup_iter = 500
    bs = 128

    # optimize trainable parameters to minimize the FAPE loss
    optimizer = Adam(list(model.parameters()), lr=0.0003)
    scheduler = PolynomialLR(optimizer, step_size=1, iter_max=n_iters, warmup_iter=warmup_iter)

    for idx_iter in range(n_iters):

        loss = 0.0
        for j in range(bs):
            idx_data = np.random.randint(0, len(dataset)-1)
            inputs = send_to_device(dataset[idx_data], device='cuda:0') if is_cuda else dataset[idx_data]
            # build input features from MSA & structral templates
            aa_seq = inputs['base']['seq'][0]
            msa_tokens = inputs['feat']['msa-t'] # use unmasked msa and sample msa on-the-fly
            t1ds_tns = inputs['feat']['t1ds']
            t2ds_tns = inputs['feat']['t2ds']
            # forward pass
            output = model(msa_tokens, t1ds_tns, t2ds_tns, is_3d = True, aa_seq = aa_seq)
            output = output[-1]
            loss, metrics_af2 = loss_helper.calc_loss(inputs, output)
            loss += loss / bs

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        logging.info('zFold loss @ %d-th iteration: lr-%.7f loss-%.4f', idx_iter + 1, scheduler.get_lr()[0], loss.item())

if __name__ == '__main__':
    # hyperparams for training m1 model
    # optimizer adam
    # lr0.0003 + polynomial_decay
    # warmup 500 steps
    # bs128
    # 125k iters
    main()
