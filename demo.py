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

def main():
    # setup configurations
    esm_root = './facebook_esm_checkpoints'
    root = './384_256_lm4_lp4_md128_mp0.15_gr1_bs64_pld0.3-MSATrans'
    data_yaml = './configs/data.yaml'
    config_yaml = f'{root}/model.yaml'
    weight_path = f'{root}/checkpoint_slim.pt'

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
        wc_clsh=1.0,   # weighting coefficient for structural violation loss
    )

    MODEL_PARAM = Box.from_yaml(filename=config_yaml)
    MODEL_PARAM = update_config(MODEL_PARAM)
    MODEL_PARAM.msa_bert.msa_bert_config.model_yaml = f'{esm_root}/msa_trans_official.yaml'
    MODEL_PARAM.msa_bert.msa_bert_config.model_weight = f'{esm_root}/msa_trans_official.pt'

    model = XFold(MODEL_PARAM)
    model = load_pretrain(model, weight_path, is_fair = False)
    model = model.cuda() if is_cuda else model

    for i in range(len(dataset)):
        inputs = send_to_device(dataset[i], device='cuda:0') if is_cuda else dataset[i]
        # build input features from MSA & structral templates
        aa_seq = inputs['base']['seq'][0]
        msa_tokens = inputs['feat']['msa-t'] # use unmasked msa and sample msa on-the-fly
        t1ds_tns = inputs['feat']['t1ds']
        t2ds_tns = inputs['feat']['t2ds']
        # forward pass
        output = model(msa_tokens, t1ds_tns, t2ds_tns, is_3d = True, aa_seq = aa_seq)
        output = output[-1]
        loss, metrics_af2 = loss_helper.calc_loss(inputs, output)
        print(inputs['base']['id'], loss, metrics_af2)

if __name__ == '__main__':
    main()
