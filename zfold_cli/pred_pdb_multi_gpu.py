import argparse
import os
import sys
import torch
import numpy as np

import ray
from ray.actor import ActorHandle

import torch.nn.functional as F
from box import Box
from zfold.utils.progress_bar import *
from zfold.zfoldnet_e2e import XFold
from zfold.utils import save_png, get_features, load_pretrain, exists
from zfold.config import update_config
from zfold.network.af2_smod.prot_struct import ProtStruct

from absl import flags
from absl import app

TEMPL_NUM = 4
max_len = 1024

def gen_npz(data, out_npz, is_save_png, tag = None):

    if tag is not None:
        out_npz = out_npz.replace('.npz', f'-{tag}.npz')

    os.makedirs(os.path.split(out_npz)[0], exist_ok=True)

    if is_save_png:
        try:
            save_png(data['dist'], data['omega'], data['theta'], data['phi'],
                     save_path = out_npz.replace('.npz','.png'))
        except:
            print(f'sth wrong in saving png')

    np.savez_compressed(out_npz, **data)
    print(f'save npz to: {out_npz}\n')

def gen_pdb(aa_seq, cord_tns_pred, mask_mat, pdb_fpath_pred):
    """Calculate evaluation metrics for 3D outputs."""
    for idx_smpl in range(cord_tns_pred.shape[0]):
        ProtStruct.save(aa_seq, cord_tns_pred[idx_smpl], mask_mat, pdb_fpath_pred)
        print(f'save pdb to: {pdb_fpath_pred}\n')

def pred(aa_seq, input, model, is_gpu, is_fp16):
    keys = ['msa_tokens',
            't1ds',
            't2ds',
            'extra_tokens_feats']

    for key in keys:
        if exists(input[key]):
            # print(key, input[key].shape)
            input[key] = input[key].unsqueeze(0)
            if is_gpu:
                input[key] = input[key].cuda()
            if not 'token' in key and is_fp16:
                input[key] = input[key].half()

    output = model.forward(tokens = input['msa_tokens'],
                           extra_tokens_feats = input['extra_tokens_feats'],
                           t1ds = input['t1ds'],
                           t2ds = input['t2ds'],
                           is_3d = True,
                           fape_helper = None,
                           aa_seq = aa_seq,
                           )

    output = output[-1]
    logit_dist, logit_omega, logit_theta, logit_phi = \
                output['cb'], output['om'], output['th'], output['ph']

    # da predictions
    logit_dist = (logit_dist + logit_dist.permute(0, 1, 3, 2)) / 2.0
    logit_omega = (logit_omega + logit_omega.permute(0, 1, 3, 2)) / 2.0
    logit_dist = F.softmax(logit_dist, dim=1).squeeze(dim=0).data.cpu().numpy().transpose((1, 2, 0))
    logit_omega = F.softmax(logit_omega, dim=1).squeeze(dim=0).data.cpu().numpy().transpose((1, 2, 0))
    logit_theta = F.softmax(logit_theta, dim=1).squeeze(dim=0).data.cpu().numpy().transpose((1, 2, 0))
    logit_phi = F.softmax(logit_phi, dim=1).squeeze(dim=0).data.cpu().numpy()
    logit_phi = np.concatenate([logit_phi[:1, :, :], logit_phi[13:, :, :]], axis=0).transpose((1, 2, 0))

    output_da = {}
    output_da['dist'] = logit_dist
    output_da['omega'] = logit_omega
    output_da['theta'] = logit_theta
    output_da['phi'] = logit_phi

    output_3d = {}
    output_3d['cords'] = output['cords'][-1].unsqueeze(0).cpu()
    output_3d['cmask'] = output['cmask'][-1].cpu()

    return output_da, output_3d

@ray.remote(num_gpus=1)
@torch.no_grad()
def infer(split_idx,
          tgt_list,
          config_yaml,
          weight_path,
          actor: ActorHandle,
          is_gpu = False,
          is_extra_fea = False,
          n_recycle = None,
          msa_depth = 384,
          is_save_png = True,
          is_fair = False,
          is_fp16 = True,
          ):

    num_tgts = len(tgt_list)
    print(f'Xfold prediction starts from here')
    print('Number of tgts on split{}: {}.'.format(split_idx, num_tgts))

    if len(tgt_list) == 0:
        return

    print('    model yaml', config_yaml)
    MODEL_PARAM = Box.from_yaml(filename=config_yaml)
    print(MODEL_PARAM)
    # update missing param
    MODEL_PARAM = update_config(MODEL_PARAM)
    # skip_load_msa_bert
    MODEL_PARAM['msa_bert']['skip_load_msa_bert'] = True
    # TODO
    MODEL_PARAM['msa_bert']['msa_bert_config']['model_yaml'] = '/data1/protein/facebook_esm_checkpoints/msa_trans_official.yaml'
    MODEL_PARAM['msa_bert']['msa_bert_config']['model_weight'] = '/data1/protein/facebook_esm_checkpoints/msa_trans_official.pt'

    if exists(n_recycle):
        MODEL_PARAM['n_recycle'] = n_recycle
        print(f'    set n_recycle to {n_recycle}')

    if exists(n_recycle):
        MODEL_PARAM['n_recycle'] = n_recycle
        print(f'    set n_recycle to {n_recycle}')

    MODEL_PARAM.basic.msa_depth_max = msa_depth
    model = XFold(MODEL_PARAM)
    print(model)
    model = load_pretrain(model, weight_path, is_fair=is_fair, is_exit=True)
    model.eval()

    if is_gpu:
        print('--> inference using GPU')
        model = model.cuda()
    else:
        print('--> inference with cpu mode')

    if is_fp16:
        print('--> inference using fp16')
        model.half()

    for msa_, tpl_, out_, out_pdb_ in tgt_list:

        try:
            torch.cuda.empty_cache()

            if not os.path.exists(msa_):
                print(f'{msa_} does not exist')
                actor.update.remote(1)
                continue

            with open(msa_, 'r') as f:
                lines = f.readlines()
                aa_seq = lines[1].strip()
                prot_id = lines[0].strip()[1:]
                n_resds = len(aa_seq)

            if max_len > 0 and n_resds > max_len:
                actor.update.remote(1)
                continue

            tpl_inputs = [tpl_, None] if os.path.exists(tpl_) else None

            input = get_features(msa_,
                                 tpl_inputs,
                                 msa_depth = msa_depth * 4,
                                 tpl_topk = TEMPL_NUM,
                                 is_train = False,
                                 tpl_mode = MODEL_PARAM['tpl_mode'])

            input['extra_tokens_feats'] = None

            os.makedirs(out_pdb_, exist_ok=True)
            pdb_fpath_pred = os.path.join(out_pdb_, f'{prot_id}.pdb')
            if not os.path.exists(pdb_fpath_pred):
                # t1ds & t2ds are template related features
                output_da, output_3d = pred(aa_seq, input, model, is_gpu, is_fp16)
                # generate trrosetta npz
                gen_npz(output_da, out_, is_save_png)
                # generate pdb files
                gen_pdb(aa_seq, output_3d['cords'], output_3d['cmask'], pdb_fpath_pred = pdb_fpath_pred)

        except:
            print('sth is wrong')

        actor.update.remote(1)


def inference(tgtlist,
              config_yaml,
              weight_path,
              is_gpu = True,
              is_extra_fea = False,
              n_recycle = None,
              msa_depth = 384,
              is_save_png = True,
              is_fair = False,
              is_fp16 = False,
              gpu_id = '0,1,2,3,4,5,6,7',
              num_cpus = 16):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    num_gpus = len(gpu_id.split(','))

    num_tgts = len(tgtlist)
    print('Number of tgts: {}.'.format(num_tgts))

    if num_tgts == 0:
        return

    tgt_lists = [tgtlist[i::num_gpus] for i in range(num_gpus)]
    tgt_lists = [tgtlist for tgtlist in tgt_lists if len(tgtlist) > 0]

    ray.shutdown()

    if num_cpus != 0:
        ray.init(num_cpus=len(tgt_lists))
    else:
        ray.init()

    pb = ProgressBar(len(tgtlist))
    actor = pb.actor
    print('Number of GPUs: {}.'.format(num_gpus))

    infer_list = []
    for i in range(len(tgt_lists)):
        infer_list.append(infer.remote(i, tgt_lists[i],
                                         config_yaml,
                                         weight_path,
                                         actor,
                                         is_gpu,
                                         is_extra_fea,
                                         n_recycle,
                                         msa_depth,
                                         is_save_png,
                                         is_fair,
                                         is_fp16))
    pb.print_until_done()
    ray.get(infer_list)
    ray.get(actor.get_counter.remote())

# Params for pred_pdb
flags.DEFINE_list('msa_paths', None, 'Paths to MSA files')
flags.DEFINE_list('tpl_paths', None, 'Paths to tpl files')
flags.DEFINE_list('save_npzs', None, 'Path to')
flags.DEFINE_list('save_pdb_dirs', None, 'Path to')
flags.DEFINE_string('config_yaml', None, 'Path to')
flags.DEFINE_string('weight_path', None, 'Path to')

flags.DEFINE_boolean('is_gpu', False, '')
flags.DEFINE_boolean('is_fp16', False, '')
flags.DEFINE_boolean('is_fair', False, '')

flags.DEFINE_string('gpu_id', '0,1,2,3,4,5,6,7', ' when gpu_id is None, cpu is used for inference')
flags.DEFINE_integer('msa_depth', 384, '')

FLAGS = flags.FLAGS

def pred_pdb(argv):
    tgtlist = zip(FLAGS.msa_paths, FLAGS.tpl_paths, FLAGS.save_npzs, FLAGS.save_pdb_dirs)
    tgtlist = list(tgtlist)
    inference(tgtlist,
              config_yaml = FLAGS.config_yaml,
              weight_path = FLAGS.weight_path,
              msa_depth = FLAGS.msa_depth,
              is_fair = FLAGS.is_fair,
              is_fp16 = FLAGS.is_fp16,
              is_gpu = FLAGS.is_gpu,
              gpu_id = FLAGS.gpu_id)

def main():
    flags.mark_flags_as_required([
        'msa_paths',
        'tpl_paths',
        'save_npzs',
        'save_pdb_dirs',
        'config_yaml',
        'weight_path'
    ])
    app.run(pred_pdb)

if __name__ == '__main__':
    main()
