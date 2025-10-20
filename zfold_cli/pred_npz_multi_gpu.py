import argparse
import os
import sys
import torch
import numpy as np
import ray
from ray.actor import ActorHandle
import torch.nn.functional as F
from zfold.zfoldnet import XFold2D
from zfold.utils import save_png, get_features, load_pretrain, exists
from zfold.config import update_config
from box import Box
from zfold.utils.progress_bar import *

from absl import flags
from absl import app

# params for msa & template features
TEMPL_NUM = 4

def gen_npz(data, out_npz, is_save_png):
    if is_save_png:
        save_png(data['dist'], save_path=out_npz.replace('.npz','.png'))
    os.makedirs(os.path.split(out_npz)[0], exist_ok=True)
    np.savez_compressed(out_npz, **data)

@ray.remote(num_gpus=1)
@torch.no_grad()
def infer(split_idx,
         tgt_list,
         config_yaml,
         weight_path,
         actor: ActorHandle,
         is_gpu=False,
         n_recycle = None,
         msa_depth = 384,
         is_save_png = True,
         is_fair = False,
         is_fp16 = True,
          ):

    num_tgts = len(tgt_list)
    print(f'Xfold prediction starts from here')
    print('Number of tgts on split{}: {}.'.format(split_idx, num_tgts))
    print('    model yaml', config_yaml)

    MODEL_PARAM = Box.from_yaml(filename=config_yaml)

    MODEL_PARAM = update_config(MODEL_PARAM)

    MODEL_PARAM['msa_bert']['skip_load_msa_bert'] = True
    # TODO
    MODEL_PARAM['msa_bert']['msa_bert_config']['model_yaml'] = '/data1/protein/facebook_esm_checkpoints/msa_trans_official.yaml'
    MODEL_PARAM['msa_bert']['msa_bert_config']['model_weight'] = '/data1/protein/facebook_esm_checkpoints/msa_trans_official.pt'

    tpl_mode = MODEL_PARAM['templ']['tpl_mode']

    if exists(n_recycle):
        MODEL_PARAM['n_recycle'] = n_recycle
        print(f'    set n_recycle to {n_recycle}')

    if exists(n_recycle):
        MODEL_PARAM['n_recycle'] = n_recycle
        print(f'    set n_recycle to {n_recycle}')

    model = XFold2D(MODEL_PARAM)
    model = load_pretrain(model, weight_path, is_fair=is_fair, is_exit=True)
    model.eval()

    if is_gpu:
        print('--> inference using GPU')
        model = model.cuda()
    else:
        print('--> inference with cpu mode')

    if is_fp16:
        print('--> inference using fp16')
        model = model.half()

    def __pred(input, is_gpu, is_fp16):
        torch.cuda.empty_cache()

        keys = ['msa_tokens', 't1ds','t2ds', 'tokens_feats', 'extra_tokens_feats']
        for key in keys:
            if exists(input[key]):
                input[key] = input[key].unsqueeze(0)
                print(f'--> input {key} shape:', input[key].shape)
                if is_gpu:
                    input[key] = input[key].cuda()

        keys = ['t1ds','t2ds']
        for key in keys:
            if exists(input[key]):
                if is_fp16:
                    input[key] = input[key].half()

        output = model.forward(tokens = input['msa_tokens'],
                               t1ds = input['t1ds'],
                               t2ds = input['t2ds'],
                               tokens_feats = input['tokens_feats'],
                               extra_tokens_feats = input['extra_tokens_feats'],
                               is_fea = True)

        logit_dist, logit_omega, logit_theta, logit_phi = output['dist'], output['omega'], output['theta'], output['phi']
        logit_dist = (logit_dist + logit_dist.permute(0, 1, 3, 2)) / 2.0
        logit_omega = (logit_omega + logit_omega.permute(0, 1, 3, 2)) / 2.0
        logit_dist = F.softmax(logit_dist, dim=1).squeeze(dim=0).data.cpu().numpy().transpose((1, 2, 0))
        logit_omega = F.softmax(logit_omega, dim=1).squeeze(dim=0).data.cpu().numpy().transpose((1, 2, 0))
        logit_theta = F.softmax(logit_theta, dim=1).squeeze(dim=0).data.cpu().numpy().transpose((1, 2, 0))
        logit_phi = F.softmax(logit_phi, dim=1).squeeze(dim=0).data.cpu().numpy()
        logit_phi = np.concatenate([logit_phi[:1, :, :], logit_phi[13:, :, :]], axis=0).transpose((1, 2, 0))

        data = {}
        data['dist'] = logit_dist.astype(np.float16)
        data['omega'] = logit_omega.astype(np.float16)
        data['theta'] = logit_theta.astype(np.float16)
        data['phi'] = logit_phi.astype(np.float16)

        return data

    for msa_, tpl_, out_ in tgt_list:

        torch.cuda.empty_cache()

        tpl_input = [tpl_, None] if os.path.exists(tpl_) else None

        if not os.path.exists(out_):
            try:
                input = get_features(msa_,
                                     tpl_input,
                                     msa_depth = msa_depth,
                                     tpl_topk = TEMPL_NUM,
                                     is_train = False,
                                     tpl_mode = tpl_mode)
                gen_npz(__pred(input, is_gpu, is_fp16), out_, is_save_png)
            except:
                print('sth is wrong')

        actor.update.remote(1)

def inference(tgtlist,
              config_yaml,
              weight_path,
              is_gpu = True,
              n_recycle = None,
              msa_depth = 384,
              is_save_png = True,
              is_fair = False,
              is_fp16 = True,
              gpu_id = '0,1,2,3,4,5,6,7',
              num_cpus = 16):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    num_gpus = len(gpu_id.split(','))
    print(num_gpus)
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
flags.DEFINE_string('config_yaml', None, 'Path to')
flags.DEFINE_string('weight_path', None, 'Path to')

flags.DEFINE_boolean('is_gpu', False, '')
flags.DEFINE_boolean('is_fp16', False, '')
flags.DEFINE_boolean('is_fair', False, '')

flags.DEFINE_string('gpu_id', '0,1,2,3,4,5,6,7', ' when gpu_id is None, cpu is used for inference')
flags.DEFINE_integer('msa_depth', 384, '')

FLAGS = flags.FLAGS

def pred_npz(argv):
    tgtlist = zip(FLAGS.msa_paths, FLAGS.tpl_paths, FLAGS.save_npzs)
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
        'config_yaml',
        'weight_path'
    ])
    app.run(pred_npz)

if __name__ == '__main__':
    main()
