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
import ml_collections
from absl import flags
from absl import app

TEMPL_NUM = 4

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

    new_data = { n:data[n] for n in FLAGS.save_fea_names}
    np.savez_compressed(out_npz, **new_data)
    for key in new_data:
        print(key, new_data[key].shape)
    print(f'save npz to: {out_npz}\n')

def gen_pdb(aa_seq, cord_tns_pred, mask_mat, pdb_fpath_pred, glddt=None, plddt=None):
    """Calculate evaluation metrics for 3D outputs."""
    for idx_smpl in range(cord_tns_pred.shape[0]):
        ProtStruct.save(aa_seq, cord_tns_pred[idx_smpl], mask_mat, pdb_fpath_pred, glddt, plddt)
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
                           is_fea = True,
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

    output_da['msa_fea'] = output['msa_fea'].squeeze(0).data.cpu().numpy()
    output_da['pair_fea'] = output['pair_fea'].squeeze(0).data.cpu().numpy()
    output_da['single_fea'] = output['msa_fea'][0,0,:,:].squeeze(0).squeeze(0).data.cpu().numpy()

    # output lddt per-residue
    # shape (L, 50)
    #print("plDDT data origin", output['plddt'][-1])
    #print("plDDT data origin", output['plddt'].shape)
    output_da['plddt'] = output['plddt'][-1][-1].cpu().view(-1).numpy()
    #print("plDDT data", output_da['plddt'])
    
    output_3d = {}
    output_3d['cords'] = output['cords'][-1].unsqueeze(0).cpu()
    output_3d['cmask'] = output['cmask'][-1].cpu()

    return output_da, output_3d


@torch.no_grad()
def infer(tgt_list,
          config_yaml,
          weight_path,
          is_gpu = False,
          is_extra_fea = False,
          n_recycle = None,
          msa_depth = 384,
          is_save_png = True,
          is_fair = False,
          is_fp16 = True,
          max_len = 1024,
          ):

    num_tgts = len(tgt_list)
    print(f'Xfold prediction starts from here')
    print('Number of tgts: {}.'.format(num_tgts))

    if len(tgt_list) == 0:
        return

    print('    model yaml', config_yaml)
    MODEL_PARAM = Box.from_yaml(filename=config_yaml)
    # update missing param
    MODEL_PARAM = update_config(MODEL_PARAM)
    # skip_load_msa_bert
    MODEL_PARAM['msa_bert']['skip_load_msa_bert'] = True
    MODEL_PARAM['msa_bert']['msa_bert_config']['model_yaml'] = f'{os.path.split(config_yaml)[0]}/msa_trans_official.yaml'
    MODEL_PARAM['msa_bert']['msa_bert_config']['model_weight'] = f'{os.path.split(config_yaml)[0]}/msa_trans_official.pt'

    print(ml_collections.ConfigDict(MODEL_PARAM))

    if exists(n_recycle):
        MODEL_PARAM['n_recycle'] = n_recycle
        print(f'    set n_recycle to {n_recycle}')

    if exists(n_recycle):
        MODEL_PARAM['n_recycle'] = n_recycle
        print(f'    set n_recycle to {n_recycle}')

    MODEL_PARAM.basic.msa_depth_max = msa_depth
    model = XFold(MODEL_PARAM)
    model = load_pretrain(model, weight_path, is_fair=is_fair)
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
        torch.cuda.empty_cache()

        if not os.path.exists(msa_):
            print(f'{msa_} does not exist')
            continue

        assert msa_.endswith('.a3m'), print('msa input must be .a3m format')

        with open(msa_, 'r') as f:
            lines = f.readlines()
            aa_seq = lines[1].strip()
            prot_id = lines[0].strip()[1:]
            n_resds = len(aa_seq)

        if max_len > 0 and n_resds > max_len:
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

        # save plddt
        plddt_fpath = os.path.join(out_pdb_, f'{prot_id}_plddt.dat')

        if not os.path.exists(pdb_fpath_pred):
            # t1ds & t2ds are template related features
            output_da, output_3d = pred(aa_seq, input, model, is_gpu, is_fp16)
            # generate trrosetta npz
            gen_npz(output_da, out_, is_save_png)

            # save plddt
            global_plddt = output_da['plddt'].mean()
            with open(plddt_fpath, 'w') as tf:
                tf.write(f'REMARK Global plDDT = {global_plddt:.3f}\n')
                for idx in range(output_da['plddt'].shape[0]):
                    tf.write(f"{idx+1:6d} {output_da['plddt'][idx]:.6f}\n")
            print("--> Global plDDT ", global_plddt)
            
            # generate pdb files
            gen_pdb(aa_seq, output_3d['cords'], output_3d['cmask'], pdb_fpath_pred = pdb_fpath_pred, 
                    glddt=global_plddt, plddt=output_da['plddt'])

        else:
            print(f'{pdb_fpath_pred} exists, skip inference')

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
              num_cpus = 16,
              max_len = 1024,
              ):


    num_tgts = len(tgtlist)
    print('Number of tgts: {}.'.format(num_tgts))

    if num_tgts == 0:
        return

    infer(tgtlist,
             config_yaml,
             weight_path,
             is_gpu,
             is_extra_fea,
             n_recycle,
             msa_depth,
             is_save_png,
             is_fair,
             is_fp16,
             max_len=max_len)

# Params for pred_pdb
flags.DEFINE_list('msa_paths', None, 'Paths to MSA files')
flags.DEFINE_list('tpl_paths', None, 'Paths to tpl files')
flags.DEFINE_list('save_npzs', None, 'Path to')
flags.DEFINE_list('save_pdb_dirs', None, 'Path to')
flags.DEFINE_string('config_yaml', None, 'Path to')
flags.DEFINE_string('weight_path', None, 'Path to')

flags.DEFINE_list('save_fea_names', 'dist,omega,theta,phi,single_fea', 'Path to')

flags.DEFINE_boolean('is_gpu', False, '')
flags.DEFINE_boolean('is_fp16', False, '')
flags.DEFINE_boolean('is_fair', False, '')

flags.DEFINE_string('gpu_id', '0,1,2,3,4,5,6,7', ' when gpu_id is None, cpu is used for inference')
flags.DEFINE_integer('msa_depth', 384, '')
flags.DEFINE_integer('max_len', 512, '')

FLAGS = flags.FLAGS

def pred_pdb(argv):
    tgtlist = zip(FLAGS.msa_paths, FLAGS.tpl_paths, FLAGS.save_npzs, FLAGS.save_pdb_dirs)
    tgtlist = list(tgtlist)
    inference(tgtlist,
              config_yaml=FLAGS.config_yaml,
              weight_path=FLAGS.weight_path,
              msa_depth=FLAGS.msa_depth,
              is_fair=FLAGS.is_fair,
              is_fp16=FLAGS.is_fp16,
              is_gpu=FLAGS.is_gpu,
              max_len=FLAGS.max_len,)

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


