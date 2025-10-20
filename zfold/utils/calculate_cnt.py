#!/usr/bin/env python
import fire
import sys
import os
import numpy as np
import math

## both pred and truth shall be a 2D matrix
def TopAccuracy(pred=None, truth=None, ratio=[1, 0.5, 0.2, 0.1]):
    if pred is None:
        print ('please provide a predicted contact matrix')
        sys.exit(-1)

    if truth is None:
        print ('please provide a true contact matrix')
        sys.exit(-1)

    assert pred.shape[0] == pred.shape[1]
    assert pred.shape == truth.shape, print('pred and groundtruth',pred.shape, truth.shape)

    pred_truth = np.dstack((pred, truth))

    M1s = np.ones_like(truth, dtype = np.int8)
    mask_LR = np.triu(M1s, 24)
    mask_MLR = np.triu(M1s, 12)
    mask_SMLR = np.triu(M1s, 6)

    mask_MR = mask_MLR - mask_LR
    mask_SR = mask_SMLR - mask_MLR
    seqLen = pred.shape[0]

    accs = []
    for mask in [mask_LR, mask_MR, mask_MLR, mask_SR]:
        mask[truth==-1] = 0
        res = pred_truth[mask.nonzero()]
        res_sorted = res[(-res[:,0]).argsort()]

        for r in ratio:
            numTops = int(seqLen * r)
            numTops = min(numTops, res_sorted.shape[0])
            topLabels = res_sorted[:numTops, 1]
            numCorrects = ( (0 < topLabels) & (topLabels < 8) ).sum()
            accuracy = numCorrects * 1./ (numTops + 0.00001)
            accs.append(accuracy)

    return np.array(accs)

def TopAccuracy_v2(pred, truth):
    """"Run the assessor for inter-residue contact predictions w/ GT 3D coordinates.

    Args:
    * truth: contact masks for CB-CB atom pairs (CA for Glycine)
    * pred: predicted inter-residue contact probabilities

    Returns:
    * prec: top-L precision for long-range contact predictions
    """

    # initialization
    n_resds = truth.shape[0]

    # find-out top-L predicted inter-residue contacts
    cntc_infos = []
    for ir in range(n_resds):
        for ic in range(ir + 24, n_resds):
            if truth[ir, ic] != -1:
                cntc_infos.append((ir, ic, pred[ir, ic]))
    cntc_infos.sort(key=lambda x: x[2], reverse=True)

    # count the number of correct predictions
    n_pairs_true = 0
    n_pairs_full = min(n_resds, len(cntc_infos))
    for idx in range(n_pairs_full):
        ir, ic, _ = cntc_infos[idx]
        if truth[ir, ic] == 1:
            n_pairs_true += 1

    # calculcate the top-L precision for long-range contact predictions
    prec = n_pairs_true / (n_pairs_full + 0.00001)

    return prec

def LoadContactMatrix(file=None):
    if file is None:
        print ('please provide a contact matrix file')
        sys.exit(-1)

    if not os.path.isfile(file):
        print ('please provide a valid contact matrix file')
        sys.exit(-1)

    content = np.genfromtxt(file, dtype=np.float32)
    """
    fh = open(file, 'r')
    content = []
    for line in list(fh):
	row = [ np.float32(x) for x in line.strip().split() ]
	content.append(row)
    fh.close()
    return np.array(content)
    """
    return content

def get_contact_precision_distcb(npz_list, label):

    if isinstance(npz_list, list):
        dist_list = []
        for npz in npz_list:
            dist = np.load(npz)['dist']
            dist_list.append(dist)
        dist = np.mean(dist_list, 0)
    else:
        dist = npz_list

    pos = np.sum(dist[:, :, 1:13], axis=2)
    label_contact = LoadContactMatrix(label)
    accs = TopAccuracy(pos, label_contact, ratio=[1, 0.5, 0.2, 0.1])

    accsStr = [str(a) for a in accs]
    resultStr = 'xxxx' + ' ' + str(label_contact.shape[0]) + ' '
    resultStr += (' '.join(accsStr))
    return resultStr, dist

def softmax_np(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    re = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return re

def get_contact_precision__(pos, label, CB_index=1, pdb_id = 'T1xx'):
    mask = (label[:, :, CB_index] == -1)
    label_dis = label[:, :, CB_index]
    label_contact = np.zeros(label_dis.shape)
    label_contact[label_dis < 8.0] = 1
    label_contact[mask] = -1

    # accs = TopAccuracy(pos, label_contact, ratio=[1, 0.5, 0.2, 0.1])
    # accsStr = [str(a)[:6] for a in accs]
    # resultStr =  f'{pdb_id} {str(label.shape[0])} '
    # resultStr += (' '.join(accsStr))
    # return resultStr, accs

    accs = TopAccuracy_v2(pos, label_contact)
    resultStr = pdb_id
    return resultStr, [accs]

def get_contact_precision_(dist, label, CB_index=1, pdb_id = 'T1xx'):
    #non_contact_bin = 0
    pos = np.sum(dist[:, :, 1:13], axis=2)

    mask = (label[:, :, CB_index] == -1)
    label_dis = label[:, :, CB_index]
    label_contact = np.zeros(label_dis.shape)
    label_contact[label_dis < 8.0] = 1
    label_contact[mask] = -1

    accs = TopAccuracy(pos, label_contact, ratio=[1, 0.5, 0.2, 0.1])
    accsStr = [str(a)[:6] for a in accs]
    resultStr =  f'{pdb_id} {str(label.shape[0])} '
    resultStr += (' '.join(accsStr))
    return resultStr, accs

def RR2contact(rr, seq_len):
    cmap = np.zeros([seq_len, seq_len])

    f = open(rr, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        line = line.strip()
        line = line.split(' ')

        try:
            i = int(line[0])
            j = int(line[1])
            p = float(line[2])

            cmap[i, j] = p
            cmap[j, i] = p
        except:
            print(line)

    return cmap

def evaluate(results, label_dir, CB_index=1, is_print = True):
    npys = os.listdir(label_dir)
    npys = [npy for npy in npys if '.npy' in npy]

    count = 0
    accs_all = []
    for npy in npys:
        label = os.path.join(label_dir, npy)
        pdb_id   = os.path.basename(npy).replace('.npy', '')

        if pdb_id in results.keys() and os.path.exists(label):
            label = np.load(label)
            try:
                resultStr, accs = get_contact_precision_(results[pdb_id], label, CB_index=CB_index, pdb_id=pdb_id)
                accs_all.append(accs)
                if is_print:
                    print(resultStr)
                count += 1
            except:
                continue

    accs_all = np.asarray(accs_all)
    accs_ave = np.mean(accs_all, axis=0)
    # print('Top L long range contact: ',str(accs_ave[0])[:6])
    return np.asarray(accs_all)[:, :1]

def get_ids(txt):
    f = open(txt)
    ids = f.readlines()
    f.close()
    return [id.strip() for id in ids]

def get_CASP_lists(casp = '13', catagory = 'FM'):

    if catagory == "FM":
        ids = get_ids(f'{os.path.split(os.path.realpath(__file__))[0]}/targets/casp{casp}_fm.txt')
    elif catagory == "TBM":
        ids = get_ids(f'{os.path.split(os.path.realpath(__file__))[0]}/targets/casp{casp}_tbm.txt')
    elif catagory == "ALL":
        ids = get_ids(f'{os.path.split(os.path.realpath(__file__))[0]}/targets/casp{casp}_fm.txt') + \
              get_ids(f'{os.path.split(os.path.realpath(__file__))[0]}/targets/casp{casp}_tbm.txt')

    return ids

def eval_casp_from_dict(re_dict, CB_index=1, MODE = 'CASP13', val_root = './DISTANCE_VAL', is_print = False):

    if 'CASP13' in MODE:
        label_dir = f'{val_root}/CASP13/CASP13DM_label_v2'
        fm_ids = get_CASP_lists(casp='13', catagory='FM')
        tbm_ids = get_CASP_lists(casp='13', catagory='TBM')
    elif 'CASP14' in MODE:
        label_dir = f'{val_root}/CASP14/casp14.targ.domains_labels_v2'
        fm_ids = get_CASP_lists(casp='14', catagory='FM')
        tbm_ids = get_CASP_lists(casp='14', catagory='TBM')
    else:
        raise RuntimeError()

    results_dict = {}

    for id in fm_ids:
        if id not in re_dict:
            print(f'missing pdb id {id}')
            continue
        results_dict[id] = re_dict[id]

    topl_fm = evaluate(results_dict, label_dir, CB_index=CB_index, is_print=is_print)

    if 'FM' not in MODE:
        results_dict = {}
        for id in tbm_ids:
            if id not in re_dict:
                print(f'missing pdb id {id}')
                continue
            results_dict[id] = re_dict[id]
        topl_tbm  = evaluate(results_dict, label_dir, CB_index=CB_index, is_print=is_print)
        topl_all = np.concatenate([topl_fm,topl_tbm], axis=0)
    else:
        topl_tbm = [0.0]
        topl_all = [0.0]

    results_str = f'{MODE} topl-lr ' \
                  f'FM {str(np.mean(topl_fm))[:6]} ' \
                  f'TBM {str(np.mean(topl_tbm))[:6]} ' \
                  f'ALL {str(np.mean(topl_all))[:6]} ' \
                  f'{len(topl_fm)} ' \
                  f'{len(topl_tbm)} ' \
                  f'{len(topl_all)}'

    return results_str, [topl_fm, topl_tbm, topl_all]

def eval_casp_from_npz(npz_dir, CB_index=1, MODE = 'CASP13', is_print = False):

    re_dict = {}
    for npz in os.listdir(npz_dir):
        if '.npz' in npz:
            pdb_id = npz.replace('.npz','')
            npz = f'{npz_dir}/{npz}'
            dist = np.load(npz)['dist']
            re_dict[pdb_id] = dist

    return eval_casp_from_dict(re_dict, CB_index=CB_index, MODE = MODE, is_print = is_print)

def eval_contact(npz_dir, lbl_dir, tgt_list, npz_mode = 'trRosetta', is_print = False):

    pdb_ids = get_ids(tgt_list)

    accs_all = []
    for pdb_id in pdb_ids:
        npz = f'{npz_dir}/{pdb_id}.npz'
        lbl = f'{lbl_dir}/{pdb_id}.npy'

        if not os.path.exists(npz):
            print(f'missing {npz}')
            continue
        if not os.path.exists(lbl):
            print(f'missing {lbl}')
            continue

        dist = np.load(npz)['dist']

        if npz_mode == 'RoseTTAFold':
            dist = np.concatenate([dist[:, :, -1:], dist[:, :, :-1]], axis=-1)

        label = np.load(lbl)
        resultStr, accs = get_contact_precision_(dist, label, pdb_id = pdb_id)
        accs_all.append(accs)

        if is_print:
            print(resultStr)

    accs_ave = np.mean(np.asarray(accs_all), axis=0)
    print('Top L long range contact: ',str(accs_ave[0])[:6])

def calc_cnt_af2(npz_fpath_mod, labelv2_fpath_ref, pdb_id = 'T1xx', truncate = -1, softmax = False):
    pos = dict(np.load(npz_fpath_mod))

    if softmax:
        pos['logits'] = softmax_np(pos['logits'], axis=-1)

    index = 19

    # "distogram": {
    #     "min_bin": 2.3125,
    #     "max_bin": 21.6875,
    #     "no_bins": 64,
    #     "eps": eps,  # 1e-6,
    #     "weight": 0.3,
    # }

    pos = np.sum(pos['logits'][:, :, :index], axis=2)
    lbl = np.load(labelv2_fpath_ref)

    if truncate > 0:
        pos = pos[:truncate,:truncate]
        lbl = lbl[:truncate,:truncate]

    resultStr, accs = get_contact_precision__(pos, lbl, CB_index=1, pdb_id=pdb_id)
    return accs[0], resultStr

def calc_cnt_trros(npz_fpath_mod, labelv2_fpath_ref, pdb_id='T1xx', truncate = -1, softmax = False):

    if isinstance(npz_fpath_mod,list):
        pos_ = []
        for npz in npz_fpath_mod:
            dist = dict(np.load(npz))['dist']

            if np.sum(np.isnan(dist)) >= 1 or np.sum(np.isinf(dist)) >= 1 :
                print(f'nan in {npz}, remove')
                os.remove(npz)
                break

            if softmax:
                dist = softmax_np(dist, axis=-1)

            dim = dist.shape[-1]
            ratio = (dim-1) // 36
            pos = np.sum(dist[:, :, 1:1 + ratio * 12], axis=2, keepdims=True)
            pos_.append(pos)

        pos = np.concatenate(pos_, axis = -1)
        pos = np.mean(pos, axis = -1)
    else:
        dist = dict(np.load(npz_fpath_mod))['dist']

        if np.sum(np.isnan(dist)) >= 1 or np.sum(np.isinf(dist)) >= 1:
            print(f'nan in {npz_fpath_mod}, remove')
            os.remove(npz_fpath_mod)

        if softmax:
            dist = softmax_np(dist, axis=-1)

        dim = dist.shape[-1]
        ratio = (dim - 1) // 36
        pos = np.sum(dist[:, :, 1:1 + ratio * 12], axis=2, keepdims=True)

    lbl = np.load(labelv2_fpath_ref)

    if truncate > 0:
        pos = pos[:truncate,:truncate]
        lbl = lbl[:truncate,:truncate]

    resultStr, accs = get_contact_precision__(pos, lbl, CB_index=1, pdb_id = pdb_id)
    return accs[0]

def calc_cnt_rosettafold(npz_fpath_mod, labelv2_fpath_ref, pdb_id='T1xx', truncate = -1):

    if isinstance(npz_fpath_mod,list):
        pos_ = []
        for npz in npz_fpath_mod:
            dist = dict(np.load(npz))['dist']
            pos = np.sum(dist[:, :, :13-1], axis=2, keepdims=True)
            pos_.append(pos)
        pos = np.concatenate(pos_, axis = -1)
        pos = np.mean(pos, axis = -1)
    else:
        dist = dict(np.load(npz_fpath_mod))['dist']
        pos = np.sum(dist[:, :, :13-1], axis=2)

    lbl = np.load(labelv2_fpath_ref)

    if truncate > 0:
        pos = pos[:truncate,:truncate]
        lbl = lbl[:truncate,:truncate]

    resultStr, accs = get_contact_precision__(pos, lbl, CB_index=1, pdb_id=pdb_id)
    return accs[0]#, resultStr

def eval_contact_file(npz_fpath_mod, labelv2_fpath_ref, npz_mode, result_dict=None):
    """Evaluate the PDB file w/ specified metric."""

    if npz_mode == 'alphafold2':
        score = calc_cnt_af2(npz_fpath_mod, labelv2_fpath_ref, npz_mode)
    elif npz_mode == 'trrosetta':
        score = calc_cnt_trros(npz_fpath_mod, labelv2_fpath_ref, npz_mode)
    else:
        raise ValueError('unrecognized evaluation metric: ' + npz_mode)

    if result_dict is not None:
        result_dict[(pdb_fpath_mod, metric)] = score

    return score

def convert_distogram_af2trros(af_dist):
    seq_lem, seq_len, bins = af_dist.shape
    trros_dist = np.zeros([seq_len, seq_len, 37])

    inter_af2 = 20 / 64 # 2A - 22A
    inter_trros = 18 / 36 # 2A - 20A

    for i in range(36):
        s_ = (inter_trros * i) / inter_af2
        e_ = (inter_trros * (i + 1)) / inter_af2
        s_idx = math.ceil(s_)
        e_idx = int(e_)
        trros_dist[:,:,i+1] = np.sum(af_dist[:,:, s_idx:e_idx], axis = -1) \
                                   - (s_ - s_idx) * af_dist[:,:, s_idx] \
                                   - (e_idx - e_) * af_dist[:, :, e_idx]

    trros_dist[:,:,0] = 1 - np.sum(trros_dist[:,:,1:], axis = -1)

    return trros_dist

def eval_npz():
    fire.Fire(eval_contact_file)
