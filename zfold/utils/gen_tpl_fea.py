import os
import random
import pickle
import numpy as np
from zfold.utils.constants import *

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

########################################################################################################################
def parse_tpl_file(path, n_resds, tpl_topk, is_train, mode = 'v1'):
    """Parse the TPL file (structural templates)."""

    idx_mask = -1

    t1ds_tns, t2ds_tns = get_tpl_feature(
            path, tpl_topk=tpl_topk, is_train=is_train, mask_index=idx_mask, mode=mode, seq_len = n_resds)

    # pack into a dict
    data_dict = {
        't1ds': t1ds_tns,  # M x L x d1d
        't2ds': t2ds_tns,  # M x L x L x 1
    }  # M: number of structural templates

    return data_dict

########################################################################################################################
def get_tpl_feature(tpl_path,
                    tpl_topk,
                    is_train,
                    mask_index,
                    seq_len,
                    mode = 'v1'):

    d1d, d2d = tpl_fea_dict[mode][0], tpl_fea_dict[mode][1]

    if tpl_path is None:
        t1ds = torch.ones([tpl_topk, seq_len, d1d]) * mask_index
        t2ds = torch.ones([tpl_topk, seq_len,seq_len, d2d]) * mask_index
        return t1ds, t2ds

    if mode == 'v1':
        with open(tpl_path[0], 'rb') as f:
            feature_dict = pickle.load(f)
        return get_template_features(feature_dict,
                                     tpl_num = tpl_topk,
                                     atom = 'CB',
                                     mask_index = mask_index,
                                     is_train = is_train)

    elif mode == 'v3':
        feature_dict = dict(np.load(tpl_path[1]))
        return get_template_features_v3(feature_dict,
                                        tpl_num = tpl_topk,
                                        mask_index = mask_index,
                                        is_train = is_train,
                                        tpl_mode = mode)

    elif mode == 'v3.1':
        with open(tpl_path[0], 'rb') as f:
            feature_dict = pickle.load(f)
        feature_dict.update(dict(np.load(tpl_path[1])))

        return get_template_features_v3(feature_dict,
                                        tpl_num = tpl_topk,
                                        mask_index = mask_index,
                                        is_train = is_train,
                                        tpl_mode = mode)

    else:
        raise NotImplementedError



def squared_difference(x, y):
    return np.square(x - y)

def squared_difference_tensor(x, y):
    return torch.square(x - y)

def get_atom_pair_distance(tpl_index, atom, feature_dict):
    index = atom_order[atom]
    mask = feature_dict['template_all_atom_masks'][tpl_index, :, index]
    atoms = feature_dict['template_all_atom_positions'][tpl_index, :, index, :]
    mask = np.matmul(mask.reshape([-1, 1]), mask.reshape([1, -1]))
    dist = np.sqrt(1e-10 + np.sum(squared_difference(atoms[:, None, :], atoms[None, :, :]), axis=-1))
    return dist, mask

def get_atom_pair_distance_tensor(atoms, mask):
    mask = torch.matmul(mask.reshape([-1, 1]), mask.reshape([1, -1]))
    dist = torch.sqrt(1e-10 + torch.sum(squared_difference_tensor(atoms[:, None, :], atoms[None, :, :]), dim=-1))
    return dist, mask

# template feature v1
def get_template_features(feature_dict, tpl_num = 4, atom = 'CB', mask_index = -1, is_train = False):
    #  Return:
    #   - t1d: 1D template info (B, T, L, 2)
    #   - t2d: 2D template info (B, T, L, L, 10)

    max_ = 50.75
    min_ = 3.25
    bins = 38
    inter = (max_ - min_) / bins

    seq_len, _ = feature_dict['aatype'].shape
    t1ds = np.ones([tpl_num, seq_len, 23]) * mask_index
    t2ds = np.ones([tpl_num, seq_len, seq_len, 1]) * mask_index

    if isinstance(feature_dict['template_all_atom_masks'], list) or feature_dict['template_all_atom_masks'].size == 0:
        t1ds = torch.FloatTensor(t1ds)
        t2ds = torch.FloatTensor(t2ds)
        return t1ds, t2ds

    #for data augmentation
    if is_train and random.randint(0, 1) == 0:
        t1ds = torch.FloatTensor(t1ds)
        t2ds = torch.FloatTensor(t2ds)
        return t1ds, t2ds

    tpl_num_pkl, _, _ = feature_dict['template_all_atom_masks'].shape

    for tpl in range(tpl_num):
        tpl_index = tpl

        # for data augmentation
        if is_train and random.randint(0, 1) == 0:
            tpl = random.randint(0, tpl_num_pkl + tpl_num - 1)

        if tpl + 1 > tpl_num_pkl:
            # print(f'pad tpl: {tpl+1}/{tpl_num_pkl}')
            break

        dist, mask = get_atom_pair_distance(tpl, atom, feature_dict)

        distogram = np.ones([seq_len, seq_len]) * bins

        distogram[(dist < min_)] = 0

        for i in range(bins):
            s = min_ + inter * i
            e = max_ + inter * (i + 1)
            distogram[(dist >= s) & (dist < e)] = i

        # mask index: -1
        distogram[mask == 0] = mask_index
        t2ds[tpl_index,:,:,0] = distogram

        # template_aatype(20, 391, 22)
        # template_confidence_scores(20, 391)
        template_aatype = feature_dict['template_aatype'][tpl,:,:]
        template_confidence_scores = feature_dict['template_confidence_scores'][tpl,:]
        template_confidence_scores = template_confidence_scores.reshape([seq_len, 1])
        t1d = np.concatenate([template_aatype, template_confidence_scores], axis = -1)
        t1ds[tpl_index,:,:] = t1d

    t1ds = torch.FloatTensor(t1ds)
    t2ds = torch.FloatTensor(t2ds)

    return t1ds, t2ds

def get_template_features_v3(feature_dict, tpl_num = 4, mask_index = -1, is_train = False, tpl_mode = 'v3'):
    #  Return:
    #   - t1d: 1D template info (B, T, L, 2)
    #   - t2d: 2D template info (B, T, L, L, 10)

    seq_len, _ = feature_dict['aatype'].shape

    t1ds = torch.ones([tpl_num, seq_len, tpl_fea_dict[tpl_mode][0]]) * mask_index
    t2ds = torch.ones([tpl_num, seq_len, seq_len, tpl_fea_dict[tpl_mode][1]]) * mask_index

    int_keys = ['aatype',
                'between_segment_residues',
                'residue_index',
                'seq_length',
                'template_aatype',
                'template_all_atom_mask',
                'template_mask',
                'template_pseudo_beta_mask',
                'template_torsion_angles_mask']

    fp_keys = ['template_pseudo_beta',
               'template_torsion_angles_sin_cos',
               'template_alt_torsion_angles_sin_cos',
               'template_all_atom_positions',
               'template_sum_probs',
               'template_confidence_scores']

    template_feats = {}
    for k, v in feature_dict.items():
        if k in fp_keys:
            v = torch.FloatTensor(v)
        elif k in int_keys:
            v = torch.LongTensor(v)
        else:
            # raise NotImplementedError
            continue
        template_feats[k] = v

    if template_feats["template_aatype"].shape[0] == 0:
        return t1ds, t2ds

    def get_idx_fea(idx, templ_dim = 0, config = model_config('model_1')):
        idx = template_feats["template_aatype"].new_tensor(idx)
        single_template_feats = tensor_tree_map(
            lambda t: torch.index_select(t, templ_dim, idx),
            template_feats,
        )
        # [*, S_t, N, N, C_t]
        t2d = build_template_pair_feat(
            single_template_feats,
            inf=config.model.template.inf,
            eps=config.model.template.eps,
            **config.model.template.distogram,
        )
        t1d = build_template_angle_feat(
            single_template_feats,
        )
        return t1d, t2d

    tpl_num_pkl = template_feats["template_aatype"].shape[0]

    for tpl in range(tpl_num):
        tpl_index = tpl

        # for data augmentation
        if is_train and random.randint(0, 1) == 0:
            tpl = random.randint(0, tpl_num_pkl + tpl_num - 1)

        if tpl + 1 > tpl_num_pkl:
            break

        t1d, t2d = get_idx_fea(tpl, templ_dim=0)

        if 'template_confidence_scores' in feature_dict:
            template_confidence_scores = torch.FloatTensor(feature_dict['template_confidence_scores'][tpl,:])
            template_confidence_scores = template_confidence_scores.reshape([1, seq_len, 1])
            t1d = torch.cat([t1d, template_confidence_scores], dim = -1)

        t1ds[tpl_index,:,:] = t1d
        t2ds[tpl_index,:,:,:] = t2d

    return t1ds, t2ds




