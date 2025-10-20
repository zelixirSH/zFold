import random
import torch
import torch.nn as nn
import ml_collections

from zfold.zfoldnet import XFold2D
from zfold.network.utils import init_zero_mlp
from zfold.network.af2_smod.net.af2_smod_net import AF2SModNet
from zfold.network.af2_smod.net.rc_embed_net import RcEmbedNet
from zfold.utils import exists, sample_msa

class XFold(nn.Module):
    def __init__(self, config):
        super(XFold, self).__init__()
        config = ml_collections.ConfigDict(config)
        self.config = config

        self.msa_depth_max = config.basic.msa_depth_max
        self.msa_mask_prob = config.basic.msa_mask_prob

        # detach loss from 3d module
        self.detach_3d = config.af2_smod.detach_3d
        self.n_lyrs = config.af2_smod.n_lyrs
        self.g_recycle = config.g_recycle if 'g_recycle' in config else config.af2_smod.g_recycle

        self.xfold2d = XFold2D(config)
        self.xfold2d.mode(freeze = config.af2_smod.freeze)
        self.rc_embnet = RcEmbedNet(n_dims_mfea=config.basic.d_msa,
                                    n_dims_pfea=config.basic.d_pair) if self.g_recycle > 1 else None
        self.af2_smod = AF2SModNet(
                        n_dims_mfea = self.xfold2d.d_msa,
                        n_dims_pfea = self.xfold2d.d_pair,
                        n_lyrs = config.af2_smod.n_lyrs,
                        n_dims_attn = config.af2_smod.n_dim_attn,
                        v2 = config.af2_smod.v2,
                        plddt = config.af2_smod.plddt,
                        highres = config.af2_smod.highres,
                        finalloss = config.af2_smod.finalloss,
                        )

        if config.initialization.is_init_zero:
            self.af2_smod.apply(init_zero_mlp)

    def half(self):
        r"""Casts floating point parameters and buffers to ``half`` datatype.
        Returns:
            Module: self
        """

        '''
        self.xfold2d._apply(lambda t: t.half() if t.is_floating_point() else t)
        '''

        # FP16 part of xfold2d model
        self.xfold2d.msa_emb._apply(lambda t: t.half() if t.is_floating_point() else t)
        self.xfold2d.pair_emb._apply(lambda t: t.half() if t.is_floating_point() else t)
        self.xfold2d.feat_extractor._apply(lambda t: t.half() if t.is_floating_point() else t)

        if exists(self.xfold2d.bert_model):
            self.xfold2d.bert_model._apply(lambda t: t.half() if t.is_floating_point() else t)
        if exists(self.xfold2d.token_reduction):
            self.xfold2d.token_reduction._apply(lambda t: t.half() if t.is_floating_point() else t)
        if exists(self.xfold2d.fas_bert):
            self.xfold2d.fas_bert._apply(lambda t: t.half() if t.is_floating_point() else t)
        if exists(self.xfold2d.fas_bert_reduction):
            self.xfold2d.fas_bert_reduction._apply(lambda t: t.half() if t.is_floating_point() else t)

        # da & lm predictor are both in FP32
        # self.xfold2d.c6d_predictor._apply(lambda t: t.half() if t.is_floating_point() else t)
        # self.xfold2d.lm_head._apply(lambda t: t.half() if t.is_floating_point() else t)

        if exists(self.rc_embnet):
            self.rc_embnet._apply(lambda t: t.half() if t.is_floating_point() else t)

        return self

    def __pred(self, msa_fea, pair_fea, aa_seq, is_3d, is_fea):
        rc_inputs = None
        output = {}

        if self.xfold2d.is_lm_head:
            # B, K, L, D
            output['lm'] = self.xfold2d.lm_head(msa_fea.float())

        if self.xfold2d.is_dist_head:
            output['cb'], output['om'], output['th'], output['ph'] = self.xfold2d.c6d_predictor(pair_fea.float())

        # default = -1: use all the <AF2SMod> layers
        n_lyrs_sto = random.randint(1, self.n_lyrs) if self.training else -1

        if is_3d:
            # TODO fp32 for structure module
            tns_type_2d = msa_fea.dtype
            tns_type_3d = self.af2_smod.dtype()

            seq_fea = msa_fea[:, 0, ...]
            if self.detach_3d:
                seq_fea = seq_fea.detach()
                pair_fea = pair_fea.detach()
                print('detach seq_fea & pair fea')

            seq_fea = seq_fea.to(dtype = tns_type_3d)
            pair_fea = pair_fea.to(dtype = tns_type_3d)

            output['af2_smod_param'], output['highres'], output['plddt'], cord_list, output['fram_tns_sc'] = \
                        self.af2_smod(seq_fea, pair_fea, n_lyrs_sto = n_lyrs_sto, cords_fb = True, aa_seq = aa_seq)

            output['cords'] = [cord_[0] for cord_ in cord_list]
            output['cmask'] = [cord_[1] for cord_ in cord_list]

            if exists(self.rc_embnet):
                # record the current iteration's outputs for recycling embeddings
                rc_inputs = {
                    'sfea': seq_fea.detach().to(dtype=tns_type_2d),
                    'pfea': pair_fea.detach().to(dtype=tns_type_2d),
                    'cord': cord_list[-1][0].detach().to(dtype=tns_type_2d),
                }

        if is_fea:
            output['msa_fea'], output['pair_fea'] = msa_fea, pair_fea

        return output, rc_inputs


    def forward(self, tokens, t1ds = None, t2ds = None, is_BKL = True, is_fea = False, is_3d = True,
                aa_seq = None,
                is_sample_msa = False,
                recycle_ensemble = False,
                n_g_recycle = None,
                n_recycle = None,
                **unused):

        assert tokens.ndim == 3

        # Extract features
        if exists(self.rc_embnet):
            assert is_3d

        rc_inputs = None
        msa_feas, pair_feas, output_list = [], [], []

        if not exists(n_g_recycle) and not exists(n_recycle):
            # global 3d recycling, default 1
            n_g_recycle = random.randint(1, self.g_recycle) \
                if self.training and self.g_recycle > 1 else self.g_recycle
            # inner 2d recycling refinement, default 0
            n_recycle = random.randint(0, self.xfold2d.n_recycle) \
                if self.xfold2d.training and self.xfold2d.n_recycle > 0 else self.xfold2d.n_recycle

        mode = 'random' if self.training else 'topN'

        for _r in range(n_g_recycle):
            # sample msas
            msa_tokens_true, msa_tokens_pert, msa_masks = \
                sample_msa(tokens, msa_depth = self.msa_depth_max, is_train = True
                           if is_sample_msa else self.training, mask_prob = self.msa_mask_prob, mode = mode)

            K = msa_tokens_pert.size()[2] if not is_BKL else msa_tokens_pert.size()[1]  # batch_size, num_alignments, seq_len

            msa_fea, pair_fea, _ = self.xfold2d.forward_fea_init(tokens = msa_tokens_pert,
                                                                 t1ds = t1ds,
                                                                 t2ds = t2ds,
                                                                 is_BKL = is_BKL,
                                                                 )

            if exists(self.rc_embnet):
                if not self.training:
                    print(f'g_recycle {_r} {mode}')
                if _r > 1 and rc_inputs is None:
                    assert NotImplementedError

                msa_fea, pair_fea = self.rc_embnet(aa_seq, msa_fea, pair_fea, rc_inputs)

            msa_fea, pair_fea = self.xfold2d.forward_fea_refine(msa_fea, pair_fea, n_recycle)

            output, rc_inputs = self.__pred(msa_fea[:,:K,:,:], pair_fea, aa_seq, is_3d, is_fea)

            output['msa-t'] = msa_tokens_true
            output['msa-m'] = msa_masks

            output_list.append(output)
            msa_feas.append(msa_fea)
            pair_feas.append(pair_fea)

        if recycle_ensemble and n_g_recycle > 1:
            msa_fea = torch.mean(torch.stack(msa_feas, dim=0), dim=0)
            pair_fea = torch.mean(torch.stack(pair_feas, dim=0), dim=0)
            output, _ = self.__pred(msa_fea[:,:K,:,:], pair_fea, aa_seq, is_3d, is_fea)
            output_list.append(output)

        return output_list
