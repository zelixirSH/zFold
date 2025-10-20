import random
import torch
import torch.nn as nn
from argparse import Namespace
import torch
from box import Box

import zfold.network.esm as esm
import zfold.network.esm_latest as esm_latest
from zfold.network.esm.modules import RobertaLMHead, init_bert_params
from zfold.network.embeddings import MSA_emb, Pair_emb
from zfold.network.modules import IterativeFeatureExtractor2D
from zfold.network.predictor import PairPredictor
from zfold.network.utils import init_zero_mlp
from zfold.utils import exists, load_pretrain

import ml_collections

class XFold2D(nn.Module):
    def __init__(self, config):
        super(XFold2D, self).__init__()

        config = ml_collections.ConfigDict(config)

        self.config = config
        self.alphabet = esm.Alphabet.from_architecture('MSA Transformer')
        self.is_dist_head = config.dist.is_dist_head
        self.is_lm_head = config.mlm.is_lm_head
        self.d_msa = config.basic.d_msa
        self.d_pair = config.basic.d_pair
        self.n_recycle = config.basic.n_recycle
        self.use_templ = config.templ.use_templ
        self.msa_bert_config = config.msa_bert.msa_bert_config

        # initial msa fea
        msa_emb_d_msa = config.modules.msa_emb.d_msa \
            if config.modules.msa_emb.is_onehot else len(self.alphabet)

        self.msa_emb = MSA_emb(d_model = self.d_msa,
                               d_msa = msa_emb_d_msa,
                               padding_idx = self.alphabet.padding_idx,
                               is_pos_emb = config.pos_emb.is_pos_emb,
                               is_peg = config.pos_emb.is_peg,
                               n_peg_block = config.pos_emb.n_peg_block,
                               is_onehot = config.modules.msa_emb.is_onehot,
                               )

        self.pair_emb = Pair_emb(d_model = self.d_pair,
                                     d_templ = config.templ.d_templ,
                                     d_msa = len(self.alphabet),
                                     d_t1d = config.templ.d_t1d,
                                     d_t2d = config.templ.d_t2d,
                                     use_templ = config.templ.use_templ,
                                     is_pos_emb = config.pos_emb.is_pos_emb,
                                     is_peg = config.pos_emb.is_peg,
                                     n_peg_block = config.pos_emb.n_peg_block,
                                     pb_relax = config.normalization.pb_relax,
                                     )

        """
        # MSA extra stack
        if config.modules.extra_msa_stack.enable:
            if config.modules.extra_msa_stack.is_onehot:
                self.extra_msa_embedder = nn.Linear(config.modules.extra_msa_stack.d_msa,
                                                    config.modules.extra_msa_stack.dim)
            else:
                self.extra_msa_embedder = MSA_emb(d_model = config.modules.extra_msa_stack.dim,
                                                  d_msa = len(self.alphabet),
                                                  padding_idx=self.alphabet.padding_idx,
                                                  is_pos_emb=config.pos_emb.is_pos_emb,
                                                  is_peg=config.pos_emb.is_peg,
                                                  n_peg_block=config.pos_emb.n_peg_block,
                                                  )
            self.msa_extra_stack = MSAExtraStack(d_msa = config.modules.extra_msa_stack.dim,
                                                d_pair = config.basic.d_pair,
                                                is_rezero=config.initialization.is_rezero,
                                                is_use_ln=config.normalization.is_use_ln,
                                                rpe_type=config.pos_emb.rpe_type,
                                                is_peg=config.pos_emb.is_peg,
                                                n_peg_block=config.pos_emb.n_peg_block,
                                                use_checkpoint=config.use_checkpoint,
                                                pair2msa_type=config.modules.pair2msa_type,
                                                msa2msa_type=config.modules.msa2msa_type,
                                                pair2pair_type=config.modules.pair2pair_type,
                                                n_layer_shift_tokens=config.attention.n_layer_shift_tokens,
                                                proj_type=config.attention.proj_type,
                                                )
        """

        self.feat_extractor = IterativeFeatureExtractor2D(n_module = config.basic.n_module,
                                                          n_layer_msa = config.basic.n_layer_msa,
                                                          n_layer_pair = config.basic.n_layer_pair,
                                                          d_msa = config.basic.d_msa,
                                                          d_pair = config.basic.d_pair,
                                                          n_head_msa = config.basic.n_head_msa,
                                                          n_head_pair = config.basic.n_head_pair,
                                                          r_ff = config.basic.r_ff,
                                                          p_drop = config.dropout.p_drop,
                                                          p_attn_drop = config.dropout.p_attn_drop,
                                                          p_layer_drop = config.dropout.p_layer_drop,
                                                          is_p_layer_drop = config.dropout.is_p_layer_drop,
                                                          activation = config.activation,
                                                          is_rezero = config.initialization.is_rezero,
                                                          is_use_ln = config.normalization.is_use_ln,
                                                          is_sandwich_norm = config.normalization.is_sandwich_norm,
                                                          pb_relax = config.normalization.pb_relax,
                                                          rpe_type = config.pos_emb.rpe_type,
                                                          is_peg = config.pos_emb.is_peg,
                                                          n_peg_block = config.pos_emb.n_peg_block,
                                                          use_checkpoint = config.use_checkpoint,
                                                          pair2msa_type = config.modules.pair2msa_type,
                                                          msa2msa_type = config.modules.msa2msa_type,
                                                          pair2pair_type = config.modules.pair2pair_type,
                                                          n_layer_shift_tokens = config.attention.n_layer_shift_tokens,
                                                          proj_type = config.attention.proj_type,
                                                          ) if config.basic.n_module > 0 else None

        if self.is_dist_head:
            self.c6d_predictor = PairPredictor(n_feat = config.basic.d_pair,
                                               p_drop = config.dropout.p_drop,
                                               activation = config.activation,
                                               bins = config.dist.bins,
                                               pb_relax = config.normalization.pb_relax,
                                               )

        if self.is_lm_head:
            self.lm_head = RobertaLMHead(embed_dim = config.basic.d_msa,
                                         output_dim = len(self.alphabet),
                                         weight = self.msa_emb.get_emb_weight(),
                                         n_in = msa_emb_d_msa,
                                         n_out = self.d_msa,
                                         )

        # Init params before pretrained msa_bert is defined
        self.apply(init_bert_params)

        if config.initialization.is_init_zero:
            self.apply(init_zero_mlp)

        self.fas_bert, self.fas_bert_reduction = None, None
        if config.fas_bert.enable:
            # Load ESM-1b model
            self.fas_bert_dim = 1280
            self.fas_bert, _ = esm_latest.pretrained.esm1b_t33_650M_UR50S(local_dir = config.fas_bert.local_dir)
            self.fas_bert.eval()
            for param in self.fas_bert.parameters():
                param.detach_()
            self.fas_bert_reduction = nn.Linear(self.fas_bert_dim + self.d_msa, self.d_msa)

        self.bert_model, self.token_reduction = None, None
        if exists(config.msa_bert.msa_bert_config) and not self.config.modules.msa_emb.is_onehot:
            params = Box.from_yaml(filename = config.msa_bert.msa_bert_config.model_yaml)
            self.bert_layer_num, self.bert_dim = params.num_layers, params.embed_dim
            self.bert_model = esm.MSATransformer(Namespace(**params))
            self.bert_pos_emb = config.msa_bert.pos_emb

            if exists(config.msa_bert.msa_bert_config.model_weight) and not config.msa_bert.skip_load_msa_bert:
                self.bert_model = load_pretrain(self.bert_model, config.msa_bert.msa_bert_config.model_weight)

            if config.msa_bert.msa_bert_config.freeze:
                print('    frezze pretrained msa transformer')
                for param in self.bert_model.parameters():
                    param.detach_()
                self.bert_model.eval()

            self.token_reduction = nn.Linear(self.bert_dim, self.d_msa)

    def forward_fea_init(self, tokens, t1ds = None, t2ds = None, is_BKL = True,
                               tokens_feats = None, extra_tokens_feats = None, **unused,):

        assert tokens.ndim == 3
        if not is_BKL:
            tokens = tokens.permute(0, 2, 1)
        B, K, L = tokens.size()# batch_size, num_alignments, seq_len

        if exists(self.msa_bert_config) and not self.config.modules.msa_emb.is_onehot:
            bert_results = self.bert_model(tokens, repr_layers=[self.bert_layer_num], pos_emb = self.bert_pos_emb)
            msa_fea = bert_results["representations"][self.bert_layer_num].view(B * K * L, -1)
            msa_fea = self.token_reduction(msa_fea).view(B * K, L, -1).view(B, K, L, self.d_msa)
        else:
            msa_fea = self.msa_emb(tokens)

        if self.config.fas_bert.enable:
            fas_bert_results = self.fas_bert(tokens[:,0,:], repr_layers=[33], return_contacts = False)
            token_representations = fas_bert_results["representations"][33].unsqueeze(1).expand(-1,K,-1,-1)
            msa_fea = self.fas_bert_reduction(torch.cat([token_representations, msa_fea], dim = -1))

        init_msa = msa_fea
        pair_fea = self.pair_emb(tokens, t1ds, t2ds)

        return msa_fea, pair_fea, init_msa

    def forward_fea_refine(self, msa_fea, pair_fea, n_recycle):

        if not exists(self.feat_extractor):
            return msa_fea, pair_fea

        for i in range(n_recycle):
            msa_fea, pair_fea = self.feat_extractor(msa_fea, pair_fea)
            msa_fea, pair_fea = msa_fea.detach_(), pair_fea.detach_()

        msa_fea, pair_fea = self.feat_extractor(msa_fea, pair_fea)

        return msa_fea, pair_fea

    def forward_fea(self, tokens, t1ds = None, t2ds = None, is_BKL = True,
                    tokens_feats = None, extra_tokens_feats = None, n_recycle = None, **unused):
        ''

        msa_fea, pair_fea, init_msa = self.forward_fea_init(tokens= tokens, t1ds = t1ds, t2ds = t2ds, is_BKL = is_BKL,
                                                            tokens_feats = tokens_feats, extra_tokens_feats = extra_tokens_feats)

        if n_recycle is None:
            n_recycle = random.randint(0, self.n_recycle) if self.training else self.n_recycle
        else:
            print(f'n recycle: {n_recycle}')

        msa_fea, pair_fea = self.forward_fea_refine(msa_fea, pair_fea, n_recycle)

        return msa_fea, pair_fea, init_msa


    def forward(self, tokens, t1ds = None, t2ds = None, masked_tokens = None, is_fea = False, is_BKL = True,
                tokens_feats = None,
                extra_tokens_feats = None,
                n_recycle = None,
                **unused):
        ''

        assert tokens.ndim == 3

        K = tokens.size()[2] if not is_BKL else tokens.size()[1]  # batch_size, num_alignments, seq_len

        # Extract features
        msa_fea, pair_fea, init_msa_fea = self.forward_fea(tokens = tokens,
                                                           t1ds = t1ds,
                                                           t2ds = t2ds,
                                                           is_BKL = is_BKL,
                                                           n_recycle = n_recycle,
                                                           )
        # Keep top K msa embedding
        msa_fea = msa_fea[:,:K,:,:]

        output = {}
        if is_fea:
            output['msa_fea'], output['pair_fea'] = msa_fea, pair_fea

        if self.is_dist_head:
            output['dist'], output['omega'], output['theta'], output['phi'] = \
                self.c6d_predictor(pair_fea)

        if self.is_lm_head:
            # This is for MLM loss
            msa_fea = msa_fea.permute((0,2,1,3))# B, K, L, D -> B, L, K, D
            B, L, K, D = msa_fea.shape
            msa_fea = msa_fea.reshape([B, L * K, D])
            msa_fea = msa_fea[masked_tokens, :]
            output['logits_mlm'] = self.lm_head(msa_fea)

        return output

    def mode(self, freeze = False):
        if freeze:
            self.eval()
            for param in self.parameters():
                param.detach_()
