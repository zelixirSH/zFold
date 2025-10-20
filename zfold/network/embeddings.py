import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from zfold.network.attention import LayerNorm
from zfold.network.conv_blocks import SEBottleneck
from zfold.network.esm.modules import LearnedPositionalEmbedding

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, p_drop=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.drop = nn.Dropout(p_drop)#, inplace=True
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0., d_model_half, 2) *
                             -(math.log(10000.0) / d_model_half))
        self.register_buffer('div_term', div_term)

    def forward(self, x, idx_s):
        B, L, _, K = x.shape
        K_half = K//2
        pe = torch.zeros_like(x)
        i_batch = -1
        for idx in idx_s:
            i_batch += 1

            if idx.device != self.div_term.device:
                idx = idx.to(self.div_term.device)

            sin_inp = idx.unsqueeze(1) * self.div_term
            emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1) # (L, K//2)
            pe[i_batch,:,:,:K_half] = emb.unsqueeze(1)
            pe[i_batch,:,:,K_half:] = emb.unsqueeze(0)
        x = x + torch.autograd.Variable(pe, requires_grad=False)
        return self.drop(x)

class MSA_emb(nn.Module):
    def __init__(self,
                 d_model = 64,
                 d_msa = 21,
                 padding_idx = None,
                 max_len = 4096,
                 is_pos_emb = True,
                 is_peg = False,
                 n_peg_block = 1,
                 is_onehot = False,
                 p_drop=0.,
                 **unused):
        super(MSA_emb, self).__init__()
        self.is_pos_emb = is_pos_emb
        self.is_peg = is_peg
        self.is_onehot = is_onehot

        if self.is_onehot:
            self.embed_tokens = nn.Linear(d_msa, d_model)
        else:
            self.embed_tokens = nn.Embedding(d_msa, d_model, padding_idx = padding_idx)

        if self.is_pos_emb:
            self.embed_positions = LearnedPositionalEmbedding(max_len, d_model, padding_idx)

        if self.is_peg:
            layer_s = [SEBottleneck(d_model, d_model, stride=1, dilation=1, is_depthwize=True)
                       for i in range(n_peg_block)]
            self.update = nn.Sequential(*layer_s)

    def forward(self, tokens):

        if self.is_onehot:
            B, K, L, _ = tokens.shape
            msa_fea = self.embed_tokens(tokens)
        else:
            B, K, L = tokens.shape
            msa_fea = self.embed_tokens(tokens)

            if self.is_pos_emb:
                msa_fea += self.embed_positions(tokens.reshape(B * K, L)).view(msa_fea.size())

        if self.is_peg:
            msa_fea = msa_fea.permute(0, 3, 1, 2).contiguous()  # prep for convolution layer
            msa_fea = self.update(msa_fea)
            msa_fea = msa_fea.permute(0, 2, 3, 1).contiguous()  # (B, L, L, D)

        return msa_fea

    def get_emb_weight(self):
        return self.embed_tokens.weight

class Pair_emb(nn.Module):
    def __init__(self,
                 d_model = 64,
                 d_templ = 64,
                 d_msa = 21,
                 d_t1d = None,
                 d_t2d = None,
                 p_drop = 0.,
                 use_templ = False,
                 is_pos_emb = True,
                 is_peg = False,
                 n_peg_block = 1,
                 pb_relax = False
                 ):
        super(Pair_emb, self).__init__()
        self.use_templ = use_templ
        self.is_peg = is_peg

        if self.use_templ:
            self.templ_emb = Templ_emb_simple(d_t1d=d_t1d, d_t2d=d_t2d, d_templ=d_templ)
            self.pair_emb = Pair_emb_w_templ(d_model = d_model,
                                             d_templ = d_templ * 4,
                                             p_drop  = p_drop,
                                             d_seq = d_msa,
                                             is_pos_emb = is_pos_emb,
                                             pb_relax = pb_relax,
                                             )
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model= d_model,
                                              p_drop = p_drop,
                                              d_seq  = d_msa,
                                              is_pos_emb = is_pos_emb,
                                              add_sep= False)

        if self.is_peg:
            layer_s = [SEBottleneck(d_model, d_model, stride=1, dilation=1, is_depthwize=True)
                        for i in range(n_peg_block)]
            self.update = nn.Sequential(*layer_s)

    def forward(self, msa_tokens, t1ds, t2ds):
        seq_tokens = msa_tokens[:, 0, :]

        B, L = seq_tokens.shape
        idx = torch.cat([torch.arange(L).long().unsqueeze(0) for i in range(B)], dim=0)

        if idx.device != seq_tokens.device:
            idx = idx.to(seq_tokens.device)

        if self.use_templ:
            pair_fea = self.pair_emb(seq_tokens, idx, self.templ_emb(t1ds, t2ds))
        else:
            pair_fea = self.pair_emb(seq_tokens, idx)

        if self.is_peg:
            pair_fea = pair_fea.permute(0, 3, 1, 2).contiguous()  # prep for convolution layer
            pair_fea = self.update(pair_fea)
            pair_fea = pair_fea.permute(0, 2, 3, 1).contiguous()  # (B, L, L, D)

        return pair_fea

class Templ_emb_simple(nn.Module):
    def __init__(self, d_t1d=3, d_t2d=10, d_templ=64):
        super(Templ_emb_simple, self).__init__()
        self.proj = nn.Linear(d_t1d * 2 + d_t2d, d_templ)

    def forward(self, t1d, t2d):
        # Input
        #   - t1d: 1D template info (B, T, L, 2)
        #   - t2d: 2D template info (B, T, L, L, 10)
        B, T, L, _ = t1d.shape
        left = t1d.unsqueeze(3).expand(-1, -1, -1, L, -1)
        right = t1d.unsqueeze(2).expand(-1, -1, L, -1, -1)
        feat = torch.cat((t2d, left, right), -1)
        feat = self.proj(feat).reshape(B, T, L, L, -1)
        feat = feat.permute((0,2,3,1,4))
        return feat.reshape(B, L, L, -1)

class Pair_emb_w_templ(nn.Module):
    def __init__(self,
                 d_model = 128,
                 d_seq = 21,
                 d_templ = 64,
                 p_drop = 0.1,
                 is_use_ln = True,
                 is_pos_emb = True,
                 pb_relax = False):
        super(Pair_emb_w_templ, self).__init__()

        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.norm_templ = LayerNorm(d_templ, pb_relax=pb_relax) if is_use_ln else nn.Sequential()
        self.projection = nn.Linear(d_model + d_templ + 1, d_model)

        self.is_pos_emb = is_pos_emb
        if self.is_pos_emb:
            self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)

    def forward(self, seq, idx, templ):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        #
        # get initial sequence pair features
        seq = self.emb(seq) # (B, L, d_model//2)
        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        seqsep = torch.abs(idx[:,:,None]-idx[:,None,:]) + 1
        seqsep = torch.log(seqsep.float()).view(B,L,L,1)#.half()

        if isinstance(templ, torch.cuda.HalfTensor) or isinstance(templ, torch.HalfTensor):
            seqsep = seqsep.half()

        templ = self.norm_templ(templ)

        # print(left.shape, right.shape, seqsep.shape, templ.shape)

        pair = torch.cat((left, right, seqsep, templ), dim=-1)
        pair = self.projection(pair) # (B, L, L, d_model)

        if self.is_pos_emb:
            pair = self.pos(pair, idx)
        return pair

class Pair_emb_wo_templ(nn.Module):
    def __init__(self, d_model=128, d_seq=21, p_drop=0.1, add_sep = True,
                 is_pos_emb = True):
        super(Pair_emb_wo_templ, self).__init__()
        self.d_model = d_model
        self.d_emb = d_model // 2
        self.emb = nn.Embedding(d_seq, self.d_emb)
        self.add_sep = add_sep
        if self.add_sep:
            self.projection = nn.Linear(d_model + 1, d_model)
        else:
            self.projection = nn.Linear(d_model, d_model)

        self.is_pos_emb = is_pos_emb
        if self.is_pos_emb:
            self.pos = PositionalEncoding2D(d_model, p_drop=p_drop)

    def forward(self, seq, idx):
        # input:
        #   seq: target sequence (B, L, 20)
        B = seq.shape[0]
        L = seq.shape[1]
        seq = self.emb(seq) #(B, L, d_model//2)
        left  = seq.unsqueeze(2).expand(-1,-1,L,-1)
        right = seq.unsqueeze(1).expand(-1,L,-1,-1)
        if self.add_sep:
            seqsep = torch.abs(idx[:,:,None]-idx[:,None,:])+1
            seqsep = torch.log(seqsep.float()).view(B,L,L,1).half()
            pair = torch.cat((left, right, seqsep), dim=-1)
        else:
            pair = torch.cat((left, right), dim=-1)

        pair = self.projection(pair)
        if self.is_pos_emb:
            pair = self.pos(pair, idx)
        return pair
