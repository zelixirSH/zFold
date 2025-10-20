import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from zfold.network.attention import *
from zfold.network.conv_blocks import SEBottleneck
from zfold.network.interact import CoevolExtractor

# These functions will not be used in the final version
class SequenceWeight(nn.Module):
    def __init__(self,
                 d_model,
                 heads,
                 p_attn_drop = 0.1,
                 proj_type = 'linear'):
        super(SequenceWeight, self).__init__()

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.proj_type = proj_type
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(p_attn_drop)

    def forward(self, msa):
        B, N, L = msa.shape[:3]

        msa = msa.permute(0,2,1,3) # (B, L, N, K)
        tar_seq = msa[:,:,0].unsqueeze(2) # (B, L, 1, K)

        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(0,1,3,2,4).contiguous() # (B, L, h, 1, k)
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0,1,3,4,2).contiguous() # (B, L, h, k, N)

        q = q * self.scale
        attn = torch.matmul(q, k) # (B, L, h, 1, N)
        attn = F.softmax(attn, dim=-1)
        return self.attn_drop(attn)

class MSA2Pair(nn.Module):
    def __init__(self,
                 n_feat = 64,
                 n_feat_out = 128,
                 n_feat_proj = 32,
                 p_drop = 0.1,
                 p_attn_drop = 0.,
                 p_layer_drop = 0.,
                 is_rezero = False,
                 is_use_ln = True,
                 is_peg = False,
                 n_peg_block=1,
                 is_sandwich_norm = False,
                 pb_relax = False,
                 **kwargs):
        super(MSA2Pair, self).__init__()
        self.is_peg = is_peg

        # project down embedding dimension (n_feat --> n_feat_proj)
        self.norm_1 = LayerNorm(n_feat, pb_relax = pb_relax) if is_use_ln else nn.Sequential()
        self.proj_1 = nn.Linear(n_feat, n_feat_proj)

        self.encoder = SequenceWeight(n_feat_proj, 1, p_attn_drop=p_attn_drop)
        self.coevol = CoevolExtractor(n_feat_proj, n_feat_out, pb_relax = pb_relax)

        self.norm_down = LayerNorm(n_feat_proj, pb_relax=pb_relax) if is_use_ln else nn.Sequential()
        self.norm_orig = LayerNorm(n_feat_out, pb_relax=pb_relax) if is_use_ln else nn.Sequential()
        self.norm_new  = LayerNorm(n_feat_out, pb_relax=pb_relax) if is_use_ln else nn.Sequential()

        fusion_dim = n_feat_out * 2 + n_feat_proj * 4

        if self.is_peg:
            layer_s = [SEBottleneck(fusion_dim, n_feat_out, stride=1, dilation=1, is_depthwize=True)]
            layer_s.extend([SEBottleneck(n_feat_out, n_feat_out, stride=1, dilation=1, is_depthwize=True)
                            for i in range(n_peg_block - 1)])
            self.update = nn.Sequential(*layer_s)

        else:
            self.update_norm = LayerNorm(fusion_dim, pb_relax=pb_relax)
            self.update = FeedForwardLayer(fusion_dim, fusion_dim,
                                           p_drop = p_drop,
                                           d_model_out = n_feat_out,
                                           is_post_act_ln = is_sandwich_norm,
                                           **kwargs)

            self.update_postnorm = LayerNorm(n_feat_out, pb_relax=pb_relax) if is_sandwich_norm else nn.Identity()
            # Define the Resisdual Weight for ReZero
            self.layer_drop = DropPath(p_layer_drop) if p_layer_drop > 0. else nn.Identity()
            # Define the Resisdual Weight for ReZero
            self.resweight = torch.nn.Parameter(torch.Tensor([0]), requires_grad=True) if is_rezero else 1


    def forward(self, msa, pair_orig, att = None):
        # Input: MSA embeddings (B, K, L, D), original pair embeddings (B, L, L, D)
        # Output: updated pair info (B, L, L, D)
        B, N, L, _ = msa.shape
        # project down to reduce memory
        msa = self.norm_1(msa)
        x_down = self.proj_1(msa) # (B, N, L, n_feat_proj)

        # get sequence weight
        x_down = self.norm_down(x_down)
        w_seq = self.encoder(x_down).reshape(B, L, 1, N).permute(0,3,1,2)
        feat_1d = w_seq * x_down

        pair = self.coevol(x_down, feat_1d)

        # average pooling over N of given MSA info
        feat_1d = feat_1d.sum(1)

        # query sequence info
        query = x_down[:, 0] # (B,L,K)
        feat_1d = torch.cat((feat_1d, query), dim=-1) # additional 1D features

        # tile 1D features
        left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)
        right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)

        # update original pair features through convolutions after concat
        pair_orig = self.norm_orig(pair_orig)
        pair = self.norm_new(pair)

        if att is None:
            pair = torch.cat((pair_orig, pair, left, right), -1)
        else:
            pair = torch.cat((pair_orig, pair, left, right, att), -1)

        if self.is_peg:
            pair = pair.permute(0,3,1,2).contiguous() # prep for convolution layer
            pair = self.update(pair)
            pair = pair.permute(0,2,3,1).contiguous() # (B, L, L, D)
        else:
            pair = pair_orig + self.resweight * self.layer_drop(self.update_postnorm(self.update(self.update_norm(pair))))

        return pair

class Pair2Pair(nn.Module):
    def __init__(self,
                 n_layer = 1,
                 n_att_head = 8,
                 n_feat = 128,
                 r_ff = 4,
                 p_drop = 0.1,
                 p_attn_drop = 0.,
                 p_layer_drop = 0.,
                 is_rezero = False,
                 n_layer_shift_tokens = 0,
                 pair2pair_type = 'PAB',
                 **kwargs):
        super(Pair2Pair, self).__init__()

        if isinstance(p_layer_drop, float):
            p_layer_drop = [p_layer_drop for i in range(n_layer)]

        if pair2pair_type == 'PAB':
            layers = [PairwiseAttentionBlock(pair_embed_dim = n_feat,
                                             heads = n_att_head,
                                             dim_head = n_feat // n_att_head,
                                             p_drop = p_drop,
                                             p_attn_drop = p_attn_drop,
                                             p_layer_drop = p_layer_drop[i],
                                             is_rezero = is_rezero,
                                             r_ff = r_ff,
                                             n_layer_shift_tokens = 0,
                                             **kwargs) for i in range(n_layer)]

        elif pair2pair_type == 'PABNew':
            layers = [PairwiseAttentionBlock(pair_embed_dim = n_feat,
                                             heads = n_att_head,
                                             dim_head = n_feat // n_att_head,
                                             p_drop = p_drop,
                                             p_attn_drop = p_attn_drop,
                                             p_layer_drop = p_layer_drop[i],
                                             is_rezero = is_rezero,
                                             r_ff = r_ff,
                                             n_layer_shift_tokens = 0,
                                             tri_new = True,
                                             **kwargs) for i in range(n_layer)]

        elif pair2pair_type == 'MAB':
            layers = [MsaAttentionBlock(msa_embed_dim = n_feat,
                                         heads = n_att_head,
                                         dim_head = n_feat // n_att_head,
                                         p_drop = p_drop,
                                         p_attn_drop = p_attn_drop,
                                         p_layer_drop = p_layer_drop[i],
                                         is_rezero = is_rezero,
                                         r_ff = r_ff,
                                         n_layer_shift_tokens = n_layer_shift_tokens,
                                         **kwargs) for i in range(n_layer)]
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x

class MSA2MSA(nn.Module):
    def __init__(self,
                 n_layer = 1,
                 n_heads = 8,
                 n_feats = 256,
                 r_ff = 4,
                 p_drop = 0.1,
                 p_attn_drop = 0.,
                 p_layer_drop = 0.,
                 is_rezero = False,
                 pair2pair_type = 'MAB',
                 is_global_query_attn = False,
                 **kwargs):
        super(MSA2MSA, self).__init__()

        if isinstance(p_layer_drop, float):
            p_layer_drop = [p_layer_drop for i in range(n_layer)]

        if pair2pair_type == 'MAB':
            layers = [MsaAttentionBlock(msa_embed_dim = n_feats,
                                         heads = n_heads,
                                         dim_head = n_feats // n_heads,
                                         p_drop = p_drop,
                                         p_attn_drop = p_attn_drop,
                                         p_layer_drop = p_layer_drop[i],
                                         is_rezero = is_rezero,
                                         r_ff = r_ff,
                                         is_global_query_attn = is_global_query_attn,
                                         **kwargs) for i in range(n_layer)]
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # Input: MSA embeddings (B, K, L, D)
        # Output: updated MSA embeddings (B, K, L, D)
        B, N, L, _ = x.shape

        x = self.encoder(x)

        return x

class Pair2MSA(nn.Module):
    def __init__(self,
                 n_att_head = 4,
                 n_feat_in = 128,
                 n_feat_out = 256,
                 r_ff = 4,
                 p_drop = 0.1,
                 p_attn_drop = 0.,
                 p_layer_drop = 0.,
                 is_rezero = False,
                 is_peg = False,
                 n_peg_block = 1,
                 pair2msa_type = 'direct',
                 pb_relax = False,
                 **kwargs):
        super(Pair2MSA, self).__init__()
        self.is_peg = is_peg

        self.pair2msa_type = pair2msa_type
        if self.pair2msa_type == 'direct':
            self.encoder = DirectEncoderLayer(heads=n_att_head,
                                              d_in=n_feat_in,
                                              d_out=n_feat_out,
                                              d_ff=n_feat_out * r_ff,
                                              p_drop=p_drop,
                                              p_attn_drop=p_attn_drop,
                                              p_layer_drop=p_layer_drop,
                                              is_rezero = is_rezero,
                                              pb_relax=pb_relax,
                                              )
        elif self.pair2msa_type == 'bias_fusion':
            self.encoder = MsaAttentionBlock(msa_embed_dim = n_feat_out,
                                             heads = n_att_head,
                                             dim_head = n_feat_in // n_att_head,
                                             pair_embed_dim = n_feat_in,
                                             p_drop = p_drop,
                                             p_attn_drop = p_attn_drop,
                                             p_layer_drop = p_layer_drop,
                                             is_rezero = is_rezero,
                                             r_ff = r_ff,
                                             pb_relax=pb_relax,
                                             **kwargs)
        else:
            raise NotImplementedError

        if self.is_peg:
            self.update = nn.Sequential(*[SEBottleneck(n_feat_out, n_feat_out, stride=1, dilation=1, is_depthwize=True)
                                          for i in range(n_peg_block)])

    def forward(self, pair, msa):

        if self.pair2msa_type == 'direct':
            msa = self.encoder(pair, msa)
        else:
            msa = self.encoder(msa, pair=pair)

        if self.is_peg:
            msa = msa.permute(0,3,1,2).contiguous() # prep for convolution layer
            msa = self.update(msa)
            msa = msa.permute(0,2,3,1).contiguous() # (B, L, L, D)

        return msa

class IterBlock(nn.Module):
    def __init__(self,
                 n_layer_pair = 1,
                 n_layer_msa = 1,
                 d_msa = 64,
                 d_pair = 128,
                 n_head_msa = 4,
                 n_head_pair = 8,
                 r_ff = 4,
                 p_drop = 0.,
                 p_attn_drop = 0.,
                 p_layer_drop = [],
                 is_rezero = False,
                 n_layer_shift_tokens = 0,
                 pair2msa_type = 'direct',
                 pair2pair_type = 'PAB',
                 msa2msa_type = 'MAB',
                 **kwargs):
        super(IterBlock, self).__init__()

        self.msa2msa = MSA2MSA(n_layer = n_layer_msa,
                               n_att_head = n_head_msa,
                               n_feats = d_msa,
                               r_ff = r_ff,
                               p_drop = p_drop,
                               p_attn_drop = p_attn_drop,
                               p_layer_drop = p_layer_drop[:n_layer_msa],
                               is_rezero = is_rezero,
                               n_layer_shift_tokens = n_layer_shift_tokens,
                               msa2msa_type = msa2msa_type,
                               **kwargs)

        self.msa2pair = MSA2Pair(n_feat = d_msa,
                                 n_feat_out = d_pair,
                                 n_feat_proj = 32,
                                 p_drop = p_drop,
                                 p_attn_drop = p_attn_drop,
                                 p_layer_drop = p_layer_drop[1],
                                 is_rezero = is_rezero,
                                 n_att_head = n_head_msa,
                                 **kwargs)

        self.pair2pair = Pair2Pair(n_layer = n_layer_pair,
                                   n_att_head = n_head_pair,
                                   n_feat = d_pair,
                                   r_ff = r_ff,
                                   p_drop = p_drop,
                                   p_attn_drop = p_attn_drop,
                                   p_layer_drop = p_layer_drop[1 + n_layer_msa :
                                                               1 + n_layer_msa + n_layer_pair],
                                   is_rezero=is_rezero,
                                   n_layer_shift_tokens = n_layer_shift_tokens,
                                   pair2pair_type = pair2pair_type,
                                   **kwargs)

        self.pair2msa = Pair2MSA(n_att_head = 4,
                                 n_feat_in = d_pair,
                                 n_feat_out = d_msa,
                                 r_ff = r_ff,
                                 p_drop = p_drop,
                                 p_attn_drop = p_attn_drop,
                                 p_layer_drop = p_layer_drop[-1],
                                 is_rezero = is_rezero,
                                 pair2msa_type= pair2msa_type,
                                 **kwargs)

    def forward(self, msa, pair):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)

        # 1. process MSA features
        msa = self.msa2msa(msa)

        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair)

        # 3. process pair features
        pair = self.pair2pair(pair)

        # 4. update MSA features using updated pair features
        msa = self.pair2msa(pair, msa)

        return msa, pair

class IterativeFeatureExtractor2D(nn.Module):
    def __init__(self,
                 n_module = 4,
                 n_layer_pair = 4,
                 n_layer_msa = 1,
                 d_msa = 256,
                 d_pair = 128,
                 n_head_msa = 8,
                 n_head_pair = 8,
                 r_ff = 4,
                 n_resblock = 1,
                 p_drop = 0.1,
                 p_attn_drop = 0.,
                 p_layer_drop = 0.,
                 is_p_layer_drop = False,
                 is_rezero = False,
                 use_checkpoint = True,
                 n_layer_shift_tokens = 0,
                 pair2msa_type = 'direct',
                 pair2pair_type = 'PAB',
                 msa2msa_type = 'MAB',
                 **kwargs):
        super(IterativeFeatureExtractor2D, self).__init__()

        self.n_module = n_module
        self.use_checkpoint = use_checkpoint
        # stochastic depth
        depths = n_module * (n_layer_pair + n_layer_msa + 2) + 1

        # stochastic depth decay rule
        if is_p_layer_drop:
            dpr = [x.item() for x in torch.linspace(0, p_layer_drop, depths)]
        else:
            dpr = [p_layer_drop for i in range(depths)]

        self.initial_pair = Pair2Pair( n_layer = 1,
                                       n_att_head = n_head_pair,
                                       n_feat = d_pair,
                                       r_ff = r_ff,
                                       p_drop = p_drop,
                                       p_attn_drop = p_attn_drop,
                                       p_layer_drop = dpr[0],
                                       is_rezero = is_rezero,
                                       n_layer_shift_tokens = n_layer_shift_tokens,
                                       pair2pair_type = pair2pair_type,
                                       **kwargs)

        self.initial_msa = MSA2MSA( n_layer = 1,
                                    n_heads = n_head_msa,
                                    n_feats = d_msa,
                                    p_drop = p_drop,
                                    p_attn_drop = p_attn_drop,
                                    n_layer_shift_tokens = n_layer_shift_tokens,
                                    p_layer_drop = dpr[1],
                                    **kwargs)

        self.iter_block_1 = nn.ModuleList([
                            IterBlock(n_layer_pair=n_layer_pair,
                                      n_layer_msa=n_layer_msa,
                                      d_msa=d_msa,
                                      d_pair=d_pair,
                                      n_head_msa=n_head_msa,
                                      n_head_pair=n_head_pair,
                                      r_ff=r_ff,
                                      n_resblock=n_resblock,
                                      p_drop=p_drop,
                                      p_attn_drop=p_attn_drop,
                                      p_layer_drop=dpr[ 2 + (n_layer_pair + n_layer_msa + 2) * i :
                                                        2 + (n_layer_pair + n_layer_msa + 2) * (i + 1) ],
                                      is_rezero=is_rezero,
                                      n_layer_shift_tokens = n_layer_shift_tokens,
                                      pair2msa_type = pair2msa_type,
                                      pair2pair_type = pair2pair_type,
                                      msa2msa_type = msa2msa_type,
                                      **kwargs) for i in range(self.n_module)])

    def forward_block(self, msa, pair, i):
        return self.iter_block_1[i](msa, pair)

    def forward(self, msa, pair):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)

        pair = self.initial_pair(pair)

        msa = self.initial_msa(msa)

        for i_m in range(self.n_module):
            # extract features from MSA & update original pair features
            if self.use_checkpoint:
                # print('use checkpoint')
                msa, pair = checkpoint(self.forward_block, msa, pair, i_m)
            else:
                msa, pair = self.iter_block_1[i_m](msa, pair)

        return msa, pair

class MSAExtraStack(nn.Module):
    def __init__(self,
                 d_msa = 64,
                 d_pair = 128,
                 n_head_msa = 4,
                 n_head_pair = 8,
                 r_ff = 4,
                 p_drop = 0.,
                 p_attn_drop = 0.,
                 is_rezero = False,
                 n_layer_shift_tokens = 0,
                 pair2msa_type = 'direct',
                 pair2pair_type = 'PAB',
                 msa2msa_type = 'MAB',
                 is_global_query_attn = True,
                 **kwargs):
        super(MSAExtraStack, self).__init__()

        n_layer_pair = 1
        n_layer_msa = 1

        self.msa2msa = MSA2MSA(n_layer = n_layer_msa,
                               n_att_head = n_head_msa,
                               n_feats = d_msa,
                               r_ff = r_ff,
                               p_drop = p_drop,
                               p_attn_drop = p_attn_drop,
                               is_rezero = is_rezero,
                               n_layer_shift_tokens = n_layer_shift_tokens,
                               msa2msa_type = msa2msa_type,
                               is_global_query_attn = is_global_query_attn,
                               **kwargs)

        self.msa2pair = MSA2Pair(n_feat = d_msa,
                                 n_feat_out = d_pair,
                                 n_feat_proj = 32,
                                 p_drop = p_drop,
                                 p_attn_drop = p_attn_drop,
                                 is_rezero = is_rezero,
                                 n_att_head = n_head_msa,
                                 **kwargs)

        self.pair2pair = Pair2Pair(n_layer = n_layer_pair,
                                   n_att_head = n_head_pair,
                                   n_feat = d_pair,
                                   r_ff = r_ff,
                                   p_drop = p_drop,
                                   p_attn_drop = p_attn_drop,
                                   is_rezero=is_rezero,
                                   n_layer_shift_tokens = n_layer_shift_tokens,
                                   pair2pair_type = pair2pair_type,
                                   **kwargs)

        self.pair2msa = Pair2MSA(n_att_head = 4,
                                 n_feat_in = d_pair,
                                 n_feat_out = d_msa,
                                 r_ff = r_ff,
                                 p_drop = p_drop,
                                 p_attn_drop = p_attn_drop,
                                 is_rezero = is_rezero,
                                 pair2msa_type = pair2msa_type,
                                 **kwargs)

    def forward(self, msa, pair):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)

        # 1. process MSA features
        msa = self.msa2msa(msa)

        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair)

        # 3. process pair features
        pair = self.pair2pair(pair)

        # 4. update MSA features using updated pair features
        msa = self.pair2msa(pair, msa)

        return msa, pair

if __name__ == '__main__':
    pair_fea = torch.zeros([1, 128, 128, 256]).cuda()
    msa_fea  = torch.zeros([1, 1024, 128, 64]).cuda()  # n,r,c,d

    net = MSAExtraStack(d_msa = 64,
                        d_pair = 256,
                        is_global_query_attn = False)

    net.cuda()
    print(net)
    while 1:
        msa, pair = net(msa_fea, pair_fea)
        print('msa', msa.shape)
        print('pair', pair.shape)


