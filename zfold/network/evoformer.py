from zfold.network.modules import *

class Evoformer(nn.Module):
    def __init__(
        self,
        pair_embed_dim,
        pair_attn_heads,
        msa_embed_dim,
        msa_attn_heads,
        depth,
        dim_head,
        attn_drop = 0.,
        p_drop = 0.,
        layer_drop = 0.,
        global_col_attn = False,
    ):
        super().__init__()
        self.use_checkpoint = True
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MsaAttentionBlock(msa_embed_dim = msa_embed_dim, heads = msa_attn_heads, dim_head = dim_head,
                                  pair_embed_dim = pair_embed_dim, global_col_attn = global_col_attn,
                                  attn_drop = attn_drop, p_drop = p_drop, layer_drop = layer_drop,
                                  is_ffn = True),
                PairwiseAttentionBlock(pair_embed_dim = pair_embed_dim, msa_embed_dim = msa_embed_dim,
                                       heads = pair_attn_heads, dim_head = dim_head, edge_layernorm = True,
                                       attn_drop=attn_drop, p_drop=p_drop, layer_drop=layer_drop,
                                       is_ffn = True),
            ]))

    def forward_block(self, msa_fea, pair_fea, mask, msa_mask, i):
        msa_attn, pair_attn = self.layers[i]
        # pairwise attention and transition
        msa_fea = msa_attn(msa_fea, mask=msa_mask, pair=pair_fea)
        # pairwise attention and transition
        pair_fea = pair_attn(pair_fea, mask=mask, msa_repr=msa_fea)

        return msa_fea, pair_fea

    def forward(
        self,
        msa_fea,
        pair_fea,
        mask = None,
        msa_mask = None
    ):

        for i in range(len(self.layers)):
            if self.use_checkpoint:
                msa_fea, pair_fea = checkpoint(self.forward_block, msa_fea, pair_fea, mask, msa_mask, i)
            else:
                msa_fea, pair_fea = self.forward_block(msa_fea, pair_fea, mask, msa_mask, i)

        return msa_fea, pair_fea

if __name__ == '__main__':
    import torch
    import numpy as np
    param = {
            'msa_embed_dim': 384,
            'pair_embed_dim': 288,
            'depth': 4,
            'msa_attn_heads':12,
            'pair_attn_heads':8,
            'dim_head': 64,
            'attn_drop': .1,
            'p_drop': .1,
            'global_col_attn':True
            }
    net = Evoformer(**param)
    print(net)
    B, K, L, D = 1, 16, 32, 256
    x = torch.FloatTensor(np.zeros([B,L,L,288]))
    m = torch.FloatTensor(np.zeros([B,K,L,384]))
    m, x = net.forward(m, x)
    print(x.shape, m.shape)
