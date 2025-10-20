import torch
import torch.nn as nn
from zfold.network.attention import LayerNorm

class CoevolExtractor(nn.Module):
    def __init__(self, n_feat_proj, n_feat_out,
                 p_drop = 0.1,
                 use_ln = True,
                 pb_relax = False):
        super(CoevolExtractor, self).__init__()

        self.norm_2d = LayerNorm(n_feat_proj * n_feat_proj, pb_relax = pb_relax) if use_ln else nn.Sequential()
        # project down to output dimension (pair feature dimension)
        self.proj_2 = nn.Linear(n_feat_proj ** 2, n_feat_out)

    def forward(self, x_down, x_down_w = None):
        B, N, L = x_down.shape[:3]

        if x_down_w is None:
            x_down_w = x_down

        #TODO average pool needs to be improved
        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down_w)  # outer-product & average pool
        pair = pair.reshape(B, L, L, -1)
        pair = self.norm_2d(pair)
        pair = self.proj_2(pair)  # (B, L, L, n_feat_out) # project down to pair dimension
        return pair
