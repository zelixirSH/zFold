import torch
import torch.nn as nn
from zfold.network.attention import LayerNorm, FeedForwardLayer

# predict distance map from pair features
class PairPredictor(nn.Module):
    def __init__(self,
                 n_feat,
                 p_drop = 0.0,
                 bins = [37,25,25,25],
                 is_use_ln = True,
                 pb_relax = False,
                 **kwargs):
        super(PairPredictor, self).__init__()
        self.norm = LayerNorm(n_feat, pb_relax=pb_relax) if is_use_ln else nn.Sequential()
        self.proj = nn.Linear(n_feat, n_feat)
        self.drop = nn.Dropout(p_drop)
        self.resnet_dist = FeedForwardLayer(d_model=n_feat, d_ff = n_feat*4, d_model_out=bins[0], p_drop=p_drop, **kwargs)
        self.resnet_omega = FeedForwardLayer(d_model=n_feat, d_ff = n_feat*4, d_model_out=bins[1], p_drop=p_drop, **kwargs)
        self.resnet_theta = FeedForwardLayer(d_model=n_feat, d_ff = n_feat*4, d_model_out=bins[2], p_drop=p_drop, **kwargs)
        self.resnet_phi = FeedForwardLayer(d_model=n_feat, d_ff = n_feat*4, d_model_out=bins[3], p_drop=p_drop, **kwargs)

    def forward(self, x):
        # input: pair info (B, L, L, C)
        x = self.norm(x)
        x = self.drop(self.proj(x))

        # predict theta, phi (non-symmetric)
        logits_theta = self.resnet_theta(x).permute(0, 3, 1, 2)
        logits_phi = self.resnet_phi(x).permute(0, 3, 1, 2)
        # predict dist, omega
        x = 0.5 * (x + x.permute(0, 2, 1, 3))
        logits_dist = self.resnet_dist(x).permute(0, 3, 1, 2)
        logits_omega = self.resnet_omega(x).permute(0, 3, 1, 2)

        return logits_dist, logits_omega, logits_theta, logits_phi
