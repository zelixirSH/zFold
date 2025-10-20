"""The invariant point attention (IPA) module."""

import numpy as np
import torch
from torch import nn

from zfold.network.af2_smod.utils import apply_trans
from zfold.network.af2_smod.utils import quat2rot
from zfold.network.af2_smod.utils import rot2quat
from zfold.network.attention import LayerNorm, FeedForwardLayer

class InvPntAttn(nn.Module):
    """The invariant point attention (IPA) module."""

    def __init__(
            self,
            n_dims_sfea=384,  # number of dimensions in single features
            n_dims_pfea=256,  # number of dimensions in pair features
            n_dims_attn=16,   # number of dimensions in query/key/value embeddings
            n_heads=12,       # number of attention heads
            n_qpnts=4,        # number of points for query embeddings
            n_vpnts=8,        # number of points for value embeddings
            v2 = False,
            quat_type=None,  # type of quaternion vectors (choices: 'full' or 'part')
        ):
        """Constructor function."""

        super().__init__()

        # setup hyper-parameters
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_attn = n_dims_attn
        self.n_heads = n_heads
        self.n_qpnts = n_qpnts
        self.n_vpnts = n_vpnts

        # setup additional configurations
        self.quat_type = quat_type
        self.quat2rot = quat2rot_full if self.quat_type == 'full' else quat2rot
        self.rot2quat = rot2quat_full if self.quat_type == 'full' else rot2quat
        self.n_dims_quat = 4 if self.quat_type == 'full' else 3  # DO NOT MODIFY!

        # setup additional configurations
        self.n_dims_cord = 3  # THIS MUST NOT BE MODIFIED
        self.n_dims_shid = self.n_heads * \
            (self.n_dims_pfea + self.n_dims_attn + self.n_vpnts * 3 + self.n_vpnts)
        self.wc = np.sqrt(2.0 / (9.0 * self.n_qpnts))
        self.wl = np.sqrt(1.0 / 3.0)
        self.ws = 1.0  # np.log(np.exp(1.0) - 1.0)  # so that its softplus output equals to 1

        # sub-networks - Invariant Point Attention
        # print(self.n_dims_attn)
        self.linear_q = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_k = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_v = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn, bias=False)

        self.linear_qp = nn.Linear(
            self.n_dims_sfea, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
        self.linear_kp = nn.Linear(
            self.n_dims_sfea, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
        self.linear_vp = nn.Linear(
            self.n_dims_sfea, self.n_heads * self.n_vpnts * self.n_dims_cord, bias=False)

        self.linear_b = nn.Linear(self.n_dims_pfea, self.n_heads, bias=False)
        self.linear_s = nn.Linear(self.n_dims_shid, self.n_dims_sfea)

        self.register_parameter(
            name='scale', param=nn.Parameter(self.ws * torch.ones((self.n_heads))))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=2)

        # sub-networks - Feed-Forward Network
        p_drop = 0.0
        self.v2 = v2
        if self.v2:
            activation = 'relu'
            is_use_ln = True
            is_post_act_ln = False
            is_sandwich_norm = True
            r_ff = 4
            pb_relax = False

            self.pre_ln = LayerNorm(self.n_dims_sfea, pb_relax = pb_relax) if is_use_ln else nn.Identity()
            self.ffn_ln = LayerNorm(self.n_dims_sfea, pb_relax = pb_relax) if is_use_ln else nn.Identity()
            self.ffn_postln = LayerNorm(self.n_dims_sfea, pb_relax = pb_relax) if is_sandwich_norm else nn.Identity()
            self.ffn = FeedForwardLayer(self.n_dims_sfea,
                                        self.n_dims_sfea*r_ff,
                                        p_drop = p_drop,
                                        is_post_act_ln = is_post_act_ln,
                                        activation=activation,
                                        )
        else:
            self.drop_1 = nn.Dropout(p=p_drop)
            self.norm_1 = nn.LayerNorm(self.n_dims_sfea)
            self.mlp = nn.Sequential(
                nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
                nn.ReLU(),
                nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
                nn.ReLU(),
                nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
            )
            self.drop_2 = nn.Dropout(p=p_drop)
            self.norm_2 = nn.LayerNorm(self.n_dims_sfea)


    def forward(self, sfea_tns, pfea_tns, quat_tns, trsl_tns):
        """Perform the forward pass.

        Args:
        * sfea_tns: single features of size N x L x D_s
        * pfea_tns: pair features of size N x L x L x D_p
        * quat_tns: partial quaternion vectors of size N x L x 3
        * trsl_tns: translation vectors of size N x L x 3

        Returns:
        * sfea_tns: single features of size N x L x D_s
        """

        if self.v2:
            sfea_tns_ = sfea_tns
            sfea_tns = self.pre_ln(sfea_tns)

        # initialization
        n_smpls, n_resds, _ = sfea_tns.shape

        # calculate query/key/value embeddings
        q_tns = self.linear_q(sfea_tns).view(n_smpls, n_resds, 1, self.n_heads, self.n_dims_attn)
        k_tns = self.linear_k(sfea_tns).view(n_smpls, 1, n_resds, self.n_heads, self.n_dims_attn)
        v_tns = self.linear_v(sfea_tns).view(n_smpls, n_resds, self.n_heads, self.n_dims_attn)
        qp_tns = self.linear_qp(sfea_tns).view(n_smpls, n_resds, self.n_heads, self.n_qpnts, self.n_dims_cord)
        kp_tns = self.linear_kp(sfea_tns).view(n_smpls, n_resds, self.n_heads, self.n_qpnts, self.n_dims_cord)
        vp_tns = self.linear_vp(sfea_tns).view(n_smpls, n_resds, self.n_heads, self.n_vpnts, self.n_dims_cord)
        b_tns = self.linear_b(pfea_tns).view(n_smpls, n_resds, n_resds, self.n_heads)

        ###use fp32 calculation
        q_tns = q_tns.float()
        k_tns = k_tns.float()
        v_tns = v_tns.float()
        qp_tns = qp_tns.float()
        kp_tns = kp_tns.float()
        vp_tns = vp_tns.float()
        b_tns = b_tns.float()
        tns_type = sfea_tns.dtype
        # print('convert to fp32', tns_type)

        # apply global transformation on Q/K/V points
        rot_tns = self.quat2rot(quat_tns.view(-1, self.n_dims_quat)).view(n_smpls, n_resds, 3, 3)

        qp_tns_proj = apply_trans(qp_tns, rot_tns, trsl_tns, grouped=True).view(
            n_smpls, n_resds, 1, self.n_heads, self.n_qpnts, 3)
        kp_tns_proj = apply_trans(kp_tns, rot_tns, trsl_tns, grouped=True).view(
            n_smpls, 1, n_resds, self.n_heads, self.n_qpnts, 3)
        vp_tns_proj = apply_trans(vp_tns, rot_tns, trsl_tns, grouped=True).view(
            n_smpls, n_resds, self.n_heads, self.n_vpnts, 3)

        # compute attention weights
        qk_tns = torch.sum(q_tns * k_tns, dim=-1) / np.sqrt(self.n_dims_attn)  # N x L x L x H
        qkp_tns = self.wc * self.softplus(self.scale).view(1, 1, 1, -1) / 2.0 * torch.sum(
            torch.square(torch.norm(qp_tns_proj - kp_tns_proj, dim=-1)), dim=-1)  # N x L x L x H
        a_tns = self.softmax(self.wl * (qk_tns + b_tns - qkp_tns))  # N x L x L x H

        # update single features
        op_tns = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            pfea_tns.view(n_smpls, n_resds, n_resds, 1, self.n_dims_pfea)
        , dim=2)  # N x L x H x D_p
        ov_tns = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            v_tns.view(n_smpls, 1, n_resds, self.n_heads, self.n_dims_attn)
        , dim=2)  # N x L x H x D_a
        ovp_tns_proj = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            vp_tns_proj.view(n_smpls, 1, n_resds, self.n_heads, self.n_vpnts * 3)
        , dim=2)  # N x L x H x (P_v x 3)

        ovp_tns = apply_trans(
            ovp_tns_proj, rot_tns, trsl_tns, grouped=True, reverse=True,
        ).view(n_smpls, n_resds, self.n_heads, self.n_vpnts * 3)  # N x L x H x (P_v x 3)
        ovp_tns_norm = torch.norm(
            ovp_tns.view(n_smpls, n_resds, self.n_heads, self.n_vpnts, 3), dim=-1)  # N x L x H x P_v
        shid_tns = torch.cat([op_tns, ov_tns, ovp_tns, ovp_tns_norm], dim=3)  # N x L x (H x D_h')

        shid_tns = shid_tns.to(tns_type)

        # pass single features through a feed-forward network
        if self.v2:
            sfea_tns = self.linear_s(shid_tns.view(n_smpls, n_resds, self.n_dims_shid)) + sfea_tns_
            sfea_tns = self.ffn_postln(self.ffn(self.ffn_ln(sfea_tns))) + sfea_tns
        else:
            sfea_tns = sfea_tns + self.linear_s(shid_tns.view(n_smpls, n_resds, self.n_dims_shid))
            sfea_tns = self.norm_1(self.drop_1(sfea_tns))
            sfea_tns = sfea_tns + self.mlp(sfea_tns)
            sfea_tns = self.norm_2(self.drop_2(sfea_tns))

        return sfea_tns
