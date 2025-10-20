"""The network for updating per-residue local frames & torsion angles."""

import torch
from torch import nn

from zfold.loss.fape.constants import N_ANGLS_PER_RESD_MAX
from zfold.network.af2_smod.utils import quat2rot
from zfold.network.af2_smod.utils import rot2quat
from zfold.network.attention import LayerNorm

class FramAnglNet(nn.Module):
    """The network for updating per-residue local frames & torsion angles."""

    def __init__(
            self,
            n_dims_sfea=384,  # number of dimensions in single features
            n_dims_hidd=128,  # number of dimensions in hidden features
            quat_type = None,
            v2 = True,
        ):
        """Constructor function."""

        super().__init__()

        self.v2 = v2
        if self.v2:
            self.pre_ln = LayerNorm(n_dims_sfea)

        # setup hyper-parameters
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_hidd = n_dims_hidd

        # setup additional configurations
        self.n_angls = N_ANGLS_PER_RESD_MAX
        self.n_dims_angl = 2 * self.n_angls


        # setup additional configurations
        self.quat_type = quat_type
        self.quat2rot = quat2rot_full if self.quat_type == 'full' else quat2rot
        self.rot2quat = rot2quat_full if self.quat_type == 'full' else rot2quat
        self.n_dims_quat = 4 if self.quat_type == 'full' else 3  # DO NOT MODIFY!

        # sub-networks for updating per-residue local frames
        self.linear_quat = nn.Linear(self.n_dims_sfea, self.n_dims_quat)
        self.linear_trsl = nn.Linear(self.n_dims_sfea, 3)

        # sub-networks for updating per-residue torsion angles
        self.net_angl = nn.ModuleDict()
        self.net_angl['sfea'] = nn.Sequential(
            #nn.ReLU(),  # disabling these ReLUs seems to be helpful, but are enabled in AF2
            nn.Linear(self.n_dims_sfea, self.n_dims_hidd),
        )
        self.net_angl['sfea-i'] = nn.Sequential(
            #nn.ReLU(),  # disabling these ReLUs seems to be helpful, but are enabled in AF2
            nn.Linear(self.n_dims_sfea, self.n_dims_hidd),
        )
        self.net_angl['mlp-1'] = nn.Sequential(
            #nn.ReLU(),  # disabling these ReLUs seems to be helpful, but are enabled in AF2
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
        )
        self.net_angl['mlp-2'] = nn.Sequential(
            #nn.ReLU(),  # disabling these ReLUs seems to be helpful, but are enabled in AF2
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, self.n_dims_hidd),
        )
        self.net_angl['mlp-3'] = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.n_dims_hidd, self.n_dims_angl),
        )


    def forward(self, sfea_tns, sfea_tns_init, quat_tns, trsl_tns):
        """Perform the forward pass.

        Args:
        * sfea_tns: single features of size N x L x D_s
        * sfea_tns_init: initial single features of size N x L x D_s
        * quat_tns: partial quaternion vectors of size N x L x 3
        * trsl_tns: translation vectors of size N x L x 3

        Returns:
        * quat_tns: partial quaternion vectors of size N x L x 3
        * trsl_tns: translation vectors of size N x L x 3
        * angl_tns: torsion angle matrices of size N x L x K x 2
        """

        if self.v2:
            sfea_tns = self.pre_ln(sfea_tns)

        # initialization
        n_smpls, n_resds, _ = sfea_tns.shape

        # update per-residue local frames
        quat_tns_upd = self.linear_quat(sfea_tns)
        trsl_tns_upd = self.linear_trsl(sfea_tns)

        quat_tns, trsl_tns = self.__upd_quat_n_trsl(quat_tns, trsl_tns, quat_tns_upd, trsl_tns_upd)

        # update per-residue torsion angles
        if self.v2:
            hfea_tns = self.net_angl['sfea'](sfea_tns) #+ self.net_angl['sfea-i'](sfea_tns_init)
        else:
            hfea_tns = self.net_angl['sfea'](sfea_tns) + self.net_angl['sfea-i'](sfea_tns_init)

        hfea_tns = hfea_tns + self.net_angl['mlp-1'](hfea_tns)
        hfea_tns = hfea_tns + self.net_angl['mlp-2'](hfea_tns)
        angl_tns = self.net_angl['mlp-3'](hfea_tns).view(n_smpls, n_resds, self.n_angls, 2)

        return quat_tns, trsl_tns, angl_tns


    def __upd_quat_n_trsl(self, quat_tns_old, trsl_tns_old, quat_tns_upd, trsl_tns_upd):
        """Update per-residue local frames's partial quaternion & translation vectors."""

        # initialization
        n_smpls, n_resds, _ = quat_tns_old.shape

        # update partial quaternion vectors
        rot_tns_old = self.quat2rot(quat_tns_old.view(-1, self.n_dims_quat))
        rot_tns_upd = self.quat2rot(quat_tns_upd.view(-1, self.n_dims_quat))

        rot_tns_new = torch.bmm(rot_tns_old, rot_tns_upd)

        quat_tns_new = self.rot2quat(rot_tns_new).view(n_smpls, n_resds,  self.n_dims_quat)

        # update translation vectors
        trsl_tns_new = trsl_tns_old + \
            torch.bmm(rot_tns_old, trsl_tns_upd.view(-1, 3, 1)).view(n_smpls, n_resds, 3)

        return quat_tns_new, trsl_tns_new
