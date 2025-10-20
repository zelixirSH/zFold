"""The AlphaFold2 structure module network."""

import torch
from torch import nn
from zfold.network.af2_smod.af2_smod import AF2SMod

class AF2SModNet(nn.Module):
    """The AlphaFold2 structure module network."""

    def __init__(
            self,
            n_lyrs=2,         # number of layers
            n_dims_mfea=384,  # number of dimensions in MSA features
            n_dims_pfea=256,  # number of dimensions in pair features
            n_dims_attn=16,
            v2 = False,
            plddt=False,
            highres=False,
            finalloss=True,
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_lyrs = n_lyrs
        self.n_dims_mfea = n_dims_mfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_attn = n_dims_attn

        # additional configurations
        self.n_dims_sfea = n_dims_mfea  # same number of dimension in MSA & single features

        # build the initial mapping for single features
        self.linear = nn.Linear(self.n_dims_mfea, self.n_dims_sfea)

        self.af2_smod = AF2SMod(
            n_lyrs=self.n_lyrs,
            n_dims_sfea=self.n_dims_sfea,
            n_dims_pfea=self.n_dims_pfea,
            n_dims_attn=self.n_dims_attn,
            v2 = v2,
            plddt = plddt,
            highres = highres,
            finalloss = finalloss,
        )

    def dtype(self):
        return self.linear.weight.dtype

    def forward(self, sfea_tns, pfea_tns, params_init=None, n_lyrs_sto=-1, cords_fb=True, aa_seq=None):
        """Perform the forward pass.

        Args:
        * mfea_tns: MSA features of size N x K x L x D_m
        * pfea_tns: pair features of size N x L x L x D_p
        * params_init: (optional) initial QTA parameters
        * n_lyrs_sto: (optional) number of <AF2SMod> layer for stochastic depth (-1: unlimited)
        * helper: (optional) <FapeLossHelper> object for reconstructing 3D coordinates

        Returns:
        * params_list: list of QTA parameters, one per layer
          > quat_tns: partial quaternion vectors of size N x L x 3
          > trsl_tns: translation vectors of size N x L x 3
          > angl_tns: torsion angle matrices of size N x L x K x 2
        * cord_tns: (optional) per-atom 3D coordinates of size L x M x 3
        """

        # initialization
        n_smpls, n_resds, _ = sfea_tns.shape

        # obtain single features
        sfea_tns = self.linear(sfea_tns)

        # perform the forward pass w/ <AF2SMod> module
        params_list, highres_list, plddt_list, cord_list, fram_tns_sc = \
            self.af2_smod(sfea_tns, pfea_tns, params_init, n_lyrs_sto, cords_fb = cords_fb, aa_seq = aa_seq)

        return (params_list, highres_list, plddt_list, cord_list, fram_tns_sc)


    def __repr__(self):
        """Get the string-formatted representation."""

        repr_str = 'AF2SModNet(%s)' % ', '.join([
            'n_lyrs=%d' % self.n_lyrs,
            'n_dims_mfea=%d' % self.n_dims_mfea,
            'n_dims_pfea=%d' % self.n_dims_pfea,
        ])

        return repr_str
