"""The AlphaFold2 structure module."""

import torch
from torch import nn

from zfold.network.af2_smod.net.fram_angl_net import FramAnglNet
from zfold.network.af2_smod.net.inv_pnt_attn import InvPntAttn
from zfold.network.af2_smod.prot_converter import ProtConverter

class AF2SMod(nn.Module):
    """The AlphaFold2 structure module."""

    def __init__(
            self,
            n_lyrs=2,         # number of layers
            n_dims_sfea=384,  # number of dimensions in single features
            n_dims_pfea=256,  # number of dimensions in pair features
            n_dims_attn=16,
            v2 = False,
            plddt = False,
            highres = False,
            finalloss = True,
        ):
        """Constructor function."""

        super().__init__()

        # setup hyper-parameters
        self.n_lyrs = n_lyrs
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_attn = n_dims_attn
        self.finalloss = finalloss

        self.converter = ProtConverter()

        # build sub-networks
        self.net = nn.ModuleDict()
        self.net['norm_s'] = nn.LayerNorm(self.n_dims_sfea)
        self.net['norm_p'] = nn.LayerNorm(self.n_dims_pfea)
        self.net['linear_s'] = nn.Linear(self.n_dims_sfea, self.n_dims_sfea)

        self.net['ipa'] = InvPntAttn(
            n_dims_sfea=self.n_dims_sfea,
            n_dims_pfea=self.n_dims_pfea,
            n_dims_attn=self.n_dims_attn,
            v2 = v2,
        )
        self.net['fa'] = FramAnglNet(
            n_dims_sfea=self.n_dims_sfea,
            v2 = v2,
        )

        self.plddt = plddt
        if self.plddt:
            self.n_bins_lddt = 50  # number of bins for pLDDT-Ca predictions
            self.bin_vals = (torch.arange(self.n_bins_lddt).view(1, 1, -1) + 0.5) / self.n_bins_lddt
            # per-residue lDDT-Ca predictions
            self.net['lddt'] = nn.Sequential(
                nn.LayerNorm(self.n_dims_sfea),
                nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
                nn.ReLU(),
                nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
                nn.ReLU(),
                nn.Linear(self.n_dims_sfea, self.n_bins_lddt),
            )
            self.net['sfmx'] = nn.Softmax(dim=2)

        self.highres = highres
        if self.highres:
            # per-residue lDDT-Ca predictions
            self.net['hr'] = nn.Sequential(
                nn.LayerNorm(self.n_dims_sfea),
                nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
                nn.ReLU(),
                nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
                nn.ReLU(),
                nn.Linear(self.n_dims_sfea, 2),
            )

    def forward(self, sfea_tns, pfea_tns, params_init=None, n_lyrs_sto=-1, cords_fb = False, aa_seq = None):
        """Perform the forward pass.

        Args:
        * sfea_tns: single features of size N x L x D_s
        * pfea_tns: pair features of size N x L x L x D_p
        * params_init: (optional) initial QTA parameters
        * n_lyrs_sto: (optional) number of <AF2SMod> layer for stochastic depth (-1: unlimited)

        Returns:
        * params_list: list of QTA parameters, one per layer
          > quat_tns: partial quaternion vectors of size N x L x 3
          > trsl_tns: translation vectors of size N x L x 3
          > angl_tns: torsion angle matrices of size N x L x K x 2
        """

        # initialization
        device = sfea_tns.device
        n_smpls, n_resds, _ = sfea_tns.shape
        n_lyrs_act = self.n_lyrs if n_lyrs_sto == -1 else n_lyrs_sto  # actual number of layers

        # pre-process single & pair features
        sfea_tns_init = self.net['norm_s'](sfea_tns)
        pfea_tns = self.net['norm_p'](pfea_tns)
        sfea_tns = self.net['linear_s'](sfea_tns_init)

        # initialize partial quaternion & translation vectors
        if params_init is None:
            quat_tns = torch.zeros((n_smpls, n_resds, self.net['fa'].n_dims_quat), dtype=sfea_tns.dtype, device=device)
            trsl_tns = torch.zeros((n_smpls, n_resds, 3), dtype=sfea_tns.dtype, device=device)
        else:
            quat_tns = params_init[0].detach()  # no rotation gradients between iterations
            trsl_tns = params_init[1]

        # perform the forward pass
        params_list, plddt_list, highres_list, cord_list = [], [], [], []
        fram_tns_sc = None

        for idx_lyr in range(n_lyrs_act):
            # perform the forward pass w/ a single structure module
            sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns)  # A => nm?
            quat_tns, trsl_tns, angl_tns = self.net['fa'](sfea_tns, sfea_tns_init, quat_tns, trsl_tns)

            # reconstruct per-atom 3D coordinates
            if cords_fb:
                params = {'quat': quat_tns[0], 'trsl': trsl_tns[0], 'angl': angl_tns[0]}
                cord_tns, cmsk_mat, fram_tns_sc = \
                    self.converter.param2cord(aa_seq, params, atom_set='fa', rtn_cmsk=True)
                cord_list.append([cord_tns, cmsk_mat])

            if self.plddt:
                # predict per-residue & full-chain lDDT-Ca scores
                plddt = self.net['lddt'](sfea_tns)
                self.bin_vals = self.bin_vals.to(plddt.device)
                #glddt = torch.mean(torch.sum(self.bin_vals * self.net['sfmx'](plddt), dim=2), dim=1)
                clddt = torch.sum(self.bin_vals * self.net['sfmx'](plddt), dim=2)
                glddt = torch.mean(clddt)

                # original plddt (1, L, 50), global lddt (1, ), per-Ca lddt (L, )
                plddt_list.append([plddt, glddt, clddt])

            if self.highres:
                highres = self.net['hr'](sfea_tns)
                highres_list.append(highres)

            # record current QTA parameters, and stop gradients between iterations
            params_list.append((quat_tns, trsl_tns, angl_tns))

            if idx_lyr != n_lyrs_act - 1:
                quat_tns = quat_tns.detach()  # no rotation gradients between iterations

        if self.finalloss:
            return params_list[-1:], highres_list[-1:], plddt_list[-1:], cord_list[-1:], fram_tns_sc
        else:
            return params_list, highres_list, plddt_list, cord_list, fram_tns_sc
