"""The embedding network of <XFoldNet> & <AF2SModNet> modules' outputs for recycling."""

import torch
from torch import nn

from zfold.network.af2_smod.prot_struct import ProtStruct
from zfold.dataset.tools.prot_constants import N_ATOMS_PER_RESD_MAX
from zfold.dataset.utils.math_utils import cdist

class RcEmbedNet(nn.Module):
    """The embedding network of <XFoldNet> & <AF2SModNet> modules' outputs for recycling."""

    def __init__(
            self,
            n_dims_mfea=384,  # number of dimensions in MSA features
            n_dims_pfea=256,  # number of dimensions in pair features
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_mfea = n_dims_mfea
        self.n_dims_pfea = n_dims_pfea

        # additional configurations
        self.n_bins = 18
        self.dist_min = 3.375
        self.dist_max = 21.375
        self.bin_wid = (self.dist_max - self.dist_min) / self.n_bins

        # build the initial mapping for single features
        self.norm_m = nn.LayerNorm(self.n_dims_mfea)
        self.norm_p = nn.LayerNorm(self.n_dims_pfea)
        self.linear = nn.Linear(self.n_bins, self.n_dims_pfea)

        # initialize model weights to zeros, so that pre-trained XFold models are unaffected
        nn.init.zeros_(self.norm_m.weight)
        nn.init.zeros_(self.norm_m.bias)
        nn.init.zeros_(self.norm_p.weight)
        nn.init.zeros_(self.norm_p.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


    def forward(self, aa_seq, mfea_tns, pfea_tns, rc_inputs=None):
        """Perform the forward pass.

        Args:
        * aa_seq: amino-acid sequence
        * mfea_tns: MSA features of size N x K x L x D_m
        * pfea_tns: pair features of size N x L x L x D_p
        * rc_inputs: (optional) dict of additional inputs for recycling embeddings
          > sfea: single features of size N x L x D_m
          > pfea: pair features of size N x L x L x D_p
          > cord: per-atom 3D coordinates of size L x M x 3

        Returns:
        * mfea_tns: updated MSA features of size N x K x L x D_m
        * pfea_tns: updated pair features of size N x L x L x D_p
        """

        # initialization
        n_resds = len(aa_seq)
        dtype = mfea_tns.dtype  # for compatibility w/ half-precision inputs
        device = mfea_tns.device

        # initialize additional inputs for recycling embeddings
        if rc_inputs is None:
            rc_inputs = {
                'sfea': torch.zeros((1, n_resds, self.n_dims_mfea), dtype=dtype, device=device),
                'pfea': torch.zeros((1, n_resds, n_resds, self.n_dims_pfea), dtype=dtype, device=device),
                'cord': torch.zeros((n_resds, N_ATOMS_PER_RESD_MAX, 3), dtype=dtype, device=device),
            }

        # calculate the pairwise distance between CB atoms (CA for Glycine)
        atom_names = ['CA', 'CB']
        cmsk_mat = torch.tensor(
            [[1, 0] if x == 'G' else [0, 1] for x in aa_seq], dtype=torch.int8, device=device)
        cord_tns_raw = ProtStruct.get_atoms(aa_seq, rc_inputs['cord'], atom_names)  # L x 2 x 3
        cord_mat_pcb = torch.sum(cmsk_mat.unsqueeze(dim=2) * cord_tns_raw, dim=1)  # pseudo CB

        dist_mat = cdist(cord_mat_pcb.float()).to(dtype)

        # calculate update terms for single features
        sfea_tns_rc = self.norm_m(rc_inputs['sfea'])

        # calculate update terms for pair features
        idxs_mat = torch.clip(torch.floor(
            (dist_mat - self.dist_min) / self.bin_wid).to(torch.int64), 0, self.n_bins - 1)
        onht_tns = nn.functional.one_hot(idxs_mat, self.n_bins).unsqueeze(dim=0)
        pfea_tns_rc = self.norm_p(rc_inputs['pfea']) + self.linear(onht_tns.to(dtype))

        # update MSA & pair features
        mfea_tns = torch.cat(
            [(mfea_tns[:, 0] + sfea_tns_rc).unsqueeze(dim=1), mfea_tns[:, 1:]], dim=1)
        pfea_tns = pfea_tns + pfea_tns_rc

        return mfea_tns, pfea_tns


    def __repr__(self):
        """Get the string-formatted representation."""

        repr_str = 'RcEmbedNet(%s)' % ', '.join([
            'n_dims_mfea=%d' % self.n_dims_mfea,
            'n_dims_pfea=%d' % self.n_dims_pfea,
        ])

        return repr_str

if __name__ == '__main__':
    net = RcEmbedNet()
    print(net)
