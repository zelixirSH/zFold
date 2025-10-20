"""The built-in assessor for per-residue lDDT scores."""

import os

import torch

from zfold.dataset.utils.math_utils import cdist
from zfold.network.af2_smod.prot_struct import ProtStruct

class LddtAssessor():
    """The built-in assessor for lDDT scores.

    Note:
    * For simplicity, we do not consider the naming issue of partially symmetric residues.
    """

    def __init__(self):
        """Constructor function."""

        self.eps = 1e-6
        self.radius = 15.0  # inclusion radius
        self.dthr_tns = torch.tensor([0.5, 1.0, 2.0, 4.0], dtype=torch.float32).view(1, 1, -1)


    def run(self, cord_tns_ref, cord_tns_qry, cmsk_mat, atom_set='ca'):
        """Run the built-in assessor to compute per-residue lDDT-Ca scores.

        Args:
        * cord_tns_ref: reference structure's per-atom 3D coordinates of size L x M x 3
        * cord_tns_qry: query structure's per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: reference structure's per-atom 3D coordinates' validness masks of size L x M
        * atom_set: (optional) atom set over which to compute per-residue lDDT scores

        Returns:
        * plddt_vec: per-residue lDDT-Ca scores of size L
        * plmsk_vec: per-residue lDDT-Ca scores' validness masks of size L
        * clddt_val: full-chain lDDT-Ca score of size 1
        """

        # initializations
        device = cord_tns_ref.device
        n_resds, n_atoms, _ = cord_tns_ref.shape
        aa_seq = 'A' * n_resds  # does not matter

        # move <self.dthr_tns> to the correct device
        if self.dthr_tns.device != device:
            self.dthr_tns = self.dthr_tns.to(device)

        # extract 3D coordinates for CA-atom or full-atom
        if atom_set == 'ca':
            cord_mat_ref = ProtStruct.get_atoms(aa_seq, cord_tns_ref, ['CA'])  # L x 3
            cord_mat_qry = ProtStruct.get_atoms(aa_seq, cord_tns_qry, ['CA'])  # L x 3
            cmsk_vec = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])  # L
        elif atom_set == 'fa':
            cord_mat_ref = cord_tns_ref.view(n_resds * n_atoms, 3)  # (L x M) x 3
            cord_mat_qry = cord_tns_qry.view(n_resds * n_atoms, 3)  # (L x M) x 3
            cmsk_vec = cmsk_mat.view(n_resds * n_atoms)  # (L x M)
        else:
            raise ValueError('unrecognized atom set: ' + atom_set)

        # calculate pairwise distance matrices
        dist_mat_ref = cdist(cord_mat_ref)  # L x L or (L x M) x (L x M)
        dist_mat_qry = cdist(cord_mat_qry)
        rmsk_mat = 1 - torch.eye(n_resds, dtype=torch.int8, device=device)  # 1: same residue
        if atom_set == 'fa':
            rmsk_mat = rmsk_mat.repeat_interleave(n_atoms, dim=0).repeat_interleave(n_atoms, dim=1)
        dmsk_mat = rmsk_mat * \
            torch.outer(cmsk_vec, cmsk_vec) * (dist_mat_ref <= self.radius).to(torch.int8)

        # calculate the hit ratio under each distance threshold
        derr_mat = torch.abs(dist_mat_qry - dist_mat_ref)
        dhit_mat = torch.mean((derr_mat.unsqueeze(dim=2) <= self.dthr_tns).to(torch.float32), dim=2)
        if atom_set == 'ca':
            plddt_vec = torch.sum(dmsk_mat * dhit_mat, dim=1) / (torch.sum(dmsk_mat, dim=1) + self.eps)
        else:  # then <atom_set> must be 'fa'
            plddt_vec = torch.sum(dmsk_mat.view(n_resds, -1) * dhit_mat.view(n_resds, -1), dim=1) \
                / (torch.sum(dmsk_mat.view(n_resds, -1), dim=1) + self.eps)

        # determine validness masks for per-residue lDDT-Ca scores
        if atom_set == 'ca':
            plmsk_vec = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])  # L
        else:  # then <atom_set> must be 'fa'
            plmsk_vec = torch.max(cmsk_mat, dim=1)[0]  # L

        # calculate the full-chain lDDT-Ca score
        clddt_val = torch.sum(plmsk_vec * plddt_vec) / (torch.sum(plmsk_vec) + self.eps)

        return plddt_vec, plmsk_vec, clddt_val


def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    prot_id = 'T1031-D1'  # 'T1031-D1' / 'T1060s3-D1'
    fas_fpath = os.path.join(curr_dir, 'examples', '%s.fasta' % prot_id)
    pdb_fpath_natv = os.path.join(curr_dir, 'examples', '%s_native.pdb' % prot_id)
    pdb_fpath_decy = os.path.join(curr_dir, 'examples', '%s_decoy.pdb' % prot_id)

    # load reference & query structures
    _, cord_tns_ref, cmsk_mat, error_msg = ProtStruct.load(pdb_fpath_natv, fas_fpath=fas_fpath)
    assert error_msg is None, 'failed to parse the PDB file: ' + pdb_fpath_natv
    _, cord_tns_qry, _, error_msg = ProtStruct.load(pdb_fpath_decy, fas_fpath=fas_fpath)
    assert error_msg is None, 'failed to parse the PDB file: ' + pdb_fpath_decy

    # test w/ <LddtAssessor>
    assessor = LddtAssessor()

    # reconstruct CA-atom 3D coordinates

    plddt_vec, plmsk_vec, clddt_val = assessor.run(cord_tns_ref, cord_tns_qry, cmsk_mat, atom_set='ca')
    print('lDDT (CA-only): %.4f' % clddt_val.item())
    plddt_vec, plmsk_vec, clddt_val = assessor.run(cord_tns_ref, cord_tns_qry, cmsk_mat, atom_set='fa')
    print('lDDT (full-atom): %.4f' % clddt_val.item())


if __name__ == '__main__':
    main()
