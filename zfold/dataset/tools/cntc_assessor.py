"""Assessor for inter-residue contact predictions."""

import logging

import numpy as np
from scipy.spatial.distance import cdist


class CntcAssessor():
    """Assessor for inter-residue contact predictions."""

    def __init__(self):
        """Constructor function."""

        self.eps = 1e-6
        self.dist_thres = 8.0
        self.idx_bin_beg = 1  # 2.0 - 2.5 A (inclusive)
        self.idx_bin_end = 12  # 7.5 - 8.0 A (inclusive)


    def get_masks_w_cord(self, cord_mat, mask_vec):
        """Get contact masks w/ ground-truth 3D coordinates.

        Args:
        * cord_mat: ground-truth 3D coordinates for CB atoms (CA for Glycine)
        * mask_vec: validness masks for CB atoms (CA for Glycine)

        Returns:
        * cmsk_mat: contact masks for CB-CB atom pairs (CA for Glycine)
        """

        dist_mat = cdist(cord_mat, cord_mat, metric='euclidean')
        cmsk_mat = (dist_mat <= self.dist_thres).astype(np.int8) * np.outer(mask_vec, mask_vec)

        return cmsk_mat


    def get_masks_w_labl(self, labl_mat, mask_mat):
        """Get contact masks w/ ground-truth categorical labels.

        Args:
        * labl_mat: ground-truth categorical labels (ranging from 0 to 36)
        * mask_mat: validness masks for CB-CB atom pairs (CA for Glycine)

        Returns:
        * cmsk_mat: contact masks for CB-CB atom pairs (CA for Glycine)
        """

        cmsk_mat = (labl_mat >= self.idx_bin_beg).astype(np.int8) \
            * (labl_mat <= self.idx_bin_end).astype(np.int8) * mask_mat

        return cmsk_mat


    def calc_prec(self, mask_mat, prob_mat):
        """"Run the assessor for inter-residue contact predictions w/ GT 3D coordinates.

        Args:
        * mask_mat: contact masks for CB-CB atom pairs (CA for Glycine)
        * prob_mat: predicted inter-residue contact probabilities

        Returns:
        * prec: top-L precision for long-range contact predictions
        """

        # initialization
        n_resds = mask_mat.shape[0]

        # find-out top-L predicted inter-residue contacts
        cntc_infos = []
        for ir in range(n_resds):
            for ic in range(ir + 24, n_resds):
                    cntc_infos.append((ir, ic, prob_mat[ir, ic]))
        cntc_infos.sort(key=lambda x: x[2], reverse=True)

        # count the number of correct predictions
        n_pairs_true = 0
        n_pairs_full = min(n_resds, len(cntc_infos))
        for idx in range(n_pairs_full):
            ir, ic, _ = cntc_infos[idx]
            if mask_mat[ir, ic] == 1:
                n_pairs_true += 1

        # calculcate the top-L precision for long-range contact predictions
        prec = n_pairs_true / (n_pairs_full + self.eps)

        return prec


    def calc_prec_v2(self, cord_mat, mask_vec, prob_mat):
        """Calculate the top-L precision for long-range contact predictions.

        Args:
        * cord_mat: ground-truth 3D coordinates of size L x 3
        * mask_vec: ground-truth 3D coordinates' validness masks of size L
        * prob_mat: predicted inter-residue contact probabilities of size L x L

        Returns:
        * prec: top-L precision for long-range contact predictions
        """

        # initialization
        n_resds = cord_mat.shape[0]

        # calculate the ground-truth distance matrix
        dist_mat = cdist(cord_mat, cord_mat, metric='euclidean')
        cmsk_mat = (dist_mat <= self.dist_thres).astype(np.int8) * np.outer(mask_vec, mask_vec)

        # find-out top-L predicted inter-residue contacts
        cntc_infos = []
        for ir in range(n_resds):
            for ic in range(ir + 24, n_resds):
                if mask_vec[ir] == 1 and mask_vec[ic] == 1:
                    cntc_infos.append((ir, ic, prob_mat[ir, ic]))
        cntc_infos.sort(key=lambda x: x[2], reverse=True)

        # count the number of correct predictions
        n_pairs_true = 0
        n_pairs_full = min(n_resds, len(cntc_infos))
        for idx in range(n_pairs_full):
            ir, ic, _ = cntc_infos[idx]
            if mask_mat[ir, ic] == 1:
                n_pairs_true += 1

        # calculcate the top-L precision for long-range contact predictions
        prec = n_pairs_true / (n_pairs_full + self.eps)

        return prec
