# AlphaFold2 - The Structure Module

Symbol definitions:

* `L`: number of residues
* `M`: maximal number of atoms per residue
* `K`: maximal number of torsion angles per residue

As illustrated in `constants.py`, the maximal number of atoms per residue is `M = 14`, and the
maximal number of torsion angles per residue is `K = 7`, although the first two torsion angles are
not directly computed.

To represent a protein (with all the atoms' 3D coordinates), following items are used:

* `aa_seq`: amino-acid sequence
* `cord_tns`: per-atom 3D coordinates of size L x M x 3
* `cmsk_mat`: per-atom 3D coordinates' validness masks of size L x M
* `fram_tns`: per-residue local frames of size L x 4 x 3
* `fmsk_vec`: per-residue local frames' validness masks of size L
* `angl_tns`: per-residue torsion angles of size L x K x 2
* `amsk_mat`: per-residue torsion angles' validness masks of size L x K
* `quat_mat`: partial quaternion vectors of size L x 3
* `trsl_mat`: translation vectors of size L x 3
* `angl_tns`: (un-normalized) torsion angle matrices of size L x K x 2

When processed in `nn.Module`-based modules, an additional batch dimension will be added, although
the batch size is often fixed to `N = 1`.

The last three items (`quat_mat`, `trsl_mat`, and `angl_tns`) are so-called QTA parameters, which
are outputs from AlphaFold2's structure module and will be optimized to minimize the FAPE loss.
