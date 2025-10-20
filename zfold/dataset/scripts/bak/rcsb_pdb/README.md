# The RCSB-PDB Database

Procedures for building the RCSB-PDB database (and its subsets):
* Run `prune_clusters.py` to remove empty clusters.
* Run `build_fas_n_pdb.py` to build the full RCSB-PDB database (FASTA and native PDB files).
* (Optional) Run `build_subsets.py` to build subsets of various sizes (1k for prototype test, and 27k as the non-redundant subset).
* Run `split_data.py` to generate the train/valid/test data split.
* Run `build_hdf5.py` to build HDF5 files from FASTA and native PDB files.
