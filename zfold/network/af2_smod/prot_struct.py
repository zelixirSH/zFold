"""The wrapper class for utility functions related to protein structures."""

import os
import re
import gzip
import logging
import warnings

import numpy as np
import torch
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from zfold.dataset.utils import get_rand_str
from zfold.dataset.utils import parse_fas_file
from zfold.loss.fape.constants import RESD_NAMES_3C
from zfold.loss.fape.constants import RESD_MAP_1TO3
from zfold.loss.fape.constants import RESD_MAP_3TO1
from zfold.loss.fape.constants import ATOM_NAMES_PER_RESD
from zfold.loss.fape.constants import N_ATOMS_PER_RESD_MAX

class PdbParseError(Exception):
    """Exceptions raised when parsing a PDB file w/ BioPython."""


class ProtStruct():
    """The wrapper class for utility functions related to protein structures."""

    def __init__(self):
        """Constructor function."""

        pass


    @classmethod
    def load(cls, pdb_fpath, fas_fpath=None, chain_id=None):
        """Load a protein structure from the PDB file.

        Args:
        * pdb_fpath: path to the PDB file
        * fas_fpath: (optional) path to the FASTA file
        * chain_id: (optional) chain ID

        Returns:
        * aa_seq: amino-acid sequence
        * atom_cords: per-atom 3D coordinates of size L x M x 3
        * atom_masks: per-atom 3D coordinates' validness masks of size L x M
        * error_msg: error message raised when parsing the PDB file

        Note:
        * The GZ-compressed PDB file can be provided with a suffix of ".gz".
        * The amino-acid sequence is determined in the following order:
          a) parsed from the FASTA file
          b) parsed from SEQRES records in the PDB file
          c) parsed from ATOM records in the PDB file
        * If <chain_id> is not provided, then the first chain will be returned. The specific order
          is defined by the <BioPython> package. If <chain_id> is provided, then the first model
          with the specified chain ID will be returned.
        """

        # suppress all the warnings raised by <BioPython>
        warnings.simplefilter('ignore', BiopythonWarning)

        # show the greeting message
        logging.debug('parsing the PDB file: %s (chain ID: <%s>)', pdb_fpath, chain_id)
        if fas_fpath is not None:
            logging.debug('FASTA file provided: %s', fas_fpath)

        # attempt to parse the PDB file
        try:
            # check inputs
            if not os.path.exists(pdb_fpath):
                raise PdbParseError('PDB_FILE_NOT_FOUND')
            if not (pdb_fpath.endswith('.pdb') or pdb_fpath.endswith('.gz')):
                raise PdbParseError('PDB_FILE_FORMAT_NOT_SUPPORTED')
            if (fas_fpath is not None) and (not os.path.exists(fas_fpath)):
                raise PdbParseError('FASTA_FILE_NOT_FOUND')

            # obtain the amino-acid sequence
            if fas_fpath is not None:
                _, aa_seq = parse_fas_file(fas_fpath)
            else:  # then the amino-acid sequence must be parsed from the PDB file
                aa_seq = cls.__get_aa_seq(pdb_fpath, chain_id)

            # parse the PDB file w/ biopython
            structure = cls.__get_structure(pdb_fpath)

            # find the first chain matching the chain ID
            chain = cls.__get_chain(structure, chain_id)

            # obtain atom coordinates & validness masks
            atom_cords_np, atom_masks_np = cls.__get_atoms(chain, aa_seq)
            atom_cords = torch.tensor(atom_cords_np, dtype=torch.float32)
            atom_masks = torch.tensor(atom_masks_np, dtype=torch.int8)

            # set the error message to None
            error_msg = None

        except PdbParseError as error:
            logging.warning('PDB file path: %s / Error: %s', pdb_fpath, error)
            aa_seq, atom_cords, atom_masks, error_msg = None, None, None, error

        return aa_seq, atom_cords, atom_masks, error_msg
        # atom 14

    @classmethod
    def save(cls, aa_seq, atom_cords, atom_masks, path, glddt: float = None, plddt: np.ndarray = None):
        """Save the protein structure to a PDB file.

        Args:
        * aa_seq: amino-acid sequence
        * atom_cords: per-atom 3D coordinates of size L x M x 3
        * atom_masks: per-atom 3D coordinates' validness masks of size L x M
        * path: path to the PDB file
        * glddt: global lddt score, float
        * plddt: predicted lDDT scores of size L, numpy.ndarray

        Returns: n/a
        """

        # configurations
        alt_loc = ' '
        chain_id = 'A'
        i_code = ' '
        occupancy = 1.0
        temp_factor = 0.0
        charge = ' '
        cord_min = -999.999
        cord_max = 9999.999
        seq_len = len(aa_seq)

        # reset invalid values in atom coordinates
        atom_cords_vald = torch.clip(atom_cords, cord_min, cord_max)
        atom_cords_vald[torch.isnan(atom_cords_vald)] = 0.0
        atom_cords_vald[torch.isinf(atom_cords_vald)] = 0.0

        # export the 3D structure to a PDB file
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w') as o_file:
            n_atoms = 0
            
            o_file.write("REMARK zFold v0.0.1 by Shanghai Zelixir Biotech Co Ltd\n")
            if glddt:
                o_file.write(f"REMARK Global plDDT score = {glddt:.3f}\n")

            o_file.write("MODEL 1\n")
            for idx_resd in range(len(aa_seq)):
                resd_name = RESD_MAP_1TO3[aa_seq[idx_resd]]
                atom_names = ATOM_NAMES_PER_RESD[resd_name]

                # define lddt
                if plddt is not None:
                    try:
                        temp_factor = plddt[idx_resd] * 100.
                    except IndexError:
                        temp_factor = 0.0

                for idx_atom, atom_name in enumerate(atom_names):
                    if atom_masks[idx_resd, idx_atom] == 0:
                        continue
                    n_atoms += 1
                    line_str = ''.join([
                        'ATOM  ',
                        '%5d' % n_atoms,
                        '  ' + atom_name + ' ' * (3 - len(atom_name)),
                        alt_loc,
                        '%3s' % resd_name,
                        ' %s' % chain_id,
                        '%4d' % (idx_resd + 1),
                        '%s   ' % i_code,
                        '%8.3f' % atom_cords_vald[idx_resd, idx_atom, 0],
                        '%8.3f' % atom_cords_vald[idx_resd, idx_atom, 1],
                        '%8.3f' % atom_cords_vald[idx_resd, idx_atom, 2],
                        '%6.2f' % occupancy,
                        '%6.2f' % temp_factor,
                        ' ' * 10,
                        '%2s' % atom_name[0],
                        '%2s' % charge,
                    ])
                    assert len(line_str) == 80, 'line length (%d) must be 80' % len(line_str)
                    o_file.write(line_str + '\n')
            
            o_file.write("TER\nENDMDL\n")


    @classmethod
    def get_atoms_legacy(cls, aa_seq, atom_tns_all, atom_names_sel, idx_resd=None):
        """Get per-atom 3D coordinates or validness masks for specified residue(s) & atom(s).

        Args:
        * aa_seq: amino-acid sequence
        * atom_tns_all: full per-atom 3D coordinates (L x M x 3) or validness masks (L x M)
        * atom_names_sel: list of selected atom names
        * idx_resd: (optional) residue index; if None, all the residues are returned

        Returns:
        * atom_tns_sel: selected per-atom 3D coordinates (L' x M' x 3) or validness masks (L' x M')

        Note:
        * L': number of selected residues
        * M': number of selected atoms
        * If <L'> and/or <M'> equals one, then the corresponding dimension is squeezed.
        """

        # initialization
        device = atom_tns_all.device
        n_resds = 1 if idx_resd is not None else atom_tns_all.shape[0]
        n_atoms = len(atom_names_sel)

        # initialize the tensor for selected atoms
        if atom_tns_all.ndim == 2:  # validness masks
            atom_tns_sel = torch.zeros((n_resds, n_atoms), dtype=torch.int8, device=device)
        else:  # 3D coordinates
            atom_tns_sel = torch.zeros((n_resds, n_atoms, 3), dtype=torch.float32, device=device)

        # build pairwise residue indices
        if idx_resd is not None:
            pidxs_resd = [(0, idx_resd)]
        else:
            pidxs_resd = [(x, x) for x in range(n_resds)]

        # get 3D coordinates or validness masks for the specified residue(s) & atom(s)
        for idx_resd_sel, idx_resd_all in pidxs_resd:
            resd_name = aa_seq[idx_resd_all]
            atom_names_all = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                if atom_name_sel not in atom_names_all:
                    continue
                idx_atom_all = atom_names_all.index(atom_name_sel)
                atom_tns_sel[idx_resd_sel, idx_atom_sel] = atom_tns_all[idx_resd_all, idx_atom_all]

        # squeeze the dimension whose length equals to one
        atom_tns_sel.squeeze_()

        return atom_tns_sel


    @classmethod
    def get_atoms(cls, aa_seq, atom_tns_all, atom_names_sel):
        """Get per-atom 3D coordinates or validness masks for selected atom(s).

        Args:
        * aa_seq: amino-acid sequence
        * atom_tns_all: full per-atom 3D coordinates (L x M x 3) or validness masks (L x M)
        * atom_names_sel: list of selected atom names of length M'

        Returns:
        * atom_tns_sel: selected per-atom 3D coordinates (L x M' x 3) or validness masks (L x M')

        Note:
        * If only one atom name if provided, then its corresponding dimension is squeezed.
        """

        # use the specifically optimized implementation if only CA atoms are needed
        if atom_names_sel == ['CA']:
            return atom_tns_all[:, 1]  # CA atom is always the 2nd atom, as defined in constants.py

        # initialization
        device = atom_tns_all.device
        n_resds = atom_tns_all.shape[0]
        n_atoms = len(atom_names_sel)
        # logging.info('atom_names_sel: %s', ','.join(atom_names_sel))
        # build the indexing tensor for selected atom(s)
        idxs_vec_dict = {}  # atom indices
        msks_vec_dict = {}  # atom indices' validness masks
        for resd_name in RESD_NAMES_3C:
            atom_names_all = ATOM_NAMES_PER_RESD[resd_name]
            idxs_vec_np = np.zeros((n_atoms), dtype=np.int64)
            msks_vec_np = np.zeros((n_atoms), dtype=np.int8)
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                if atom_name_sel in atom_names_all:  # otherwise, keep zeros unchanged
                    idxs_vec_np[idx_atom_sel] = atom_names_all.index(atom_name_sel)
                    msks_vec_np[idx_atom_sel] = 1
            idxs_vec_dict[resd_name] = idxs_vec_np
            msks_vec_dict[resd_name] = msks_vec_np

        # determine the overall indexing tensor based on the amino-acid sequence
        resd_names_3c = [RESD_MAP_1TO3[resd_name_1c] for resd_name_1c in aa_seq]
        idxs_mat_full_np = np.stack([idxs_vec_dict[x] for x in resd_names_3c], axis=0)
        msks_mat_full_np = np.stack([msks_vec_dict[x] for x in resd_names_3c], axis=0)
        idxs_mat_full = torch.tensor(idxs_mat_full_np, dtype=torch.int64, device=device)  # L x M'
        msks_mat_full = torch.tensor(msks_mat_full_np, dtype=torch.int64, device=device)  # L x M'

        # get per-atom 3D coordinates or validness masks for specified residue(s) & atom(s)
        if atom_tns_all.ndim == 2:
            atom_tns_sel = msks_mat_full * torch.gather(atom_tns_all, 1, idxs_mat_full)
        else:
            n_dims_addi = atom_tns_all.shape[-1]
            atom_tns_sel = msks_mat_full.unsqueeze(dim=2) * torch.gather(
                atom_tns_all, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, n_dims_addi))

        # squeeze the dimension if only one atom is selected
        if n_atoms == 1:
            atom_tns_sel.squeeze_(dim=1)

        return atom_tns_sel


    @classmethod
    def set_atoms(cls, aa_seq, atom_tns_all, atom_tns_sel, atom_names_sel, idx_resd=None):
        """Set per-atom 3D coordinates or validness masks for specified residue(s) & atom(s).

        Args:
        * aa_seq: amino-acid sequence
        * atom_tns_all: full per-atom 3D coordinates (L x M x 3) or validness masks (L x M)
        * atom_tns_sel: selected per-atom 3D coordinates (L' x M' x 3) or validness masks (L' x M')
        * atom_names_sel: list of selected atom names
        * idx_resd: (optional) residue index; if None, all the residues are returned

        Returns: n/a
        """

        # initialization
        n_resds = 1 if idx_resd is not None else atom_tns_all.shape[0]
        n_atoms = len(atom_names_sel)

        # build pairwise residue indices
        if idx_resd is not None:
            pidxs_resd = [(0, idx_resd)]
        else:
            pidxs_resd = [(x, x) for x in range(n_resds)]

        # get 3D coordinates or validness masks for the specified residue(s) & atom(s)
        for idx_resd_sel, idx_resd_all in pidxs_resd:
            resd_name = aa_seq[idx_resd_all]
            atom_names_all = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                if atom_name_sel not in atom_names_all:
                    continue
                idx_atom_all = atom_names_all.index(atom_name_sel)
                atom_tns_all[idx_resd_all, idx_atom_all] = atom_tns_sel[idx_resd_sel, idx_atom_sel]


    @classmethod
    def __get_aa_seq(cls, path, chain_id):
        """Get the FASTA sequence for the specified chain, based on SEQRES/ATOM records."""

        # obtain line strings from the PDB file
        if path.endswith('.pdb'):
            with open(path, 'r') as i_file:
                i_lines = [i_line.strip() for i_line in i_file]
        else:  # then <path> must end with '.gz'
            with gzip.open(path, 'rt') as i_file:
                i_lines = [i_line.strip() for i_line in i_file]

        # parse SEQRES records to obtain the FASTA sequence
        resd_names = []
        for i_line in i_lines:
            if not i_line.startswith('SEQRES'):
                continue
            if (chain_id is not None) and (i_line[11] != chain_id):
                continue
            resd_names.extend(i_line[19:].split())

        # parse ATOM-CA records to obtain the FASTA sequence, if SEQRES records are missing
        if len(resd_names) == 0:
            logging.debug('SEQRES records are missing; using ATOM-CA records instead ...')
            resd_infos_dict = {}
            for i_line in i_lines:
                if not i_line.startswith('ATOM'):
                    continue
                if (chain_id is not None) and (i_line[21] != chain_id):
                    continue
                resd_name = i_line[17:20]
                idx_resd = int(i_line[22:26])
                resd_infos_dict[idx_resd] = resd_name
            idx_resd_min = min(resd_infos_dict.keys())
            idx_resd_max = max(resd_infos_dict.keys())
            resd_names = [resd_infos_dict.get(x, 'UNK') for x in range(idx_resd_min, idx_resd_min + 1)]

        # check whether all the residue names are valid
        for resd_name in resd_names:
            if resd_name not in RESD_NAMES_3C:
                raise PdbParseError('HAS_UNKNOWN_RESIDUES')

        # convert residue names into the amino-acid sequence
        aa_seq = ''.join([RESD_MAP_3TO1[x] for x in resd_names])

        return aa_seq


    @classmethod
    def __get_structure(cls, path):
        """Get the structure from the PDB file."""

        try:
            parser = PDBParser()
            if path.endswith('.pdb'):
                with open(path, 'r') as i_file:
                    structure = parser.get_structure(get_rand_str(), i_file)
            else:  # then <path> must end with '.gz'
                with gzip.open(path, 'rt') as i_file:
                    structure = parser.get_structure(get_rand_str(), i_file)
        except PDBConstructionException as error:
            raise PdbParseError('BIOPYTHON_FAILED_TO_PARSE') from error

        return structure


    @classmethod
    def __get_chain(cls, structure, chain_id):
        """Get the first chain matching the specified chain ID (could be None)."""

        chain = None
        for model in structure:
            for chain_curr in model:
                if (chain_id is None) or (chain_curr.get_id() == chain_id):
                    chain = chain_curr
                    break
            if chain is not None:
                break

        # check whether the specified chain has been found
        if chain is None:
            raise PdbParseError('CHAIN_NOT_FOUND')

        return chain


    @classmethod
    def __get_atoms(cls, chain, aa_seq):
        """Get atom coordinates & masks from the specified chain."""

        # obtain all the segments' information
        seg_infos, index_dict = cls.__get_seg_infos(chain, aa_seq)

        # find the valid offset for residue indices, if possible
        # offset = cls.__get_offset(seg_infos, aa_seq)

        # obtain atom coordinates & masks
        seq_len = len(aa_seq)
        atom_cords = np.zeros((seq_len, N_ATOMS_PER_RESD_MAX, 3), dtype=np.float32)
        atom_masks = np.zeros((seq_len, N_ATOMS_PER_RESD_MAX), dtype=np.int8)

        for residue in chain:
            # skip hetero-residues, and obtain the residue's index
            het_flag, idx_resd, _ = residue.get_id()

            index = index_dict[idx_resd]

            if het_flag.strip() != '':
                continue  # skip hetero-residues

            # update atom coordinates & masks
            resd_name = residue.get_resname()
            atom_names = ATOM_NAMES_PER_RESD[resd_name]
            for idx_atom, atom_name in enumerate(atom_names):
                if residue.has_id(atom_name):
                    atom_cords[index, idx_atom] = residue[atom_name].get_coord()
                    atom_masks[index, idx_atom] = 1

        return atom_cords, atom_masks


    @classmethod
    def __get_seg_infos(cls, chain, aa_seq):
        """Get discontinous segments' information for the specified chain."""

        seg_infos = []
        for residue in chain:
            # obtain the current residue's basic information
            resd_name = residue.get_resname()
            resd_name_1c = RESD_MAP_3TO1.get(resd_name, '.')
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues

            if ins_code.strip() != '':
                raise PdbParseError('HAS_INSERTED_RESIDUES')

            # update the last segment, or add a new segment
            if len(seg_infos) >= 1 and seg_infos[-1]['ie'] == idx_resd:
                seg_infos[-1]['ie'] += 1
                seg_infos[-1]['seq'] += resd_name_1c
            else:
                seg_infos.append({
                    'ib': idx_resd,  # inclusive
                    'ie': idx_resd + 1,  # exclusive
                    'seq': resd_name_1c,  # 20 amino-acids + '.' for wild-card matches
                })

        # sort discontinous segments
        index_dict = {}
        seg_infos.sort(key=lambda x: x['ib'], reverse=False)
        for i in range(len(seg_infos)):
            regex = re.compile(seg_infos[i]['seq'])
            offset_list = [m.start() for m in re.finditer(regex, aa_seq)]
            for i_ in range(seg_infos[i]['ib'], seg_infos[i]['ie']):
                index_dict[i_] = offset_list[0] + i_ - seg_infos[i]['ib']

        # sort discontinous segments in the descending order of segment length
        seg_infos.sort(key=lambda x: x['ie'] - x['ib'], reverse=True)

        return seg_infos, index_dict


    @classmethod
    def __get_offset(cls, seg_infos, aa_seq):
        """Get a valid offset of residue indices in ATOM records & amino-acid sequence."""

        offset_list = None

        for seg_info in seg_infos:
            regex = re.compile(seg_info['seq'])

            if offset_list is None:
                offset_list = [m.start() - seg_info['ib'] for m in re.finditer(regex, aa_seq)]
            else:
                offset_list_new = []
                for offset in offset_list:
                    # print(regex, aa_seq[seg_info['ib'] + offset:seg_info['ie'] + offset],
                    # re.search(regex, aa_seq[seg_info['ib'] + offset:seg_info['ie'] + offset]))
                    if re.search(regex, aa_seq[seg_info['ib'] + offset:seg_info['ie'] + offset]):
                        offset_list_new.append(offset)
                offset_list = offset_list_new

            if len(offset_list) == 0:
                raise PdbParseError('NO_VALID_OFFSET')

        return offset_list[0]  # use the first valid offset
