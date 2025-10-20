"""Parse the FASTA sequence & atom coordinates from the RCSB-PDB file."""

import os
import re
import gzip
import logging
import warnings

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from zfold.dataset.utils import get_rand_str
from zfold.dataset.utils import parse_fas_file
from zfold.dataset.utils import AA_NAMES_DICT_3TO1


class PdbParserError(Exception):
    """Exceptions raised by the <PdbParser> class."""


class PdbParser():
    """Parser for PDB files.

    Note: GZ-compressed PDB file can also be provided (*.gz).
    """

    def __init__(self, check_mode='strict'):
        """Constructor function."""

        self.check_mode = check_mode  # choices: 'strict' OR 'lenient'
        self.atom_names = ['N', 'CA', 'C', 'O', 'CB']
        self.n_atoms_per_resd = len(self.atom_names)
        self.idx_atom_ca = self.atom_names.index('CA')
        self.ca_dist_min = 2.8  # minimal allowed CA-CA distance of adjacent residues
        self.ca_dist_max = 4.8  # maximal allowed CA-CA distance of adjacent residues
        self.atom_mask_thres = 0.3  # minimal ratio of atoms w/ coordinates
        self.parser = PDBParser()

        # suppress all the warnings raised by <BioPython>
        warnings.simplefilter('ignore', BiopythonWarning)


    def run(self, pdb_fpath, fas_fpath=None, structure=None, chain_id=None):
        """Parse the PDB file.

        Args:
        * pdb_fpath: path to the PDB file
        * fas_fpath: (optional) path to the FASTA file
        * structure: (optional) BioPython's parsing results (to parse multiple chains from one PDB)
        * chain_id: (optional) chain ID

        Returns:
        * aa_seq: amino-acid sequence
        * atom_cords: atom coordinates of size L x 3 x 3 (N-CA-C')
        * atom_masks: atom coordinates' validness masks of size L x 3 (N-CA-C')
        * structure: BioPython's parsing results
        * error_msg: error message raised when parsing the PDB file

        Note:
        1. The amino-acid sequence is determined in the following order:
           a) parsed from the FASTA file, if exists
           b) parsed from SEQRES records in the PDB file
           c) parsed from ATOM records in the PDB file
        2. If <chain_id> is not provided, then the first chain will be returned. The specific order
           is defined by the <BioPython> package. If <chain_id> is provided, then the first model
           with the specified chain ID will be returned.
        """

        # show the greeting message
        logging.debug('parsing the PDB file: %s (chain ID: <%s>)', pdb_fpath, chain_id)
        if fas_fpath is not None:
            logging.debug('FASTA file provided: %s', fas_fpath)

        # attempt to parse the PDB file
        try:
            # check inputs
            if not os.path.exists(pdb_fpath):
                raise PdbParserError('PDB_FILE_NOT_FOUND')
            if not (pdb_fpath.endswith('.pdb') or pdb_fpath.endswith('.gz')):
                raise PdbParserError('PDB_FILE_FORMAT_NOT_SUPPORTED')
            if (fas_fpath is not None) and (not os.path.exists(fas_fpath)):
                raise PdbParserError('FASTA_FILE_NOT_FOUND')

            # obtain the amino-acid sequence
            if fas_fpath is not None:
                _, aa_seq = parse_fas_file(fas_fpath)
            else:  # then the amino-acid sequence must be parsed from the PDB file
                aa_seq = self.__get_aa_seq(pdb_fpath, chain_id)

            # parse the PDB file w/ biopython
            if structure is None:
                structure = self.__get_structure(pdb_fpath)

            # find the first chain matching the chain ID
            chain = self.__get_chain(structure, chain_id)

            # obtain atom coordinates & validness masks
            atom_cords, atom_masks = self.__get_atoms(chain, aa_seq)

            # validate atom coordinates
            if self.check_mode == 'strict':
                self.__validate_cords(atom_cords, atom_masks)

            # set the error message to None
            error_msg = None
        except PdbParserError as error:
            logging.warning('PDB file path: %s / Error: %s', pdb_fpath, error)
            aa_seq, atom_cords, atom_masks, structure, error_msg = None, None, None, None, error

        return aa_seq, atom_cords, atom_masks, structure, error_msg


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
            resd_infos_list = sorted(list(resd_infos_dict.items()))
            resd_names = [x[1] for x in resd_infos_list]

        # check whether all the residue names are valid
        for resd_name in resd_names:
            if resd_name not in AA_NAMES_DICT_3TO1:
                raise PdbParserError('HAS_UNKNOWN_RESIDUES')

        # convert residue names into the amino-acid sequence
        aa_seq = ''.join([AA_NAMES_DICT_3TO1[x] for x in resd_names])

        return aa_seq


    def __get_structure(self, path):
        """Get the structure from the PDB file."""

        try:
            if path.endswith('.pdb'):
                with open(path, 'r') as i_file:
                    structure = self.parser.get_structure(get_rand_str(), i_file)
            else:  # then <path> must end with '.gz'
                with gzip.open(path, 'rt') as i_file:
                    structure = self.parser.get_structure(get_rand_str(), i_file)
        except PDBConstructionException as error:
            raise PdbParserError('BIOPYTHON_FAILED_TO_PARSE') from error

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
            raise PdbParserError('CHAIN_NOT_FOUND')

        return chain


    def __get_atoms(self, chain, aa_seq):
        """Get atom coordinates & masks from the specified chain."""

        # obtain all the segments' information
        seg_infos = self.__get_seg_infos(chain)

        # find the valid offset for residue indices, if possible
        offset = self.__get_offset(seg_infos, aa_seq)

        # obtain atom coordinates & masks
        seq_len = len(aa_seq)
        atom_cords = np.zeros((seq_len, self.n_atoms_per_resd, 3), dtype=np.float32)
        atom_masks = np.zeros((seq_len, self.n_atoms_per_resd), dtype=np.int8)
        for residue in chain:
            # skip hetero-residues, and obtain the residue's index
            het_flag, idx_resd, _ = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues

            # update atom coordinates & masks
            for idx_atom, atom_name in enumerate(self.atom_names):
                if residue.has_id(atom_name):
                    atom_cords[idx_resd + offset, idx_atom] = residue[atom_name].get_coord()
                    atom_masks[idx_resd + offset, idx_atom] = 1

        # check whether sufficient ratio of atoms have coordinates
        if (self.check_mode == 'strict') and (np.mean(atom_masks) < self.atom_mask_thres):
            raise PdbParserError('INSUFFICIENT_ATOMS_W_COORDINATES')

        return atom_cords, atom_masks


    @classmethod
    def __get_seg_infos(cls, chain):
        """Get discontinous segments' information for the specified chain."""

        seg_infos = []
        for residue in chain:
            # obtain the current residue's basic information
            resd_name = residue.get_resname()
            resd_name_1c = AA_NAMES_DICT_3TO1.get(resd_name, '.')
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues
            if ins_code.strip() != '':
                raise PdbParserError('HAS_INSERTED_RESIDUES')

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

        # sort discontinous segments in the descending order of segment length
        seg_infos.sort(key=lambda x: x['ie'] - x['ib'], reverse=True)

        return seg_infos


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
                    if re.search(regex, aa_seq[seg_info['ib'] + offset:seg_info['ie'] + offset]):
                        offset_list_new.append(offset)
                offset_list = offset_list_new
            if len(offset_list) == 0:
                raise PdbParserError('NO_VALID_OFFSET')

        return offset_list[0]  # use the first valid offset


    def __validate_cords(self, atom_cords, atom_masks):
        """Validate atom coordinates based on adjacent residues' CA-CA distance."""

        # obtain CA atoms' coordinates & masks
        atom_cords_ca = atom_cords[:, self.idx_atom_ca, :]
        atom_masks_ca = atom_masks[:, self.idx_atom_ca]

        # validate adjacent residues' CA-CA distance
        is_valid = True
        dist_vec = np.linalg.norm(atom_cords_ca[1:] - atom_cords_ca[:-1], axis=-1)
        for idx in range(dist_vec.size):
            if atom_masks_ca[idx] == 0 or atom_masks_ca[idx + 1] == 0:
                continue
            if dist_vec[idx] < self.ca_dist_min or dist_vec[idx] > self.ca_dist_max:
                is_valid = False
                logging.warning('improper CA-CA distance: %.4f', dist_vec[idx])
                logging.warning('atom #%d: %s', idx, str(atom_cords_ca[idx]))
                logging.warning('atom #%d: %s', idx + 1, str(atom_cords_ca[idx + 1]))

        # raise an error if any improper CA-CA distance is detected
        if not is_valid:
            raise PdbParserError('IMPROPER_CA_CA_DISTANCE')
