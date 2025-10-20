"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Mapping, Sequence

import numpy as np

# Internal import (7716).

from zfold.data import residue_constants
from zfold.data import parsers
from zfold.data import templates
from zfold.data.tools import hhblits
from zfold.data.tools import hhsearch
from zfold.data.tools import jackhmmer
from zfold.data.tools.stockholm_reformat import parse_a3m as convert_sto_a3m

FeatureDict = Mapping[str, np.ndarray]

def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: str,
               uniclust30_database_path: str,
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000):
    """Constructs a feature dict for a given FASTA file."""
    self.bfd_database_path = bfd_database_path
    if bfd_database_path is None and uniclust30_database_path is not None:
        self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
            binary_path=hhblits_binary_path,
            databases=[uniclust30_database_path])
    elif bfd_database_path is not None and uniclust30_database_path is not None:
        self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
            binary_path=hhblits_binary_path,
            databases=[bfd_database_path, uniclust30_database_path])
    else:
        raise Exception('uniclust30_database_path must be provided')

    if uniref90_database_path is not None:
        self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path)
    else:
        self.jackhmmer_uniref90_runner = None

    if mgnify_database_path is not None:
        self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
            binary_path=jackhmmer_binary_path,
            database_path=mgnify_database_path)
    else:
        self.jackhmmer_mgnify_runner = None

    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path])
    self.template_featurizer = template_featurizer
    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    hhsearch_result_a3m = None
    msas, deletion_matrices = [],[]

    if self.jackhmmer_uniref90_runner is not None:
        uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')

        if os.path.exists(uniref90_out_path):
            with open(uniref90_out_path, 'r') as f:
                jackhmmer_uniref90_result_sto = f.read()
        else:
            jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
                input_fasta_path)
            jackhmmer_uniref90_result_sto = jackhmmer_uniref90_result['sto']
            with open(uniref90_out_path, 'w') as f:
              f.write(jackhmmer_uniref90_result_sto)

        hhsearch_result_a3m = parsers.convert_stockholm_to_a3m(
            jackhmmer_uniref90_result_sto, max_sequences=self.uniref_max_hits)
        uniref90_msa, uniref90_deletion_matrix = parsers.parse_stockholm(
            jackhmmer_uniref90_result_sto)

        msas.append(uniref90_msa)
        deletion_matrices.append(uniref90_deletion_matrix)

    if self.jackhmmer_mgnify_runner is not None:
        mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
        if os.path.exists(mgnify_out_path):
            with open(mgnify_out_path, 'r') as f:
                jackhmmer_mgnify_result_sto = f.read()
        else:
            jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
                input_fasta_path)
            jackhmmer_mgnify_result_sto = jackhmmer_mgnify_result['sto']
            with open(mgnify_out_path, 'w') as f:
              f.write(jackhmmer_mgnify_result_sto)

        mgnify_msa, mgnify_deletion_matrix = parsers.parse_stockholm(
            jackhmmer_mgnify_result_sto)
        mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
        mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]

        msas.append(mgnify_msa)
        deletion_matrices.append(mgnify_deletion_matrix)

    hhblits_bfd_uniclust_result = self.hhblits_bfd_uniclust_runner.query(
        input_fasta_path)

    bfd_out_path = os.path.join(msa_output_dir, 'uniclust_hits.a3m') if self.bfd_database_path is None\
        else os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')

    with open(bfd_out_path, 'w') as f:
      f.write(hhblits_bfd_uniclust_result['a3m'])

    bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
        hhblits_bfd_uniclust_result['a3m'])

    msas.append(bfd_msa)
    deletion_matrices.append(bfd_deletion_matrix)

    if hhsearch_result_a3m is None:
        hhsearch_result_a3m = hhblits_bfd_uniclust_result['a3m']

    hhsearch_result = self.hhsearch_pdb70_runner.query(hhsearch_result_a3m)
    hhsearch_hits = parsers.parse_hhr(hhsearch_result)

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        query_pdb_code=None,
        query_release_date=None,
        hhr_hits=hhsearch_hits)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features(
        msas=msas,
        deletion_matrices=deletion_matrices)

    return {**sequence_features, **msa_features, **templates_result.features}

class DataPipelineGivenMSA:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               hhsearch_binary_path: str,
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer):

    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
                        binary_path=hhsearch_binary_path,
                        databases=[pdb70_database_path])
    self.template_featurizer = template_featurizer

  def process(self, input_fasta_path: str, input_msa_path: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    with open(input_msa_path) as f:
      input_msa_str = f.read()

    msa, msa_deletion_matrix = parsers.parse_a3m(input_msa_str)

    hhsearch_result = self.hhsearch_pdb70_runner.query(input_msa_str)

    hhsearch_hits = parsers.parse_hhr(hhsearch_result)

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        query_pdb_code=None,
        query_release_date=None,
        hhr_hits=hhsearch_hits)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features(
        msas=(msa,msa,msa),
        deletion_matrices=(msa_deletion_matrix,msa_deletion_matrix,msa_deletion_matrix))

    # return {**sequence_features, **msa_features}
    return {**sequence_features, **msa_features, **templates_result.features}


class DataPipeline6MSA:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: str,
               uniclust30_database_path: str,
               uniref_max_hits: int = 10000):
    """Constructs a feature dict for a given FASTA file."""

    self.hhblits_bfd_runner_3_1 = hhblits.HHBlits(
            n_iter = 3,
            e_value = 1e-1,
            binary_path=hhblits_binary_path,
            databases=[bfd_database_path],
            n_cpu = os.cpu_count()-2)
    self.hhblits_bfd_runner_3_3 = hhblits.HHBlits(
            n_iter=3,
            e_value=1e-3,
            binary_path=hhblits_binary_path,
            databases=[bfd_database_path],
            n_cpu = os.cpu_count()-2)
    self.hhblits_bfd_runner_3_5 = hhblits.HHBlits(
            n_iter=3,
            e_value=1e-5,
            binary_path=hhblits_binary_path,
            databases=[bfd_database_path],
            n_cpu = os.cpu_count()-2)

    self.jackhmmer_uniref90_runner_3_3 = jackhmmer.Jackhmmer(
            n_iter = 3,
            e_value = 1e-3,
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path,
            n_cpu=os.cpu_count()-2)
    self.jackhmmer_uniref90_runner_3_5 = jackhmmer.Jackhmmer(
            n_iter = 3,
            e_value = 1e-5,
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path,
            n_cpu=os.cpu_count()-2)
    self.jackhmmer_uniref90_runner_3_10 = jackhmmer.Jackhmmer(
            n_iter = 3,
            e_value = 1e-10,
            binary_path=jackhmmer_binary_path,
            database_path=uniref90_database_path,
            n_cpu=os.cpu_count()-2)

    self.uniref_max_hits = uniref_max_hits

  def process(self, input_fasta_path: str, msa_output_dir: str):
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')

    def get_ur90_msa(jackhmmer_uniref90_runner, save_path):

        if os.path.exists(save_path.replace('.sto','.a3m')):
            print(f'{save_path.replace(".sto",".a3m")} exists')
            return

        jackhmmer_uniref90_result = jackhmmer_uniref90_runner.query(
            input_fasta_path)
        jackhmmer_uniref90_result_sto = jackhmmer_uniref90_result['sto']
        with open(save_path, 'w') as f:
          f.write(jackhmmer_uniref90_result_sto)

        convert_sto_a3m(save_path, save_path.replace('.sto','.a3m'))
        os.remove(save_path)

    def get_bfd_msa(hhblits_bfd_runner, save_path):
        if not os.path.exists(save_path):
            hhblits_bfd_result = hhblits_bfd_runner.query(input_fasta_path)
            with open(save_path, 'w') as f:
                f.write(hhblits_bfd_result['a3m'])
        else:
            print(f'{save_path} exists')

    get_ur90_msa(self.jackhmmer_uniref90_runner_3_3, os.path.join(msa_output_dir, 'ur90_3_3.sto'))
    get_ur90_msa(self.jackhmmer_uniref90_runner_3_5, os.path.join(msa_output_dir, 'ur90_3_5.sto'))
    get_ur90_msa(self.jackhmmer_uniref90_runner_3_10, os.path.join(msa_output_dir, 'ur90_3_10.sto'))

    get_bfd_msa(self.hhblits_bfd_runner_3_1, os.path.join(msa_output_dir, 'bfd_3_1.a3m'))
    get_bfd_msa(self.hhblits_bfd_runner_3_3, os.path.join(msa_output_dir, 'bfd_3_3.a3m'))
    get_bfd_msa(self.hhblits_bfd_runner_3_5, os.path.join(msa_output_dir, 'bfd_3_5.a3m'))


class DataPipelineTPL:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               hhsearch_binary_path: str,
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer):

    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
                        binary_path=hhsearch_binary_path,
                        databases=[pdb70_database_path])
    self.template_featurizer = template_featurizer

  def process(self, input_fasta_path: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    with open(input_fasta_path) as f:
      input_msa_str = f.read()

    hhsearch_result = self.hhsearch_pdb70_runner.query(input_msa_str)

    hhsearch_hits = parsers.parse_hhr(hhsearch_result)

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        query_pdb_code=None,
        query_release_date=None,
        hhr_hits=hhsearch_hits)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    # return {**sequence_features, **msa_features}
    return {**sequence_features, **templates_result.features}

class DataPipelineMMseq2CF:
  #TODO
  #convert bash script to python

  def __init__(self, search_script_path: str, mmseq2_binary_path: str, database_path: str):
    """Constructs a feature dict for a given FASTA file."""
    self.search_script_path = search_script_path
    self.mmseq2_binary_path = mmseq2_binary_path
    self.database_path = database_path

  def process(self, input_fasta_path: str, msa_output_dir: str):
    """Runs alignment tools on the input sequence and creates features."""
    cmd = f'{self.search_script_path} ' \
          f'{self.mmseq2_binary_path} ' \
          f'"{input_fasta_path}" ' \
          f'"{self.database_path}" ' \
          f'"{msa_output_dir}" ' \
          f'"uniref30_2103_db" "" "colabfold_envdb_202108_db" "1" "0" "1"'

    os.system(cmd)

