""""""
import json
import os
import pathlib
from absl import app
from absl import flags
from absl import logging
from zfold.data import pipeline

flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_enum('profile_db_option', '6msa',
                  ['6msa', 'mmseq2cf', 'uc30','uc30+ur90','uc30+ur90+mgy','uc30+ur90+mgy+bfd'], 'profile database options')

flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_list('model_names', None, 'Names of models to use.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path', '/usr/bin/jackhmmer',
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path', 'hhblits',
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path', 'hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', '/usr/bin/kalign',
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', None, 'Path to the Uniref90 '
                    'database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', None, 'Path to the MGnify '
                    'database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', None, 'Path to the BFD '
                    'database for use by HHblits.')
flags.DEFINE_string('uniclust30_database_path', None, 'Path to the Uniclust30 '
                    'database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')

flags.DEFINE_string('mmseq2_binary_path', f'./MMSEQ2/mmseqs/bin/mmseqs', 'Path to xxx'
                    'database for use by HHblits.')
flags.DEFINE_string('mmseq2_database_path', f'./MMSEQ2/DATABASE', 'Path to xxx'
                    'database for use by HHblits.')

flags.DEFINE_enum('preset', 'full_dbs', ['full_dbs', 'casp14'],
                  'Choose preset model configuration - no ensembling '
                  '(full_dbs) or 8 model ensemblings (casp14).')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')

flags.DEFINE_boolean('save_distogram', False, '')
flags.DEFINE_boolean('save_embeddings', False, '')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20

def generate_msas(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline):
  """Predicts structure using AlphaFold for the given sequence and MSA."""

  output_dir = output_dir_base
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  data_pipeline.process(input_fasta_path=fasta_path, msa_output_dir=output_dir)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  logging.info(f'profile_db_option: {FLAGS.profile_db_option}')

  if FLAGS.profile_db_option == '6msa':

      if FLAGS.uniclust30_database_path is None:
          raise Exception('uniclust30_database_path can not be None')

      data_pipeline = pipeline.DataPipeline6MSA(
          jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
          hhblits_binary_path=FLAGS.hhblits_binary_path,
          hhsearch_binary_path=FLAGS.hhsearch_binary_path,
          uniref90_database_path=FLAGS.uniref90_database_path,
          mgnify_database_path=FLAGS.mgnify_database_path,
          bfd_database_path=FLAGS.bfd_database_path,
          uniclust30_database_path=FLAGS.uniclust30_database_path)
      # Predict structure for each of the sequences.
      for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
            generate_msas(fasta_path=fasta_path,
                          fasta_name=fasta_name,
                          output_dir_base=FLAGS.output_dir,
                          data_pipeline=data_pipeline)

  elif FLAGS.profile_db_option == 'mmseq2cf':
      data_pipeline = pipeline.DataPipelineMMseq2CF(search_script_path = f'{os.path.split(os.path.realpath(__file__))[0]}/tools/colabfold_search.sh',
                                                    mmseq2_binary_path = FLAGS.mmseq2_binary_path,
                                                    database_path = FLAGS.mmseq2_database_path)

      # Predict structure for each of the sequences.
      for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
            generate_msas(fasta_path=fasta_path,
                          fasta_name=fasta_name,
                          output_dir_base=FLAGS.output_dir,
                          data_pipeline=data_pipeline)

def search_msa():
    flags.mark_flags_as_required([
        'fasta_paths',
        'output_dir',
        # 'uniclust30_database_path'
    ])
    app.run(main)

