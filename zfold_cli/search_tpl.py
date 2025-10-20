import json
import os
import sys
sys.path.append('./')

import pathlib
import pickle
from absl import app
from absl import flags
from absl import logging
from zfold.data import pipeline
from zfold.data import templates
from multiprocessing import Pool, Manager

flags.DEFINE_integer('cpus', 16, 'number of threads')
flags.DEFINE_list('fasta_paths', None, 'Paths to FASTA files, each containing '
                  'one sequence. Paths should be separated by commas. '
                  'All FASTA paths must have a unique basename as the '
                  'basename is used to name the output directories for '
                  'each prediction.')
flags.DEFINE_string('output_path', None, 'Path to the .pkl results.')
flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('hhsearch_binary_path', 'hhsearch',
                    'Path to the HHsearch executable.')
flags.DEFINE_string('kalign_binary_path', '/usr/bin/kalign',
                    'Path to the Kalign executable.')

flags.DEFINE_string('pdb70_database_path', None, 'Path to the PDB70 '
                    'database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with '
                    'template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date', None, 'Maximum template release date '
                    'to consider. Important if folding historical test sets.')
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a '
                    'mapping from obsolete PDB IDs to the PDB IDs of their '
                    'replacements.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 20

def generate_template_features(
    fasta_path: str,
    fasta_name: str,
    output_path: str,
    data_pipeline):
  """"""

  output_dir = os.path.split(output_path)[0]
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if os.path.exists(output_path):
      return

  feature_dict = data_pipeline.process(input_fasta_path=fasta_path)

  with open(output_path, 'wb') as f:
    pickle.dump(feature_dict, f, protocol=4)

def gen_one(param):
    fasta_path, fasta_name, output_path, data_pipeline = param
    generate_template_features(fasta_path=fasta_path,
                               fasta_name=fasta_name,
                               output_path=output_path,
                               data_pipeline=data_pipeline)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  logging.info(f'generate TPL features')
  data_pipeline = pipeline.DataPipelineTPL(
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      pdb70_database_path=FLAGS.pdb70_database_path,
      template_featurizer=template_featurizer)

  # Predict structure for each of the sequences.
  params = []
  for fasta_path, fasta_name in zip(FLAGS.fasta_paths, fasta_names):
      params.append([fasta_path, fasta_name, f'{FLAGS.output_path}/{fasta_name}/template.pkl', data_pipeline])

  print("Starting pool with {} cpus and {} tasks.".format(FLAGS.cpus, len(params)))
  print("Start parallel model prediction...")
  with Pool(processes=FLAGS.cpus) as p:
    p.map(gen_one, params)
  print('Done')


def search_tpl():
    flags.mark_flags_as_required([
        'fasta_paths',
        'output_path',
    ])
    app.run(main)

if __name__ == '__main__':
    search_tpl()
