import json
import os
import pathlib
from absl import app
from absl import flags
from zfold.utils import ave_trros_npz_list

flags.DEFINE_list('npz_paths', None, 'Paths to FASTA files. Paths should be separated by commas.')
flags.DEFINE_string('ave_npz_out', None, 'Path to output npz file')
flags.DEFINE_boolean('is_png', False, '')
FLAGS = flags.FLAGS

def main(argv):
    ave_trros_npz_list(npz_list= FLAGS.npz_paths, npz_out = FLAGS.ave_npz_out, is_png = FLAGS.is_png)
    print(f'merge given npzs {FLAGS.npz_paths} into {FLAGS.ave_npz_out}')

def ave_npz():
    flags.mark_flags_as_required([
        'npz_paths',
        'ave_npz_out',
    ])
    app.run(main)
