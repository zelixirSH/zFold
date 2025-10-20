from setuptools import setup, find_packages

setup(
  name = 'zfold',
  packages = find_packages(),
  version = '0.0.1',
  entry_points={
    'console_scripts':
      [
       'zfold_predict = zfold_cli.pred_npz_multi_gpu:main',
       'zfold_predict_e2e = zfold_cli.pred_pdb_multi_gpu:main',
       'zfold_predict_e2e_single = zfold_cli.pred_pdb_single:main',
       'zfold_msa_gen = zfold_cli.search_msa:search_msa',
       'zfold_tpl_gen = zfold_cli.search_tpl:search_tpl',
       'zfold_eval_pdb = zfold_cli.eval_pdb:eval_pdb',
       'zfold_ave_npz = zfold_cli.ave_npz:ave_npz',
      ],
  },
  license='',
  description = 'zfold',
  author = '',
  author_email = '',
  url = '',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'protein folding'
  ],
  install_requires=[
    'fire>=0.4.0'
    'einops>=0.3',
    'torch>=1.9',
    'torchvision>=0.10.0',
    'timm',
    'matplotlib',
    'Bio',
    'python-box',
    'dm-tree',
    'ml_collections',
    'ray'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
