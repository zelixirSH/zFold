from __future__ import absolute_import
try:
    from .creformat import cparse_a3m, cparse_fasta, cparse_aln
    parse_a3m = cparse_a3m
    parse_fasta = cparse_fasta
    parse_aln = cparse_aln
except ImportError:
    import warnings
    warnings.warn(RuntimeWarning('Using slow version'))
    from .reformat import parse_a3m, parse_fasta, parse_aln

__version__ = '0.3'
