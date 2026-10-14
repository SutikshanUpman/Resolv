"""Shared table access: normalised tables, integer id maps, ground truth and split membership.

Integer ids: i1 = row of the S1 table of a split ('train' is the ghosted train S1, 'test' is test S1);
ir = row of concat(S2, S3) of that split. Every stage exchanges (i1, ir) pairs, never strings.
"""
import polars as pl
from .config import OUT, NORM_VERSION

NORM_DIR = OUT / 'norm'


def norm_path(split, s):
    return NORM_DIR / f'{split}_s{s}_{NORM_VERSION}.parquet'


def nt(split, s, cols=None):
    """Normalised table (optionally a subset of columns)."""
    return pl.read_parquet(norm_path(split, s), columns=cols)


def s1_table(split, cols=None):
    return nt(split, '1', cols).with_row_index('i1')


def r_table(split, cols=None):
    """S2 then S3, row index = ir."""
    return pl.concat([nt(split, '2', cols), nt(split, '3', cols)]).with_row_index('ir')


def id_maps(split):
    m1 = nt(split, '1', ['entity_id']).with_row_index('i1').rename({'entity_id': 's1'})
    mr = pl.concat([nt(split, '2', ['entity_id']), nt(split, '3', ['entity_id'])]).with_row_index('ir').rename({'entity_id': 'rid'})
    return m1, mr


TRUTH_PATH = OUT / f'truth_train_{NORM_VERSION}.parquet'       # (i1, ir) positives of the ghosted train split
SPLITS_PATH = OUT / f'splits_train_{NORM_VERSION}.parquet'     # (i1, country, n, split)


def truth():
    return pl.read_parquet(TRUTH_PATH)


def splits():
    return pl.read_parquet(SPLITS_PATH)


def subset_i1(name):
    return splits().filter(pl.col('split') == name)['i1']


def truth_of(i1):
    return truth().join(pl.DataFrame({'i1': i1}).with_columns(pl.col('i1').cast(pl.UInt32)), on='i1', how='semi')
