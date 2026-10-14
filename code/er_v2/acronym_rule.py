"""Acronym assignment rule, applied after the final model (country-agnostic, decision layer only).

Train contains almost no acronym-style record names, so the learned models cannot learn that "FBP" is
"Fun Boutique Primaire SAS". Rule: a record whose normalised name is a single alphabetic token of 2-6 letters is
assigned to S1 s when
  * s is a candidate of the record and the initials of s's name_core (small words skipped) equal that token,
  * s is the ONLY candidate S1 of the record with that property,
  * the record is currently unassigned and no S1 claims it with p >= STRONG,
  * p(s, r) >= MIN_P.
Precision is measured with labels on the train evaluation set (stack_train + val + closure) before use.

    python -m er_v2.acronym_rule check <eval_p file> <t1> <t2>                   # train precision / counts
    python -m er_v2.acronym_rule apply <test_p file> <t1> <t2> <out_dir>         # write a submission variant
"""
import sys, json
import polars as pl
from .config import OUT, log, save_json
from .data import nt, truth
from .metrics import exclusive
from .diag_recall import initials

STRONG, MIN_P = 0.5, 0.001


def decide(d, t1, t2):
    x = exclusive(d.select('i1', 'ir', 'p')).with_columns(k=pl.col('p').rank('ordinal', descending=True).over('i1'))
    return x.filter(((pl.col('k') == 1) & (pl.col('p') >= t1)) | (pl.col('p') >= t2)).select('i1', 'ir')


def acronym_pairs(d, pred, split):
    """d: (i1, ir, p) for every candidate pair of the split; pred: (i1, ir) current decision. Returns pairs to add."""
    s1 = nt(split, '1', ['name_core']).with_row_index('i1').with_columns(ini=initials('name_core')).select(pl.col('i1').cast(pl.UInt32), 'ini')
    r = pl.concat([nt(split, s, ['name_core']) for s in '23']).with_row_index('ir').select(pl.col('ir').cast(pl.UInt32), pl.col('name_core').alias('nr'))
    r = r.filter(pl.col('nr').str.contains(r'^[a-z]{2,6}$'))
    c = (d.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32), 'p').join(r, on='ir').join(s1, on='i1')
          .with_columns(hit=(pl.col('ini') == pl.col('nr'))))
    c = c.with_columns(n_hit=pl.col('hit').sum().over('ir'), p_max=pl.col('p').max().over('ir'))
    assigned = pred.select(pl.col('ir').cast(pl.UInt32)).unique()
    add = (c.filter(pl.col('hit') & (pl.col('n_hit') == 1) & (pl.col('p_max') < STRONG) & (pl.col('p') >= MIN_P))
             .join(assigned, on='ir', how='anti'))
    stats = {'acronym-like records among candidates': c['ir'].n_unique(), 'with >=1 initials hit': c.filter('hit')['ir'].n_unique(),
             'with exactly 1 hit': c.filter(pl.col('hit') & (pl.col('n_hit') == 1))['ir'].n_unique(),
             'already assigned (any S1)': c.select('ir').unique().join(assigned, on='ir', how='semi').height, 'added by rule': add.height}
    return add.select('i1', 'ir'), stats


def check(ev_file, t1, t2):
    d = pl.read_parquet(OUT / ev_file).select('i1', 'ir', 'p')
    pred = decide(d, t1, t2)
    add, st = acronym_pairs(d, pred, 'train')
    tr = truth().select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32))
    hits = add.join(tr, on=['i1', 'ir'], how='semi').height
    # also: how precise is the rule's (unique initials hit) S1 on acronym records the model DID assign?
    st.update(added_correct=hits, added_precision=round(hits / max(add.height, 1), 4))
    log('TRAIN check:', json.dumps(st, indent=1))
    return st


def apply(test_file, t1, t2, out_dir):
    import os
    os.environ['ER_OUT_DIR'] = out_dir
    d = pl.read_parquet(OUT / test_file).select('i1', 'ir', 'p')
    pred = decide(d, t1, t2)
    add, st = acronym_pairs(d, pred, 'test')
    c1 = nt('test', '1', ['country']).with_row_index('i1').select(pl.col('i1').cast(pl.UInt32), 'country')
    st['added_by_country'] = {k: v for k, v in add.join(c1, on='i1').group_by('country').len().sort('country').iter_rows()}
    log('TEST apply:', json.dumps(st, indent=1))
    from . import config
    config.FINAL_OUT = __import__('pathlib').Path(out_dir); config.FINAL_OUT.mkdir(parents=True, exist_ok=True)
    from . import stack
    stack.FINAL_OUT = config.FINAL_OUT
    from .stage2 import stage2_path
    stack.write_submission(pl.concat([pred.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32)), add]),
                           pl.read_parquet(stage2_path('test'), columns=['i1', 'ir']),
                           {'rule': 'acronym variant', 'source': test_file, 't1': t1, 'threshold': t2, 'acronym': st})


if __name__ == '__main__':
    op = sys.argv[1]
    if op == 'check':
        check(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    else:
        apply(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
