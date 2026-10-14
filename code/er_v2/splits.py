"""Step 2: test-like train distribution ("ghost S1") and disjoint, stratified S1 subsets.

Test has 5.75 S2/S3 records per S1 against 4.68 in train, while its true matches per S1 are unchanged (~3.45,
estimated from exact-name hit rates), i.e. ~41 % of test records match no S1 against 26 % in train. We remove
GHOST_FRAC of train S1 entities (stratified by country x group size); their S2/S3 records stay in the pool as
distractors. Every train-side model, statistic and threshold is then fitted on this test-like distribution.
"""
import polars as pl
import numpy as np
from .config import OUT, SEED, GHOST_FRAC, NORM_VERSION, raw_path, log, save_json
from .normalize import read_tsv
from .data import norm_path, TRUTH_PATH, SPLITS_PATH

# disjoint subsets of the (ghosted) train S1, in this order; 'rest' gets the remainder
SUBSETS = [('rr_train', 100_000), ('rr_tune', 100_000), ('val', 100_000), ('stack_train', 200_000),
           ('ce_train', 400_000), ('bienc', 300_000)]


def run():
    if TRUTH_PATH.exists() and SPLITS_PATH.exists() and norm_path('train', '1').exists():
        log('splits cached'); return
    gt = read_tsv(raw_path('train_ground_truth'))
    pairs = (gt.select(pl.col('source1_entity_id').alias('s1'), pl.col('matched_entity_ids').str.split(',').alias('rid'))
               .explode('rid').filter(pl.col('rid').is_not_null() & (pl.col('rid') != '')))
    s1full = pl.read_parquet(norm_path('train_full', '1'))
    meta = (s1full.select(pl.col('entity_id').alias('s1'), 'country')
                  .join(pairs.group_by('s1').len('n'), on='s1', how='left').with_columns(pl.col('n').fill_null(0)))
    rng = np.random.default_rng(SEED)
    meta = meta.with_columns(u=pl.Series(rng.random(meta.height)), strat=pl.concat_str([pl.col('country'), pl.lit('_'),
                                                                                       pl.col('n').clip(0, 6).cast(pl.String)]))
    meta = meta.with_columns(q=(pl.col('u').rank('ordinal').over('strat') - 0.5) / pl.len().over('strat'))
    meta = meta.with_columns(ghost=pl.col('q') < GHOST_FRAC)

    # ghosted S1 table: same column layout, original order, ghost rows removed
    keep = meta.filter(~pl.col('ghost')).select('s1')
    s1 = s1full.join(keep.rename({'s1': 'entity_id'}), on='entity_id', how='semi')
    s1.write_parquet(norm_path('train', '1'))
    m1 = s1.select(pl.col('entity_id').alias('s1')).with_row_index('i1')
    mr = pl.concat([pl.read_parquet(norm_path('train', s), columns=['entity_id']) for s in '23']).with_row_index('ir').rename({'entity_id': 'rid'})
    tr = pairs.join(m1, on='s1').join(mr, on='rid').select('i1', 'ir')
    tr.write_parquet(TRUTH_PATH)

    # stratified disjoint subsets of the kept S1 (second independent random order)
    km = meta.filter(~pl.col('ghost')).join(m1, on='s1').select('i1', 's1', 'country', 'n', 'strat')
    km = km.with_columns(u=pl.Series(np.random.default_rng(SEED + 1).random(km.height)))
    km = km.with_columns(q=(pl.col('u').rank('ordinal').over('strat') - 0.5) / pl.len().over('strat'))
    edges, acc = [], 0.0
    for name, n in SUBSETS:
        edges.append((name, acc, acc + n / km.height)); acc += n / km.height
    e = pl.lit('rest')
    for name, lo, hi in reversed(edges):
        e = pl.when((pl.col('q') >= lo) & (pl.col('q') < hi)).then(pl.lit(name)).otherwise(e)
    km = km.with_columns(split=e).select('i1', 's1', 'country', 'n', 'split')
    km.write_parquet(SPLITS_PATH)

    n_r = mr.height
    info = {'train S1 full': meta.height, 'ghost S1': int(meta['ghost'].sum()), 'train S1 kept': s1.height,
            'records': n_r, 'records per kept S1': round(n_r / s1.height, 3),
            'matched share (kept)': round(tr.height / n_r, 4), 'distractor share (kept)': round(1 - tr.height / n_r, 4),
            'distractor share (full)': round(1 - pairs.height / n_r, 4)}
    log(info)
    log(km.group_by('split').agg(pl.len().alias('S1'), (pl.col('n') == 0).mean().mul(100).round(2).alias('singleton %'),
                                 pl.col('n').mean().round(3).alias('mean n'), (pl.col('country') == 'India').mean().round(4).alias('India share'))
          .sort('split'))
    save_json(info, OUT / 'ghost_info.json')


if __name__ == '__main__':
    run()
