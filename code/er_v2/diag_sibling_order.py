"""Do records of the same entity sit close together (row position or id number) in the S2/S3 files?
If yes, file order would disambiguate same-name address-less records (a data artefact, not a business signal)."""
import json
import numpy as np
import polars as pl
from .config import OUT
from .data import truth, nt

R = {}
TR = truth()
n2 = nt('train', '2', ['entity_id']).height
ids = pl.concat([nt('train', s, ['entity_id']) for s in '23']).with_row_index('ir').with_columns(
    num=pl.col('entity_id').str.slice(3).cast(pl.Int64), src=pl.col('entity_id').str.slice(0, 2))
t = TR.join(ids, on='ir')
rng = np.random.default_rng(0)
for src, lo, hi in [('S2', 0, n2), ('S3', n2, ids.height)]:
    g = t.filter(pl.col('src') == src)
    pairs = g.join(g, on='i1', suffix='_b').filter(pl.col('ir') < pl.col('ir_b'))
    d_row = (pairs['ir_b'] - pairs['ir']).abs().to_numpy()
    d_id = (pairs['num_b'] - pairs['num']).abs().to_numpy()
    rr = rng.integers(lo, hi, size=(200_000, 2))
    rand_row = np.abs(rr[:, 0] - rr[:, 1])
    nums = ids['num'].to_numpy()[lo:hi]
    ri = rng.integers(0, hi - lo, size=(200_000, 2))
    rand_id = np.abs(nums[ri[:, 0]] - nums[ri[:, 1]])
    R[src] = {'sibling pairs': len(d_row),
              'row gap median (siblings)': float(np.median(d_row)), 'row gap median (random)': float(np.median(rand_row)),
              'share row gap <= 10 (siblings)': round(float((d_row <= 10).mean()), 5), 'share row gap <= 10 (random)': round(float((rand_row <= 10).mean()), 5),
              'id gap median (siblings)': float(np.median(d_id)), 'id gap median (random)': float(np.median(rand_id)),
              'share id gap <= 1000 (siblings)': round(float((d_id <= 1000).mean()), 5), 'share id gap <= 1000 (random)': round(float((rand_id <= 1000).mean()), 5)}
    print(src, R[src])
json.dump(R, open(OUT / 'metrics' / 'diag_sibling_order.json', 'w'), indent=1)
print('wrote diag_sibling_order.json')
