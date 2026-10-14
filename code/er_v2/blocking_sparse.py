"""Step 3a: IDF-weighted multi-key sparse blocking (inverted index, polars), one subprocess per (split, country).

Every record emits hashed keys (country | family | value). Keys with more than CAP postings in S2+S3 are dropped
(uninformative, and they would explode the join). The pair score is the sum of IDF over shared keys, the top K1 per S1
are kept, and per-family IDF sums are kept as features.

Families (v1 in capitals, new in v2 in lower case):
  t name tokens | n sorted token pairs | q squashed name | p 6-char prefix | a address tokens | b address bigrams
  z ZIP/PIN | m numbers (>=3 digits, de-zeroed) |
  k phonetic skeleton of the name (survives Indic transliteration)
  c name token x locality token (makes generic names like "Internal Medicine" selective within a city)
  h house number x street/locality token (order-free, survives a typo in one token)
"""
import sys, json, math, re, gc, time, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import polars as pl

KEY_FAMILIES = ['t', 'n', 'q', 'p', 'a', 'b', 'z', 'm', 'k', 'c', 'h']
BLOCK_COLS = ['entity_id', 'country', 'name_core', 'name_sq', 'name_sk', 'addr_norm', 'loc', 'zip', 'addr_nums']
CAP = 300
K1 = 50
MAX_JOIN_ROWS = 60e6
safe = lambda c: re.sub(r'[^A-Za-z0-9]', '_', c)


def explode_keys(df, idc):
    """One row per (record, key); key = hash(country | family | value)."""
    toks = pl.col('name_core').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique(maintain_order=True).list.head(6)
    stoks = pl.col('name_core').str.split(' ').list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique().list.sort().list.head(6)
    atoks = pl.col('addr_norm').fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element() != ''))
    alpha = atoks.list.eval(pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().str.contains(r'\d'))).list.unique(maintain_order=True)
    loc = pl.col('loc').fill_null(pl.lit([], dtype=pl.List(pl.String)))
    single = {
        't': toks,
        'n': pl.concat_list(stoks.list.eval((pl.element() + pl.lit(' ') + pl.element().shift(-1)).drop_nulls()),
                            stoks.list.eval((pl.element() + pl.lit(' ') + pl.element().shift(-2)).drop_nulls())).list.unique(),
        'q': pl.concat_list(pl.col('name_sq')).list.eval(pl.element().filter(pl.element().str.len_chars() >= 4)),
        'p': pl.concat_list(pl.col('name_sq').str.slice(0, 6)).list.eval(pl.element().filter(pl.element().str.len_chars() >= 5)),
        'a': alpha.list.eval(pl.element().filter(pl.element().str.len_chars() >= 4)).list.head(8),
        'b': atoks.list.eval((pl.element() + pl.lit(' ') + pl.element().shift(-1)).drop_nulls()).list.unique(maintain_order=True).list.head(10),
        'z': pl.concat_list(pl.col('zip')).list.drop_nulls(),
        'm': pl.col('addr_nums').list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.head(4),
        'k': pl.concat_list(pl.col('name_sk')).list.eval(pl.element().filter(pl.element().str.len_chars() >= 4)),
    }
    cross = {  # (left list, right list): cartesian product per record
        'c': (pl.concat_list(toks.list.head(3), pl.concat_list(pl.col('name_sq'))), loc.list.head(4)),
        'h': (pl.concat_list(pl.col('addr_nums').list.first()).list.drop_nulls(), pl.concat_list(alpha.list.head(4), loc.list.head(2)).list.unique()),
    }
    parts = []
    for f, e in single.items():
        j = KEY_FAMILIES.index(f)
        parts.append(df.select(idc, 'country', e.alias('v')).explode('v').drop_nulls('v')
                       .select(idc, pl.lit(j, dtype=pl.UInt8).alias('fam'),
                               pl.concat_str([pl.col('country'), pl.lit(f'|{f}|'), pl.col('v')]).hash(seed=7).alias('key')))
    for f, (x, y) in cross.items():
        j = KEY_FAMILIES.index(f)
        parts.append(df.select(idc, 'country', x.alias('x'), y.alias('y')).explode('x').explode('y').drop_nulls(['x', 'y'])
                       .select(idc, pl.lit(j, dtype=pl.UInt8).alias('fam'),
                               pl.concat_str([pl.col('country'), pl.lit(f'|{f}|'), pl.col('x'), pl.lit('|'), pl.col('y')]).hash(seed=7).alias('key')))
    return pl.concat(parts).unique([idc, 'key'])


def block_country(norm1, norm2, norm3, out_dir, country, cap, k1, max_join_rows):
    """Top-k1 candidates per S1 of one country, IDF-scored, written as parquet parts with integer ids (i1, ir)."""
    t = time.time()
    s1 = pl.scan_parquet(norm1).select(BLOCK_COLS).with_row_index('i1').filter(pl.col('country') == country).collect()
    r = (pl.concat([pl.scan_parquet(norm2).select(BLOCK_COLS), pl.scan_parquet(norm3).select(BLOCK_COLS)])
           .with_row_index('ir').filter(pl.col('country') == country).collect())
    n_s1, n_r = s1.height, r.height
    k1_ = explode_keys(s1, 'i1'); del s1
    kr = explode_keys(r, 'ir'); del r; gc.collect()
    dfr = kr.group_by('key').agg(pl.len().alias('df')).filter(pl.col('df') <= cap)
    dfr = dfr.with_columns(idf=((n_r + 1) / (pl.col('df') + 1)).log().cast(pl.Float32))
    kr = kr.join(dfr.select('key', 'idf'), on='key').select('key', 'ir', 'idf')
    k1_ = k1_.join(dfr.select('key', 'df'), on='key')
    n_keys = dfr.height; del dfr; gc.collect()
    join_rows = int(k1_['df'].sum())
    n_part = max(1, int(math.ceil(join_rows / max_join_rows)))
    n_c = 0
    for p in range(n_part):
        j = k1_.filter(pl.col('i1') % n_part == p).join(kr, on='key')
        agg = j.group_by(['i1', 'ir']).agg(
            pl.col('idf').sum().alias('bscore'), pl.len().cast(pl.UInt16).alias('bkeys'),
            *[pl.col('idf').filter(pl.col('fam') == jj).sum().alias(f'b_{f}') for jj, f in enumerate(KEY_FAMILIES)])
        del j
        agg = (agg.with_columns(pl.col('bscore').rank('ordinal', descending=True).over('i1').cast(pl.UInt16).alias('brank'))
                  .filter(pl.col('brank') <= k1))
        agg.write_parquet(f'{out_dir}/part_{safe(country)}_{p:04d}.parquet')
        n_c += agg.height; del agg; gc.collect()
    import psutil
    print(f'{country}: S1 {n_s1:,} | R {n_r:,} | usable keys {n_keys:,} | join rows {join_rows:,} in {n_part} parts | '
          f'cands {n_c:,} | peak RSS {psutil.Process().memory_info().rss / 2**30:.1f} GiB | {time.time() - t:.0f}s', flush=True)
    open(f'{out_dir}/_DONE_{safe(country)}', 'w').close()


def record_stats(out_dir, out_path):
    """Record-side competition stats over ALL S1 lists: how many lists contain the record, its best claim, its runner-up."""
    q = (pl.scan_parquet(f'{out_dir}/part_*.parquet').select('ir', 'bscore').group_by('ir')
           .agg(pl.len().cast(pl.UInt16).alias('r_n'), pl.col('bscore').max().alias('r_max'),
                pl.when(pl.len() > 1).then(pl.col('bscore').top_k(2).min()).otherwise(0.0).alias('r_2nd')))
    q.collect(engine='streaming').write_parquet(out_path)


# ------------------------------------------------------------------ driver (runs in the main process)
def stage1_dir(split):
    from .config import OUT, BLOCK_VERSION
    return OUT / 'blocking' / f'{split}_stage1_{BLOCK_VERSION}_cap{CAP}_k{K1}'


def rstats_path(split):
    from .config import OUT, BLOCK_VERSION
    return OUT / 'blocking' / f'{split}_rstats_{BLOCK_VERSION}_cap{CAP}_k{K1}.parquet'


def _run_worker(task):
    r_ = subprocess.run([sys.executable, '-m', 'er_v2.blocking_sparse', json.dumps(task, default=str)],
                        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    if r_.returncode != 0:
        raise RuntimeError(f'stage-1 worker failed ({task}):\n{r_.stderr[-3000:]}')
    return r_.stdout.strip()


def run(max_parallel=5):
    from .data import norm_path, nt
    from .config import log
    tasks = []
    for split in ['train', 'test']:
        d = stage1_dir(split); d.mkdir(parents=True, exist_ok=True)
        for c in nt(split, '1', ['country'])['country'].unique().sort().to_list():   # open set of countries
            if (d / f'_DONE_{safe(c)}').exists():
                log(f'{split}/{c}: cached'); continue
            for f in d.glob(f'part_{safe(c)}_*.parquet'):
                f.unlink()
            tasks.append({'op': 'country', 'norm1': norm_path(split, '1'), 'norm2': norm_path(split, '2'), 'norm3': norm_path(split, '3'),
                          'out_dir': d, 'country': c, 'cap': CAP, 'k1': K1, 'max_join_rows': MAX_JOIN_ROWS})
    with ThreadPoolExecutor(max_parallel) as ex:
        for out in ex.map(_run_worker, tasks):
            log(out)
    for split in ['train', 'test']:
        if not rstats_path(split).exists():
            log(_run_worker({'op': 'rstats', 'out_dir': stage1_dir(split), 'out_path': rstats_path(split)}) or f'{split} rstats done')


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1].startswith('{'):
    task = json.loads(sys.argv[1]); op = task.pop('op')
    {'country': block_country, 'rstats': record_stats}[op](**task)
