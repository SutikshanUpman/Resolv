"""Step 3c v3: two-pass stage-2 re-ranker and a recall-oriented cut (run with ER_STAGE2=s3).

v2 kept rank <= 8 and p_rr >= 0.01 per S1. On val that cut drops ~2,350 true pairs, mostly address-less records whose
name several S1 share (their dense margin d_gap_r is ~0, so p_rr is low for every claimant) and records of S1 with
more than 8 true matches. v3:
  pass 1  the v2 re-ranker (stage2_lgb_s2a) scores every union pair -> q1                  (cached per S1 chunk)
  hints   from q1: record-side competition over ALL S1 (q1r_*), sibling evidence (does the record's name_core /
          skeleton equal that of a confident record of the same S1, best name similarity to them), acronym match
  pass 2  a new LightGBM on the v2 features + hints, fitted on 200k S1 of the unused `rest` subset (q1 is
          out-of-sample there), early-stopped on 50k other `rest` S1                          -> p_rr
  cut     keep (rank <= kmax & p >= floor) | p >= p_hi | (S1 among the record's top-R claimants & p >= floor)
          | [optional] (address-less record matching a confident sibling) | (acronym match), with p >= 0.001 for the last two.
          Chosen on rr_tune: best oracle F0.5 with <= BUDGET candidates per S1, then the smallest set within 0.0003.
Every quantity is computed identically on train and test from that split's own data; countries are only group keys.
"""
import gc, time, json, sys
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import fuzz
from .config import OUT, SEED, STAGE2_VERSION, log, save_json
from .data import splits, truth, subset_i1
from .metrics import candidate_report
from .stage2 import (union_for, side_tables, fast_features, label, RR_FEATS, S3_FEATS, stage2_path, cp)
from .diag_recall import initials

assert STAGE2_VERSION >= 's3', 'run with ER_STAGE2=s3'
P1_MODEL = OUT / 'models' / 'stage2_lgb_s2a.txt'
TMP = OUT / 'blocking' / f'{STAGE2_VERSION}_work'
M2_PATH = OUT / 'models' / f'stage2_lgb_{STAGE2_VERSION}.txt'
CFG_PATH = OUT / 'models' / f'stage2_cfg_{STAGE2_VERSION}.json'
FEATS2 = RR_FEATS + S3_FEATS
CHUNK = 150_000
CORE_T = 0.9            # confident sibling threshold on q1
TSET_TOPK = 20          # name similarity to siblings only for the top-20 pass-1 candidates of each S1
BUDGET = 6.0            # max candidates per S1 on rr_tune
N_FIT, N_ES = 200_000, 50_000


def name_tables(s1f, rf):
    s1x = s1f.select('i1', 'name_core', 'name_sq').with_columns(ini=initials('name_core'))
    rx = rf.select('ir', pl.col('name_core').alias('name_core_r'), pl.col('name_sq').alias('name_sq_r'),
                   pl.col('name_sk').alias('name_sk_r'), pl.col('addr_norm').is_null().cast(pl.Int8).alias('r_noaddr')
                   ).with_columns(ini_r=initials('name_core_r'))
    return s1x, rx


def pass1(split, m1, s1f, rf, rs):
    """Fast features + q1 for every union pair, written per S1 chunk (the expensive part, ~12 min per split)."""
    TMP.mkdir(parents=True, exist_ok=True)
    n1 = s1f.height; t = time.time()
    for k, a in enumerate(range(0, n1, CHUNK)):
        p = TMP / f'{split}_p1_{k:03d}.parquet'
        if p.exists():
            continue
        f = fast_features(union_for(split, np.arange(a, min(a + CHUNK, n1), dtype=np.uint32)), s1f, rf, rs)
        f = f.with_columns(q1=pl.Series(m1.predict(f.select(RR_FEATS).to_numpy().astype(np.float32)).astype(np.float32)))
        f.write_parquet(p); del f; gc.collect()
    log(f'pass 1 {split}: done in {(time.time() - t) / 60:.1f} min')


def record_side(split, col, name):
    """Rank / gap / count of every pair among ALL S1 claiming the same record, from score `col` (whole split)."""
    p = TMP / f'{split}_{name}.parquet'
    if p.exists():
        return pl.read_parquet(p)
    d = pl.scan_parquet(TMP / f'{split}_{"p1" if col == "q1" else "p2"}_*.parquet').select('i1', 'ir', col).collect()
    d = d.with_columns(_m=pl.col(col).max().over('ir'),
                       _2=pl.when(pl.len().over('ir') > 1).then(pl.col(col).top_k(2).min().over('ir')).otherwise(0.0))
    pre = 'q1r' if col == 'q1' else 'p2r'
    d = d.select('i1', 'ir',
                 pl.col(col).rank('ordinal', descending=True).over('ir').cast(pl.UInt16).alias(f'{pre}_rank'),
                 pl.when(pl.col(col) >= pl.col('_m')).then(pl.col(col) - pl.col('_2')).otherwise(pl.col(col) - pl.col('_m')).alias(f'{pre}_gap'),
                 (pl.col(col) >= 0.5).sum().over('ir').cast(pl.UInt16).alias(f'{pre}_n05'))
    d = d.sort('i1'); d.write_parquet(p)
    return d


def hints(f, s1x, rx, rq):
    """Add S3_FEATS to a chunk of pass-1 rows (complete S1 groups)."""
    f = f.join(rq, on=['i1', 'ir'], how='left')
    f = f.with_columns(
        q1_rank_s=pl.col('q1').rank('ordinal', descending=True).over('i1').cast(pl.UInt16),
        _m=pl.col('q1').max().over('i1'),
        _2=pl.when(pl.len().over('i1') > 1).then(pl.col('q1').top_k(2).min().over('i1')).otherwise(0.0))
    f = f.with_columns(q1_gap_s=pl.when(pl.col('q1') >= pl.col('_m')).then(pl.col('q1') - pl.col('_2'))
                       .otherwise(pl.col('q1') - pl.col('_m'))).drop('_m', '_2')
    f = f.join(s1x, on='i1', how='left').join(rx, on='ir', how='left')
    # confident siblings: q1 >= CORE_T and this S1 is the record's top claimant
    f = f.with_columns(_core=((pl.col('q1') >= CORE_T) & (pl.col('q1r_rank') == 1)).cast(pl.UInt32))
    f = f.with_columns(sib_n=(pl.col('_core').sum().over('i1') - pl.col('_core')).cast(pl.UInt16))
    for key, out in [('name_core_r', 'sib_eq_core'), ('name_sk_r', 'sib_eq_sk')]:
        c = f.filter(pl.col('_core') == 1).group_by('i1', key).len('_n')
        f = f.join(c, on=['i1', key], how='left').with_columns(
            ((pl.col('_n').fill_null(0) - pl.col('_core')).clip(0) * (pl.col(key).fill_null('') != '').cast(pl.UInt32))
            .cast(pl.UInt16).alias(out)).drop('_n')
    core = f.filter(pl.col('_core') == 1).select('i1', pl.col('ir').alias('ir_c'), pl.col('name_core_r').alias('nc'))
    top = f.filter(pl.col('q1_rank_s') <= TSET_TOPK).select('i1', 'ir', 'name_core_r')
    j = top.join(core, on='i1').filter(pl.col('ir') != pl.col('ir_c'))
    if j.height:
        j = j.with_columns(s=pl.Series(cp(j['name_core_r'].fill_null('').to_list(), j['nc'].fill_null('').to_list(), fuzz.token_set_ratio)))
        f = f.join(j.group_by('i1', 'ir').agg(pl.col('s').max().alias('sib_tset')), on=['i1', 'ir'], how='left')
    else:
        f = f.with_columns(sib_tset=pl.lit(None, dtype=pl.Float32))
    a = pl.col('name_sq_r').str.len_chars().is_between(2, 6) & (pl.col('name_sq_r') == pl.col('ini'))
    b = pl.col('name_sq').str.len_chars().is_between(2, 6) & (pl.col('name_sq') == pl.col('ini_r'))
    ok = (pl.col('ini').str.len_chars() >= 2) & (pl.col('ini_r').str.len_chars() >= 2)
    pa = (pl.col('name_sq_r').str.len_chars() >= 2) & pl.col('ini').str.starts_with(pl.col('name_sq_r'))
    pb = (pl.col('name_sq').str.len_chars() >= 2) & pl.col('ini_r').str.starts_with(pl.col('name_sq'))
    f = f.with_columns(sib_tset=pl.col('sib_tset').fill_null(-1.0).cast(pl.Float32),
                       acr_eq=((a | b) & ok).fill_null(False).cast(pl.Int8),
                       acr_pre=((pa | pb) & ok).fill_null(False).cast(pl.Int8))
    return f.drop('name_core', 'name_sq', 'ini', 'name_core_r', 'name_sq_r', 'name_sk_r', 'ini_r', '_core')


def read_subset(split, i1, s1x, rx, rq):
    """Pass-1 rows + hints of the given S1 (reads only the chunks that contain them)."""
    i1 = np.sort(np.asarray(i1)); keep = pl.DataFrame({'i1': i1.astype(np.uint32)})
    out = []
    for k in np.unique(i1 // CHUNK):
        f = pl.read_parquet(TMP / f'{split}_p1_{k:03d}.parquet').join(keep, on='i1', how='semi')
        out.append(hints(f, s1x, rx, rq.join(keep, on='i1', how='semi')))
    return pl.concat(out)


def train_pass2(s1x, rx, rq):
    if M2_PATH.exists():
        log('pass-2 model cached'); return lgb.Booster(model_file=str(M2_PATH))
    rest = subset_i1('rest').to_numpy()
    rng = np.random.default_rng(SEED + 7); rest = rng.permutation(rest)
    fit, es = rest[:N_FIT], rest[N_FIT:N_FIT + N_ES]
    t = time.time()
    tr = label(read_subset('train', fit, s1x, rx, rq)); va = label(read_subset('train', es, s1x, rx, rq))
    log(f'pass-2 data: fit {tr.height:,} pairs ({tr["y"].sum():,} pos), early-stop {va.height:,} in {(time.time() - t) / 60:.1f} min')
    params = dict(objective='binary', learning_rate=0.06, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=28, seed=SEED)
    m = lgb.train(params, lgb.Dataset(tr.select(FEATS2).to_numpy().astype(np.float32), tr['y'].to_numpy()), 2000,
                  valid_sets=[lgb.Dataset(va.select(FEATS2).to_numpy().astype(np.float32), va['y'].to_numpy())],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    m.save_model(str(M2_PATH), num_iteration=m.best_iteration)
    imp = sorted(zip(m.feature_importance('gain'), FEATS2), reverse=True); tot = sum(v for v, _ in imp)
    log(f'pass-2 LightGBM best iteration {m.best_iteration} in {(time.time() - t) / 60:.1f} min; top gain:',
        [(f, round(float(v) / tot, 3)) for v, f in imp[:15]])
    return lgb.Booster(model_file=str(M2_PATH))


def pass2(split, m2, s1x, rx, rq):
    """p2 for every union pair, written per chunk with all features (kept rows are read back after the cut)."""
    t = time.time(); k = 0
    while (TMP / f'{split}_p1_{k:03d}.parquet').exists():
        p = TMP / f'{split}_p2_{k:03d}.parquet'
        if not p.exists():
            f = pl.read_parquet(TMP / f'{split}_p1_{k:03d}.parquet')
            f = hints(f, s1x, rx, rq.filter(pl.col('i1').is_between(int(f['i1'].min()), int(f['i1'].max()))))
            f = f.with_columns(p2=pl.Series(m2.predict(f.select(FEATS2).to_numpy().astype(np.float32)).astype(np.float32)))
            f = f.with_columns(rr_rank=pl.col('p2').rank('ordinal', descending=True).over('i1').cast(pl.UInt16))
            f.write_parquet(p); del f; gc.collect()
        k += 1
    log(f'pass 2 {split}: done in {(time.time() - t) / 60:.1f} min')


def keep_expr(kmax, floor, p_hi, R, S=0):
    e = ((pl.col('rr_rank') <= kmax) & (pl.col('p2') >= floor)) | (pl.col('p2') >= p_hi)
    if R:
        e = e | ((pl.col('p2r_rank') <= R) & (pl.col('p2') >= floor))
    if S:   # force-keep address-less records sharing a confident sibling's name (val diag: costly, so it is a grid option)
        e = e | ((pl.col('r_noaddr') == 1) & (pl.col('sib_eq_core') >= 1) & (pl.col('p2') >= 0.001))
    return e | ((pl.col('acr_eq') == 1) & (pl.col('p2') >= 0.001))


def slim(split, rr):
    d = pl.scan_parquet(TMP / f'{split}_p2_*.parquet').select('i1', 'ir', 'p2', 'rr_rank', 'r_noaddr', 'sib_eq_core', 'acr_eq').collect()
    return d.join(rr, on=['i1', 'ir'], how='left')


def tune(S_tr):
    if CFG_PATH.exists():
        return json.load(open(CFG_PATH))
    TU = subset_i1('rr_tune'); TT = truth().join(pl.DataFrame({'i1': TU}), on='i1', how='semi')
    tu = S_tr.join(pl.DataFrame({'i1': TU}).with_columns(pl.col('i1').cast(pl.UInt32)), on='i1', how='semi')
    base = candidate_report(tu, TT, TU, 'union')
    rows = [base, candidate_report(tu.filter((pl.col('rr_rank') <= 8) & (pl.col('p2') >= 0.01)), TT, TU, 'v2 rule on p2 (rank<=8, p>=0.01)')]
    grid = []
    for kmax in [7, 8, 10, 12]:
        for floor in [0.003, 0.005, 0.01, 0.02]:
            for p_hi in [0.15, 1.1]:
                for R in [0, 1, 2]:
                    for S_ in [0, 1]:
                        r = candidate_report(tu.filter(keep_expr(kmax, floor, p_hi, R, S_)), TT, TU, f'k{kmax} f{floor} hi{p_hi} R{R} S{S_}')
                        r.update(kmax=kmax, floor=floor, p_hi=p_hi, R=R, S=S_); grid.append(r)
    G = pl.DataFrame(grid)
    ok = G.filter(pl.col('cand / S1') <= BUDGET)
    best = ok['oracle F0.5'].max()
    pick = ok.filter(pl.col('oracle F0.5') >= best - 0.0003).sort('cand / S1').row(0, named=True)
    front = G.sort('cand / S1').filter(pl.col('oracle F0.5') >= pl.col('oracle F0.5').cum_max())
    with pl.Config(tbl_rows=80, tbl_width_chars=220, fmt_str_lengths=50):
        log('\n' + str(pl.DataFrame(rows).drop('S1')))
        log('frontier (rr_tune):\n' + str(front.select('stage', 'cand / S1', 'pair recall', 'oracle F0.5')))
    cfg = {k: pick[k] for k in ['kmax', 'floor', 'p_hi', 'R', 'S']}
    cfg.update(tune_oracle=pick['oracle F0.5'], tune_recall=pick['pair recall'], tune_cand_per_s1=pick['cand / S1'],
               union_oracle=base['oracle F0.5'], budget=BUDGET)
    save_json(cfg, CFG_PATH); G.write_csv(OUT / f'stage2_grid_{STAGE2_VERSION}.csv')
    log('CHOSEN', cfg)
    return cfg


def materialise(split, S, cfg):
    p = stage2_path(split)
    if p.exists():
        log(p.name, 'cached'); return
    keep = S.filter(keep_expr(cfg['kmax'], cfg['floor'], cfg['p_hi'], cfg['R'], cfg['S'])).select('i1', 'ir')
    outs = []; k = 0
    while (TMP / f'{split}_p2_{k:03d}.parquet').exists():
        f = pl.read_parquet(TMP / f'{split}_p2_{k:03d}.parquet').join(keep, on=['i1', 'ir'], how='semi')
        outs.append(f.select(['i1', 'ir'] + FEATS2 + [pl.col('p2').alias('p_rr'), 'rr_rank'])); k += 1
    out = pl.concat(outs); out.write_parquet(p)
    log(f'{p.name}: {out.height:,} pairs, {out.height / out["i1"].n_unique():.2f} per S1 with candidates')


def run():
    t = time.time()
    m1 = lgb.Booster(model_file=str(P1_MODEL))
    tabs = {}
    for split in ['train', 'test']:
        s1f, rf, rs = side_tables(split)
        pass1(split, m1, s1f, rf, rs)
        tabs[split] = name_tables(s1f, rf); del s1f, rf, rs; gc.collect()
    rq = {s: record_side(s, 'q1', 'q1r') for s in ['train', 'test']}
    m2 = train_pass2(*tabs['train'], rq['train'])
    for split in ['train', 'test']:
        pass2(split, m2, *tabs[split], rq[split])
    S = {s: slim(s, record_side(s, 'p2', 'p2r')) for s in ['train', 'test']}
    cfg = tune(S['train'])
    for split in ['train', 'test']:
        materialise(split, S[split], cfg)
    # report: val (never used for fitting or tuning) v2 vs v3, test candidates per S1 by country
    V = subset_i1('val'); TV = truth().join(pl.DataFrame({'i1': V}), on='i1', how='semi')
    rows = [candidate_report(pl.read_parquet(OUT / 'blocking' / 'train_stage2_s2a.parquet', columns=['i1', 'ir']), TV, V, 'val: v2 candidates'),
            candidate_report(pl.read_parquet(stage2_path('train'), columns=['i1', 'ir']), TV, V, f'val: {STAGE2_VERSION} candidates')]
    meta = splits().filter(pl.col('split') == 'val')
    for c in meta['country'].unique().sort().to_list():
        rows.append(candidate_report(pl.read_parquet(stage2_path('train'), columns=['i1', 'ir']), TV, meta.filter(pl.col('country') == c)['i1'], f'val {STAGE2_VERSION} [{c}]'))
    from .data import nt
    te = pl.read_parquet(stage2_path('test'), columns=['i1']).group_by('i1').len()
    c1 = nt('test', '1', ['country']).with_row_index('i1')
    dist = c1.join(te, on='i1', how='left').with_columns(pl.col('len').fill_null(0)).group_by('country').agg(
        pl.len().alias('S1'), pl.col('len').mean().round(3).alias('cand/S1'), (pl.col('len') == 0).mean().alias('empty share')).sort('country')
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        log('VAL candidates:\n' + str(pl.DataFrame(rows).drop('S1')))
        log('TEST candidates per S1:\n' + str(dist))
    save_json({'val': rows, 'test': dist.to_dicts(), 'cfg': cfg}, OUT / 'metrics' / f'stage2_{STAGE2_VERSION}_report.json')
    log(f'stage 2 {STAGE2_VERSION} done in {(time.time() - t) / 60:.1f} min')


if __name__ == '__main__':
    run()
