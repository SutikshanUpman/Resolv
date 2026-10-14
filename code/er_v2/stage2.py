"""Step 3c: union of sparse + dense candidates, LightGBM re-ranker, adaptive per-S1 cut -> final candidate set.

The union (sparse top-50 plus dense S1->R top-40 plus R->S1 top-3) has a high recall ceiling but ~80 pairs per S1.
A LightGBM re-ranker over cheap features (blocking evidence, dense cosine and ranks in both directions, rapidfuzz
similarities, record-side competition) scores every union pair; a candidate is kept iff rank <= kmax and p >= floor.
(kmax, floor) is chosen on rr_tune with a rule stated up front: the SMALLEST mean candidates per S1 whose oracle F0.5
stays within TOL of the union ceiling. The kept set is candidate_pairs.tsv (portal: smaller sets rank higher).
"""
import gc, time, json
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler
from .config import OUT, SEED, STAGE2_VERSION, BLOCK_VERSION, DENSE_VERSION, log, save_json
from .data import nt, splits, truth, subset_i1
from .metrics import candidate_report
from .blocking_sparse import KEY_FAMILIES, stage1_dir, rstats_path
from .dense import dense_path, EMB_DIR

BLK = OUT / 'blocking'
TOL = 0.002
FAST_COLS = ['name_core', 'name_sq', 'name_sk', 'addr_norm', 'zip', 'loc', 'addr_nums']
RR_FEATS = (['bscore', 'bkeys', 'brank', 'b_rel', 's_n', 'r_n', 'r_rel', 'r_best', 'r_gap2'] + [f'b_{f}' for f in KEY_FAMILIES] +
            ['n_ratio', 'n_tset', 'n_jw', 'a_tset', 'a_miss', 'zip_eq', 'zip_miss', 'src3',
             'dcos', 'drank_s', 'drank_r', 'd_gap_s', 'd_gap_r', 'r_dn', 'in_sparse', 'in_dense', 'n_sk_eq', 'loc_jacc', 'num1_eq'])
# v3 (stage2_v3.py): pass-1 evidence, record-side competition, sibling and acronym features added to the re-ranker
S3_FEATS = ['q1', 'q1_rank_s', 'q1_gap_s', 'q1r_rank', 'q1r_gap', 'q1r_n05', 'sib_n', 'sib_eq_core', 'sib_eq_sk', 'sib_tset',
            'acr_eq', 'acr_pre', 'r_noaddr']
RR_FEATS_ALL = RR_FEATS + (S3_FEATS if STAGE2_VERSION >= 's3' else [])
RR_PATH = OUT / 'models' / f'stage2_lgb_{STAGE2_VERSION}.txt'
CFG_PATH = OUT / 'models' / f'stage2_cfg_{STAGE2_VERSION}.json'


def union_path(split):
    return BLK / f'{split}_union_{BLOCK_VERSION}_{DENSE_VERSION}.parquet'


def ustats_path(split):
    return BLK / f'{split}_ustats_{BLOCK_VERSION}_{DENSE_VERSION}.parquet'


def stage2_path(split):
    return BLK / f'{split}_stage2_{STAGE2_VERSION}.parquet'


def cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def build_union(split):
    """sparse (full outer) dense, dcos filled from the stored embeddings for sparse-only pairs; sorted by i1."""
    p = union_path(split)
    if p.exists():
        log(p.name, 'cached'); return
    t = time.time()
    sp = pl.scan_parquet(stage1_dir(split) / 'part_*.parquet').collect()
    de = pl.read_parquet(dense_path(split))
    u = sp.join(de, on=['i1', 'ir'], how='full', coalesce=True)
    del sp, de; gc.collect()
    u = u.with_columns(in_sparse=pl.col('bscore').is_not_null().cast(pl.Int8), in_dense=pl.col('dcos').is_not_null().cast(pl.Int8))
    u = u.with_columns(pl.col('bscore').fill_null(0.0), pl.col('bkeys').fill_null(0), pl.col('brank').fill_null(99),
                       *[pl.col(f'b_{f}').fill_null(0.0) for f in KEY_FAMILIES],
                       pl.col('drank_s').fill_null(99), pl.col('drank_r').fill_null(99))
    # fill the dense cosine of sparse-only pairs from the embeddings (GPU gather + dot)
    miss = u['dcos'].is_null().to_numpy()
    if miss.any():
        import torch
        E1 = torch.from_numpy(np.load(EMB_DIR / f'{split}_s1.npy')).cuda(); ER = torch.from_numpy(np.load(EMB_DIR / f'{split}_r.npy')).cuda()
        a = u['i1'].to_numpy()[miss].astype(np.int64); b = u['ir'].to_numpy()[miss].astype(np.int64)
        out = np.empty(len(a), np.float32)
        for k in range(0, len(a), 4_000_000):
            ia = torch.from_numpy(a[k:k + 4_000_000]).cuda(); ib = torch.from_numpy(b[k:k + 4_000_000]).cuda()
            out[k:k + 4_000_000] = (E1[ia] * ER[ib]).sum(1, dtype=torch.float32).cpu().numpy()
        dc = u['dcos'].to_numpy().copy(); dc[miss] = out
        u = u.with_columns(dcos=pl.Series(dc, dtype=pl.Float32))
        del E1, ER; torch.cuda.empty_cache()
    u = u.sort(['i1', 'ir'])
    u.write_parquet(p, row_group_size=2_000_000)
    rs = u.group_by('ir').agg(pl.col('dcos').max().alias('r_dmax'),
                              pl.when(pl.len() > 1).then(pl.col('dcos').top_k(2).min()).otherwise(-1.0).alias('r_d2'),
                              pl.len().cast(pl.UInt16).alias('r_dn'))
    rs.join(pl.read_parquet(rstats_path(split)), on='ir', how='left').write_parquet(ustats_path(split))
    log(f'{p.name}: {u.height:,} pairs ({u.height / u["i1"].n_unique():.1f} per S1) in {(time.time() - t) / 60:.1f} min')


def side_tables(split):
    s1f = nt(split, '1', FAST_COLS).with_row_index('i1')
    rf = pl.concat([nt(split, s, FAST_COLS + ['src']) for s in '23']).with_row_index('ir')
    return s1f, rf, pl.read_parquet(ustats_path(split))


def fast_features(c: pl.DataFrame, s1f, rf, rs) -> pl.DataFrame:
    d = (c.join(s1f, on='i1').join(rf, on='ir', suffix='_r').join(rs, on='ir', how='left')
          .with_columns(s_max=pl.col('bscore').max().over('i1'), s_n=pl.len().over('i1'), s_dmax=pl.col('dcos').max().over('i1')))
    a_miss = (d['addr_norm'].is_null() | d['addr_norm_r'].is_null()).to_numpy()
    a_tset = cp(d['addr_norm'].fill_null('').to_list(), d['addr_norm_r'].fill_null('').to_list(), fuzz.token_set_ratio)
    a_tset[a_miss] = -1
    zip_miss = (d['zip'].is_null() | d['zip_r'].is_null()).to_numpy()
    zip_eq = (d['zip'] == d['zip_r']).fill_null(False).to_numpy().astype(np.float32); zip_eq[zip_miss] = -1
    n1, n2 = d['name_core'].to_list(), d['name_core_r'].to_list()
    l1, l2 = pl.col('loc').fill_null([]), pl.col('loc_r').fill_null([])
    return d.select(
        'i1', 'ir', pl.col('bscore').log1p(), 'bkeys', 'brank', (pl.col('bscore') / pl.col('s_max').clip(1e-6)).alias('b_rel'), 's_n',
        pl.col('r_n').fill_null(0), (pl.col('bscore') / pl.col('r_max').fill_null(1e-6).clip(1e-6)).alias('r_rel'),
        (pl.col('bscore') >= pl.col('r_max').fill_null(0)).cast(pl.Int8).alias('r_best'),
        (pl.col('bscore') - pl.col('r_2nd').fill_null(0)).alias('r_gap2'),
        *[f'b_{f}' for f in KEY_FAMILIES], (pl.col('src') == 3).cast(pl.Int8).alias('src3'),
        'dcos', 'drank_s', 'drank_r', (pl.col('dcos') - pl.col('s_dmax')).alias('d_gap_s'),
        pl.when(pl.col('dcos') >= pl.col('r_dmax')).then(pl.col('dcos') - pl.col('r_d2')).otherwise(pl.col('dcos') - pl.col('r_dmax')).alias('d_gap_r'),
        pl.col('r_dn').fill_null(0), 'in_sparse', 'in_dense',
        ((pl.col('name_sk') == pl.col('name_sk_r')) & (pl.col('name_sk') != '')).cast(pl.Int8).alias('n_sk_eq'),
        pl.when(l1.list.len() == 0).then(-1.0).when(l2.list.len() == 0).then(-1.0)
          .otherwise(l1.list.set_intersection(l2).list.len() / l1.list.set_union(l2).list.len()).cast(pl.Float32).alias('loc_jacc'),
        pl.when(pl.col('addr_nums').list.len().fill_null(0) == 0).then(-1)
          .when(pl.col('addr_nums_r').list.len().fill_null(0) == 0).then(-1)
          .otherwise((pl.col('addr_nums').list.first() == pl.col('addr_nums_r').list.first()).cast(pl.Int8)).alias('num1_eq'),
    ).with_columns(
        n_ratio=pl.Series(cp(n1, n2, fuzz.ratio)), n_tset=pl.Series(cp(n1, n2, fuzz.token_set_ratio)),
        n_jw=pl.Series(cp(d['name_sq'].to_list(), d['name_sq_r'].to_list(), JaroWinkler.normalized_similarity)),
        a_tset=pl.Series(a_tset), a_miss=pl.Series(a_miss.astype(np.int8)), zip_eq=pl.Series(zip_eq), zip_miss=pl.Series(zip_miss.astype(np.int8)))


def union_for(split, i1):
    """Union rows of the given S1 (i1 sorted; parquet row-group pruning on the i1 range)."""
    i1 = np.asarray(i1)
    lf = pl.scan_parquet(union_path(split)).filter(pl.col('i1').is_between(int(i1.min()), int(i1.max())))
    return lf.collect().join(pl.DataFrame({'i1': i1.astype(np.uint32)}), on='i1', how='semi')


def label(d):
    return d.join(truth().with_columns(pl.lit(1, dtype=pl.Int8).alias('y')), on=['i1', 'ir'], how='left').with_columns(pl.col('y').fill_null(0))


def keep(d, kmax, floor):
    return d.filter((pl.col('rr_rank') <= kmax) & (pl.col('p_rr') >= floor))


def train_and_tune():
    if RR_PATH.exists() and CFG_PATH.exists():
        log('stage-2 model cached'); return lgb.Booster(model_file=str(RR_PATH)), json.load(open(CFG_PATH))
    s1f, rf, rs = side_tables('train')
    t = time.time()
    tr = label(fast_features(union_for('train', np.sort(subset_i1('rr_train').to_numpy())), s1f, rf, rs))
    tu = label(fast_features(union_for('train', np.sort(subset_i1('rr_tune').to_numpy())), s1f, rf, rs))
    log(f'rr_train {tr.height:,} pairs ({tr["y"].sum():,} pos), rr_tune {tu.height:,} pairs; features in {time.time() - t:.0f}s')
    params = dict(objective='binary', learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0, seed=SEED)
    dtr = lgb.Dataset(tr.select(RR_FEATS).to_numpy().astype(np.float32), tr['y'].to_numpy())
    dva = lgb.Dataset(tu.select(RR_FEATS).to_numpy().astype(np.float32), tu['y'].to_numpy())
    m = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    RR_PATH.parent.mkdir(exist_ok=True); m.save_model(str(RR_PATH), num_iteration=m.best_iteration)
    log(f'stage-2 LightGBM: best iteration {m.best_iteration}')
    tu = tu.with_columns(p_rr=pl.Series(m.predict(tu.select(RR_FEATS).to_numpy().astype(np.float32), num_iteration=m.best_iteration).astype(np.float32)))
    tu = tu.with_columns(rr_rank=pl.col('p_rr').rank('ordinal', descending=True).over('i1'))
    TU = subset_i1('rr_tune'); TT = truth().join(pl.DataFrame({'i1': TU}), on='i1', how='semi')
    base = candidate_report(tu, TT, TU, 'union (input to stage 2)')
    rows = [base]
    for k in [1, 3, 5, 10, 20, 50]:
        rows.append(candidate_report(tu.filter(pl.col('brank') <= k), TT, TU, f'sparse-only top-{k}'))
    for k in [5, 10, 40]:
        rows.append(candidate_report(tu.filter(pl.col('drank_s') <= k), TT, TU, f'dense-only top-{k}'))
    grid = []
    for kmax in [3, 4, 5, 6, 7, 8, 10, 12, 15]:
        for fl in [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
            r = candidate_report(keep(tu, kmax, fl), TT, TU, f'stage2 kmax={kmax} p>={fl}')
            r.update(kmax=kmax, floor=fl); grid.append(r)
    G = pl.DataFrame(grid)
    ok = G.filter(pl.col('oracle F0.5') >= base['oracle F0.5'] - TOL).sort(['cand / S1', 'oracle F0.5'], descending=[False, True])
    best = ok.row(0, named=True) if ok.height else G.sort('oracle F0.5', descending=True).row(0, named=True)
    front = G.sort('cand / S1').filter(pl.col('oracle F0.5') >= pl.col('oracle F0.5').cum_max())
    with pl.Config(tbl_rows=60, tbl_width_chars=200):
        log('\n' + str(pl.DataFrame(rows)))
        log('Pareto frontier:\n' + str(front.select('stage', 'cand / S1', 'pair recall', 'pair precision', 'oracle F0.5')))
    cfg = {'kmax': int(best['kmax']), 'floor': float(best['floor']), 'TOL': TOL, 'tune_oracle': best['oracle F0.5'],
           'tune_cand_per_s1': best['cand / S1'], 'union_oracle': base['oracle F0.5'], 'union_cand_per_s1': base['cand / S1'],
           'best_iteration': m.best_iteration, 'RR_FEATS': RR_FEATS}
    save_json(cfg, CFG_PATH)
    pl.DataFrame(rows).write_csv(OUT / 'blocking_report_rr_tune.csv'); G.write_csv(OUT / 'stage2_grid_rr_tune.csv')
    log('CHOSEN', cfg)
    return lgb.Booster(model_file=str(RR_PATH)), cfg


def materialise(split, model, cfg, chunk=150_000):
    p = stage2_path(split)
    if p.exists():
        log(p.name, 'cached'); return
    s1f, rf, rs = side_tables(split)
    n1 = s1f.height; outs = []; t = time.time()
    for a in range(0, n1, chunk):
        i1 = np.arange(a, min(a + chunk, n1), dtype=np.uint32)
        f = fast_features(union_for(split, i1), s1f, rf, rs)
        f = f.with_columns(p_rr=pl.Series(model.predict(f.select(RR_FEATS).to_numpy().astype(np.float32)).astype(np.float32)))
        f = f.with_columns(rr_rank=pl.col('p_rr').rank('ordinal', descending=True).over('i1').cast(pl.UInt16))
        outs.append(keep(f, cfg['kmax'], cfg['floor'])); del f; gc.collect()
    out = pl.concat(outs); out.write_parquet(p)
    log(f'{p.name}: {out.height:,} pairs for {n1:,} S1 ({out.height / n1:.2f} per S1) in {(time.time() - t) / 60:.1f} min')


def run():
    for split in ['train', 'test']:
        build_union(split)
    model, cfg = train_and_tune()
    for split in ['train', 'test']:
        materialise(split, model, cfg)
    # blocking summary on val (never used by any blocking component)
    V = subset_i1('val'); TV = truth().join(pl.DataFrame({'i1': V}), on='i1', how='semi')
    s2 = pl.read_parquet(stage2_path('train'), columns=['i1', 'ir']).join(pl.DataFrame({'i1': V}), on='i1', how='semi')
    un = union_for('train', np.sort(V.to_numpy())).select('i1', 'ir', 'brank', 'drank_s')
    rows = [candidate_report(un.filter(pl.col('brank') <= 50), TV, V, 'sparse top-50'),
            candidate_report(un.filter(pl.col('drank_s') <= 40), TV, V, 'dense top-40'),
            candidate_report(un, TV, V, 'union'), candidate_report(s2, TV, V, 'stage 2 (final candidates)')]
    meta = splits().filter(pl.col('split') == 'val')
    for c in meta['country'].unique().sort().to_list():
        Vc = meta.filter(pl.col('country') == c)['i1']
        rows.append(candidate_report(s2, TV, Vc, f'stage 2 [{c}]'))
    rep = pl.DataFrame(rows); rep.write_csv(OUT / 'blocking_report_val.csv')
    with pl.Config(tbl_rows=30, tbl_width_chars=200):
        log('VAL blocking summary:\n' + str(rep))
    te = pl.read_parquet(stage2_path('test'), columns=['i1']).group_by('i1').len()
    c1 = nt('test', '1', ['country']).with_row_index('i1')
    dist = c1.join(te, on='i1', how='left').with_columns(pl.col('len').fill_null(0)).group_by('country').agg(
        pl.len().alias('S1'), pl.col('len').mean().round(3).alias('cand/S1'), (pl.col('len') == 0).mean().alias('empty share'))
    log('TEST candidates per S1:\n' + str(dist.sort('country')))


if __name__ == '__main__':
    run()
