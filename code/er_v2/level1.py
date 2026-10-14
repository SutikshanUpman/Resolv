"""Step 5: level-1 LightGBM on the FULL candidate graph + competition features.

A features-only LightGBM (no transformer) is cross-fitted (3-fold GroupKFold by S1) over the candidates of ALL train S1,
so every train pair gets an out-of-fold probability p1; a model fitted on all train rows scores test. Rows of S1 in
rr_train / bienc are predicted but never used for fitting (their stage-2 / dense scores are in-sample).

From p1 we derive competition features on the whole graph, exactly as they exist on test:
  S1 side    : rank of the pair among its S1's candidates, gap to the best / runner-up, number of candidates with p1>0.5
  record side: best p1 of any OTHER S1 claiming the record, this S1's rank among the claimants, claimants with p1>0.5
Each S2/S3 record belongs to at most one S1, so a record strongly claimed elsewhere is evidence against this pair.
"""
import gc, time, json
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from .config import OUT, SEED, FEAT_VERSION, L1_TAG, log, save_json
from .data import splits
from .features import feat_path, FEATS

L1_DIR = OUT / 'models' / f'level1_{FEAT_VERSION}{L1_TAG}'
L1_PARAMS = dict(objective='binary', learning_rate=0.2, num_leaves=127, max_bin=63, force_col_wise=True, min_data_in_leaf=200, feature_fraction=0.7,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=24, seed=SEED)
MAX_FIT_ROWS = 1_000_000
N_ROUNDS, ES_ROUNDS, ES_ROWS = 400, 20, 200_000
if L1_TAG:   # Test 2: all fit rows, finer histograms, smaller steps (p1 carries half of the stacker's gain)
    L1_PARAMS.update(learning_rate=0.1, max_bin=255, num_threads=28)
    MAX_FIT_ROWS, N_ROUNDS, ES_ROUNDS, ES_ROWS = 20_000_000, 1500, 50, 500_000
COMP_FEATS = ['p1', 'p1_rank_s', 'p1_gap_s', 'p1_n05_s', 'p1_sum_s', 'p1_max_other_r', 'p1_rank_r', 'p1_n05_r', 'n_claim_r', 'p1_best_r']


def l1_path(split):
    return OUT / 'features' / f'{split}_l1_{FEAT_VERSION}{L1_TAG}.parquet'


def comp_features(d: pl.DataFrame) -> pl.DataFrame:
    """d: (i1, ir, p1) over the whole candidate graph of a split."""
    d = d.with_columns(
        p1_rank_s=pl.col('p1').rank('ordinal', descending=True).over('i1').cast(pl.UInt16),
        _s_max=pl.col('p1').max().over('i1'),
        _s_2=pl.when(pl.len().over('i1') > 1).then(pl.col('p1').top_k(2).min().over('i1')).otherwise(0.0),
        p1_n05_s=(pl.col('p1') > 0.5).sum().over('i1').cast(pl.UInt16),
        p1_sum_s=pl.col('p1').sum().over('i1'),
        _r_max=pl.col('p1').max().over('ir'),
        _r_2=pl.when(pl.len().over('ir') > 1).then(pl.col('p1').top_k(2).min().over('ir')).otherwise(0.0),
        p1_rank_r=pl.col('p1').rank('ordinal', descending=True).over('ir').cast(pl.UInt16),
        p1_n05_r=(pl.col('p1') > 0.5).sum().over('ir').cast(pl.UInt16),
        n_claim_r=pl.len().over('ir').cast(pl.UInt16))
    return d.with_columns(
        p1_gap_s=pl.when(pl.col('p1') >= pl.col('_s_max')).then(pl.col('p1') - pl.col('_s_2')).otherwise(pl.col('p1') - pl.col('_s_max')),
        p1_max_other_r=pl.when(pl.col('p1') >= pl.col('_r_max')).then(pl.col('_r_2')).otherwise(pl.col('_r_max')),
        p1_best_r=(pl.col('p1') >= pl.col('_r_max')).cast(pl.Int8),
    ).select(['i1', 'ir'] + COMP_FEATS)


def run():
    if l1_path('train').exists() and l1_path('test').exists():
        log('level-1 cached'); return
    L1_DIR.mkdir(parents=True, exist_ok=True)
    t = time.time()
    tr = pl.read_parquet(feat_path('train'), columns=['i1', 'ir', 'y'] + FEATS)
    sp = splits().select('i1', 'split')
    tr = tr.join(sp, on='i1', how='left')
    X = tr.select(FEATS).to_numpy().astype(np.float32); y = tr['y'].to_numpy(); g = tr['i1'].to_numpy()
    fit_ok = ~tr['split'].is_in(['rr_train', 'bienc']).to_numpy()
    oof = np.zeros(len(y), np.float32); iters = []
    rng = np.random.default_rng(SEED)
    for k, (a, b) in enumerate(GroupKFold(3).split(X, y, g)):
        mf = L1_DIR / f'fold{k}.txt'
        a = a[fit_ok[a]]
        if len(a) > MAX_FIT_ROWS:
            a = np.sort(rng.choice(a, MAX_FIT_ROWS, replace=False))
        if mf.exists():
            m = lgb.Booster(model_file=str(mf)); iters.append(m.current_iteration())
        else:
            # early stopping on a small slice of the held-out fold (fit rows only)
            bv = b[fit_ok[b]]; bv = rng.choice(bv, min(len(bv), ES_ROWS), replace=False)
            m = lgb.train(L1_PARAMS, lgb.Dataset(X[a], y[a]), N_ROUNDS, valid_sets=[lgb.Dataset(X[bv], y[bv])],
                          callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(100)])
            m.save_model(str(mf), num_iteration=m.best_iteration); iters.append(m.best_iteration)
        oof[b] = m.predict(X[b], num_iteration=iters[-1])
        log(f'level-1 fold {k}: fit rows {len(a):,}, best iter {iters[-1]}, fold AUC {roc_auc_score(y[b], oof[b]):.5f}')
    fin = L1_DIR / 'final.txt'
    if fin.exists():
        m = lgb.Booster(model_file=str(fin))
    else:
        a = np.where(fit_ok)[0]
        if len(a) > MAX_FIT_ROWS:
            a = np.sort(rng.choice(a, MAX_FIT_ROWS, replace=False))
        m = lgb.train(L1_PARAMS, lgb.Dataset(X[a], y[a]), int(np.mean(iters) * 1.1)); m.save_model(str(fin))
    imp = sorted(zip(m.feature_importance('gain'), FEATS), reverse=True)
    log('level-1 top gain:', [(f, round(float(v) / sum(x for x, _ in imp), 3)) for v, f in imp[:20]])
    log(f'level-1 OOF AUC (all train pairs) {roc_auc_score(y, oof):.5f}')
    comp_features(tr.select('i1', 'ir').with_columns(p1=pl.Series(oof))).write_parquet(l1_path('train'))
    del tr, X; gc.collect()
    te = pl.read_parquet(feat_path('test'), columns=['i1', 'ir'] + FEATS)
    p1 = m.predict(te.select(FEATS).to_numpy().astype(np.float32)).astype(np.float32)
    comp_features(te.select('i1', 'ir').with_columns(p1=pl.Series(p1))).write_parquet(l1_path('test'))
    save_json({'iters': iters, 'oof_auc': float(roc_auc_score(y, oof))}, L1_DIR / 'info.json')
    log(f'level-1 done in {(time.time() - t) / 60:.1f} min')


if __name__ == '__main__':
    run()
