"""Test 2, step 3: collective second pass over the stacker's predictions ("resolve the easy records, then use them").

After the stacker, ~91 % of S1 are already perfect. Their confident records (p >= CORE_T and the S1 is the record's
top claimant) are trusted evidence for the doubtful ones. For each candidate pair (s, r):
  own   : sib_n (confident records of s), sib_eq_core / sib_eq_sk / sib_eq_raw (how many share r's normalised name,
          phonetic skeleton, raw lower-cased name), sib_tset (best name similarity to them), sib_addr (best address
          similarity to them, -1 if r has no address)
  rival : the same features for r's best OTHER claimant, and own - rival differences
  graph : rank / gap / count of the pair among the record's claimants and the S1's candidates (on p)
A second LightGBM on p, CE, p1 and these features is fitted 5-fold (GroupKFold by S1) on stack_train; val is only
reported. p on stack_train rows is the stacker's OOF p, so no row is scored by a model that saw its label.
Decision (chosen on stack_train OOF): global exclusivity, then a rank-aware threshold (t1 for each S1's best remaining
pair, t2 for the others); kept only if it also improves val.
"""
import sys, time, json
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from rapidfuzz import fuzz
from .config import OUT, SEED, FEAT_VERSION, L1_TAG, STACK_TAG, log, save_json
from .data import splits, truth, nt
from .metrics import macro_f05, exclusive
from .stage2 import cp, stage2_path

import os
TAG = 'e5b'
RUN = f'{FEAT_VERSION}{L1_TAG}{STACK_TAG}'
QW = os.environ.get('ER_QW', '') == '1'      # Test 3: Qwen3-Reranker score on uncertain pairs (missing elsewhere)
# final pass: several Qwen score files, 'suffix:feature,...' -> qwen_scores_{split}{suffix}.parquet as feature
QW_FILES = [tuple(x.split(':')) for x in os.environ.get('ER_QW_FILES', ':qw').split(',')] if QW else []
OUT_RUN = RUN + os.environ.get('ER_QW_TAG', '_qw' if QW else '')
GLOBAL_ONLY = os.environ.get('ER_RULE', '') == 'global'   # final pass: global threshold only (rank-aware hurts singletons)
CORE_T = 0.95
PARAMS = dict(objective='binary', learning_rate=0.04, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=max(2, (os.cpu_count() or 8) - 2), seed=SEED)
BASE = ['p', 'p1', f'ce_{TAG}']
COLL = ['p_rank_s', 'p_gap_s', 'p_n05_s', 'p_rank_r', 'p_gap_r', 'p_n05_r', 'n_claim_r', 'r_noaddr',
        'sib_n', 'sib_eq_core', 'sib_eq_sk', 'sib_eq_raw', 'sib_tset', 'sib_addr',
        'riv_sib_n', 'riv_sib_eq_core', 'riv_sib_eq_sk', 'riv_sib_tset', 'riv_sib_addr', 'riv_p',
        'd_sib_eq_core', 'd_sib_tset', 'd_sib_addr']
FEATS = BASE + COLL + [f for _, f in QW_FILES]


def add_qw(d, split):
    if not QW:
        return d
    for suf, feat in QW_FILES:
        q = pl.read_parquet(OUT / f'qwen_scores_{split}{suf}.parquet').select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32),
                                                                            pl.col('qw').cast(pl.Float32).alias(feat))
        d = d.join(q, on=['i1', 'ir'], how='left')
    return d


def rec_table(split):
    r = pl.concat([nt(split, s, ['name', 'name_core', 'name_sk', 'addr_norm']) for s in '23']).with_row_index('ir')
    return r.select(pl.col('ir').cast(pl.UInt32), pl.col('name_core').fill_null(''), pl.col('name_sk').fill_null(''),
                    pl.col('name').str.to_lowercase().str.strip_chars().fill_null('').alias('name_raw'),
                    pl.col('addr_norm'), pl.col('addr_norm').is_null().cast(pl.Int8).alias('r_noaddr'))


def coll_features(d: pl.DataFrame, R: pl.DataFrame) -> pl.DataFrame:
    """d: (i1, ir, p, ...) over a complete candidate graph. Returns d + COLL."""
    d = d.with_columns(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32))
    d = d.with_columns(
        p_rank_s=pl.col('p').rank('ordinal', descending=True).over('i1').cast(pl.UInt16),
        _sm=pl.col('p').max().over('i1'), _s2=pl.when(pl.len().over('i1') > 1).then(pl.col('p').top_k(2).min().over('i1')).otherwise(0.0),
        p_n05_s=(pl.col('p') >= 0.5).sum().over('i1').cast(pl.UInt16),
        p_rank_r=pl.col('p').rank('ordinal', descending=True).over('ir').cast(pl.UInt16),
        _rm=pl.col('p').max().over('ir'), _r2=pl.when(pl.len().over('ir') > 1).then(pl.col('p').top_k(2).min().over('ir')).otherwise(0.0),
        p_n05_r=(pl.col('p') >= 0.5).sum().over('ir').cast(pl.UInt16), n_claim_r=pl.len().over('ir').cast(pl.UInt16))
    d = d.with_columns(
        p_gap_s=pl.when(pl.col('p') >= pl.col('_sm')).then(pl.col('p') - pl.col('_s2')).otherwise(pl.col('p') - pl.col('_sm')),
        p_gap_r=pl.when(pl.col('p') >= pl.col('_rm')).then(pl.col('p') - pl.col('_r2')).otherwise(pl.col('p') - pl.col('_rm')),
        riv_p=pl.when(pl.col('p') >= pl.col('_rm')).then(pl.col('_r2')).otherwise(pl.col('_rm')),
        _core=((pl.col('p') >= CORE_T) & (pl.col('p') >= pl.col('_rm'))).cast(pl.UInt32)).drop('_sm', '_s2', '_rm', '_r2')
    d = d.join(R, on='ir', how='left')
    d = d.with_columns(sib_n=(pl.col('_core').sum().over('i1') - pl.col('_core')).cast(pl.UInt16))
    core = d.filter(pl.col('_core') == 1)
    for key, out in [('name_core', 'sib_eq_core'), ('name_sk', 'sib_eq_sk'), ('name_raw', 'sib_eq_raw')]:
        c = core.group_by('i1', key).len('_n')
        d = d.join(c, on=['i1', key], how='left').with_columns(
            ((pl.col('_n').fill_null(0) - pl.col('_core')).clip(0) * (pl.col(key) != '').cast(pl.UInt32)).cast(pl.UInt16).alias(out)).drop('_n')
    # best name / address similarity to the S1's OTHER confident records
    cr = core.select('i1', pl.col('ir').alias('ir_c'), pl.col('name_core').alias('nc'), pl.col('addr_norm').alias('ac'))
    j = d.select('i1', 'ir', 'name_core', 'addr_norm').join(cr, on='i1').filter(pl.col('ir') != pl.col('ir_c'))
    j = j.with_columns(s_n=pl.Series(cp(j['name_core'].to_list(), j['nc'].to_list(), fuzz.token_set_ratio)))
    ja = j.filter(pl.col('addr_norm').is_not_null() & pl.col('ac').is_not_null())
    ja = ja.with_columns(s_a=pl.Series(cp(ja['addr_norm'].to_list(), ja['ac'].to_list(), fuzz.token_set_ratio)))
    d = (d.join(j.group_by('i1', 'ir').agg(pl.col('s_n').max().alias('sib_tset')), on=['i1', 'ir'], how='left')
          .join(ja.group_by('i1', 'ir').agg(pl.col('s_a').max().alias('sib_addr')), on=['i1', 'ir'], how='left')
          .with_columns(pl.col('sib_tset').fill_null(-1.0).cast(pl.Float32), pl.col('sib_addr').fill_null(-1.0).cast(pl.Float32)))
    # rival = the record's best OTHER claimant; its sibling features for the same record
    riv = (d.sort('p', descending=True).group_by('ir', maintain_order=True)
             .agg(pl.col('i1').first().alias('_b1'), pl.col('i1').slice(1, 1).first().alias('_b2')))
    d = d.join(riv, on='ir', how='left').with_columns(_riv=pl.when(pl.col('i1') == pl.col('_b1')).then(pl.col('_b2')).otherwise(pl.col('_b1')))
    rv = d.select(pl.col('i1').alias('_riv'), 'ir', *[pl.col(c).alias(f'riv_{c}') for c in ['sib_n', 'sib_eq_core', 'sib_eq_sk', 'sib_tset', 'sib_addr']])
    d = d.join(rv, on=['_riv', 'ir'], how='left').with_columns(
        pl.col('riv_sib_n').fill_null(0), pl.col('riv_sib_eq_core').fill_null(0), pl.col('riv_sib_eq_sk').fill_null(0),
        pl.col('riv_sib_tset').fill_null(-1.0), pl.col('riv_sib_addr').fill_null(-1.0))
    d = d.with_columns(d_sib_eq_core=pl.col('sib_eq_core').cast(pl.Int32) - pl.col('riv_sib_eq_core').cast(pl.Int32),
                       d_sib_tset=pl.col('sib_tset') - pl.col('riv_sib_tset'), d_sib_addr=pl.col('sib_addr') - pl.col('riv_sib_addr'))
    return d.drop('_core', '_b1', '_b2', '_riv', 'name_core', 'name_sk', 'name_raw', 'addr_norm')


def excl_ranked(d, p):
    """Global exclusivity (a record stays only with its best S1), then the rank of each remaining pair within its S1."""
    return exclusive(d.select('i1', 'ir', pl.col(p).alias('q')), 'q').with_columns(k=pl.col('q').rank('ordinal', descending=True).over('i1'))


def rank_rule(x, t1, t2):
    """x from excl_ranked: keep each S1's best remaining pair if q >= t1, every other pair if q >= t2."""
    return x.filter(((pl.col('k') == 1) & (pl.col('q') >= t1)) | (pl.col('q') >= t2)).select('i1', 'ir')


def run():
    t0 = time.time()
    sp = splits()
    D = pl.read_parquet(OUT / f'eval_pfull_{RUN}_{TAG}.parquet')
    D = add_qw(coll_features(D, rec_table('train')), 'train')
    log(f'eval set {D.height:,} pairs; collective features in {(time.time() - t0) / 60:.1f} min')
    ST = D.filter(pl.col('split') == 'stack_train')
    X = ST.select(FEATS).to_numpy().astype(np.float32); y = ST['y'].to_numpy(); g = ST['i1'].to_numpy()
    oof = np.zeros(len(y), np.float32); iters = []
    mdir = OUT / 'models' / f'collective_{OUT_RUN}_{TAG}'; mdir.mkdir(parents=True, exist_ok=True)
    for k, (a, b) in enumerate(GroupKFold(5).split(X, y, g)):
        m = lgb.train(PARAMS, lgb.Dataset(X[a], y[a]), 3000, valid_sets=[lgb.Dataset(X[b], y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        iters.append(m.best_iteration); oof[b] = m.predict(X[b], num_iteration=m.best_iteration)
    fin = lgb.train(PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1)); fin.save_model(str(mdir / 'final.txt'))
    imp = sorted(zip(fin.feature_importance('gain'), FEATS), reverse=True); tot = sum(v for v, _ in imp)
    log(f'collective iters {iters}; OOF AUC {roc_auc_score(y, oof):.5f} (stacker OOF AUC {roc_auc_score(y, ST["p"].to_numpy()):.5f}); top gain:',
        [(f, round(float(v) / tot, 3)) for v, f in imp[:15]])
    q_all = fin.predict(D.select(FEATS).to_numpy().astype(np.float32)).astype(np.float32)
    D = D.with_columns(q=pl.Series(q_all)).join(ST.select('i1', 'ir').with_columns(q_oof=pl.Series(oof)), on=['i1', 'ir'], how='left') \
         .with_columns(q=pl.coalesce('q_oof', 'q')).drop('q_oof')

    TR = truth()
    S_ST = sp.filter(pl.col('split') == 'stack_train')['i1']; S_V = sp.filter(pl.col('split') == 'val')['i1']
    T_ST = TR.join(pl.DataFrame({'i1': S_ST}), on='i1', how='semi'); T_V = TR.join(pl.DataFrame({'i1': S_V}), on='i1', how='semi')
    by = sp.filter(pl.col('split') == 'val').select('i1', pl.col('country').alias('group'))
    res = []; EX = {src: excl_ranked(D, src) for src in ['p', 'q']}
    for src in ['p', 'q']:
        for t2 in np.round(np.arange(0.55, 0.86, 0.03), 2):
            for t1 in np.round(np.arange(0.25, t2 + 0.001, 0.05), 2):
                res.append((src, float(t1), float(t2), macro_f05(rank_rule(EX[src], t1, t2), T_ST, S_ST)))
    R_ = pl.DataFrame(res, schema=['src', 't1', 't2', 'oof'], orient='row')
    out = {}
    for src in ['p', 'q']:
        glob_ = R_.filter((pl.col('src') == src) & (pl.col('t1') == pl.col('t2'))).sort('oof', descending=True).row(0, named=True)
        best = R_.filter(pl.col('src') == src).sort('oof', descending=True).row(0, named=True)
        for name, r in [('global', glob_), ('rank_aware', best)]:
            v = macro_f05(rank_rule(EX[src], r['t1'], r['t2']), T_V, S_V, by=by)
            out[f'{"stacker" if src == "p" else "collective"}_{name}'] = {'t1': r['t1'], 't2': r['t2'], 'oof': round(r['oof'], 5),
                                                                          'val': {k: round(x, 5) for k, x in v.items()}}
    for k, v in out.items():
        log(k, json.dumps(v))
    # choose on OOF; the collective model must also beat the stacker on val to be used
    keys = [k for k in out if k.endswith('global')] if GLOBAL_ONLY else list(out)
    best_key = max(keys, key=lambda k: out[k]['oof'])
    if best_key.startswith('collective') and out[best_key]['val']['all'] <= max(out['stacker_global']['val']['all'], out['stacker_rank_aware']['val']['all']):
        best_key = max([k for k in ['stacker_global', 'stacker_rank_aware'] if k in keys], key=lambda k: out[k]['oof'])
    log('CHOSEN:', best_key, out[best_key])
    D.select('i1', 'ir', pl.col('q').alias('p'), 'y', 'split').write_parquet(OUT / f'eval_p_{OUT_RUN}_coll_{TAG}.parquet')

    # test
    TE = pl.read_parquet(OUT / f'test_pfull_{RUN}_{TAG}.parquet')
    TE = add_qw(coll_features(TE, rec_table('test')), 'test')
    TE = TE.with_columns(q=pl.Series(fin.predict(TE.select(FEATS).to_numpy().astype(np.float32)).astype(np.float32)))
    TE.select('i1', 'ir', pl.col('q').alias('p')).write_parquet(OUT / f'test_p_{OUT_RUN}_coll_{TAG}.parquet')
    c = out[best_key]; src = 'q' if best_key.startswith('collective') else 'p'
    pred_te = rank_rule(excl_ranked(TE, src), c['t1'], c['t2'])
    from .stack import write_submission
    write_submission(pred_te, pl.read_parquet(stage2_path('test'), columns=['i1', 'ir']),
                     {'rule': best_key, **c, 'threshold': c['t2'], 'study': out, 'collective_iters': iters})
    save_json(out, OUT / 'metrics' / f'collective_{OUT_RUN}.json')
    log(f'collective done in {(time.time() - t0) / 60:.1f} min')
    return best_key, out


if __name__ == '__main__':
    run()
