"""Step 7: cross-encoder training/scoring, LightGBM stacker, decision rule, submission files.

Evaluation set E = stack_train U val U their CLOSURE (every other S1 that claims one of their candidate records), so
exclusive assignment on validation sees exactly the competitors it will see on test. The stacker is fitted with 5-fold
GroupKFold OOF on stack_train only; val is used for reporting only; the decision rule and threshold are chosen on
stack_train OOF.
"""
import gc, time, json, shutil, sys
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from .config import OUT, FINAL_OUT, SEED, FEAT_VERSION, STAGE2_VERSION, L1_TAG, STACK_TAG, log, save_json
from .data import splits, truth, nt, id_maps
from .features import feat_path, FEATS
from .level1 import l1_path, COMP_FEATS
from .stage2 import stage2_path
from .cross_encoder import train_ce, score_ce
from .metrics import macro_f05, exclusive, rule_threshold, rule_expected_f, tune_threshold, candidate_report

CE_TAGS = [t for t in sys.argv[1:] if not t.startswith('-')] or ['e5b']
CE_FEATS_PER = ['ce', 'ce_rank_s', 'ce_gap_s', 'ce_n0_s', 'ce_max_s']
if 'r' in STACK_TAG:   # Test 2: the CE's view from the record's side (the final assignment is made per record)
    CE_FEATS_PER += ['ce_rank_r', 'ce_gap_r', 'ce_max_other_r', 'ce_n0_r']
RUN = f'{FEAT_VERSION}{L1_TAG}{STACK_TAG}'
STACK_PARAMS = dict(objective='binary', learning_rate=0.04, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=0, seed=SEED)
MAX_CE_TRAIN_PAIRS = 2_600_000


def ce_side_feats(d: pl.DataFrame, tag):
    c = f'ce_{tag}'
    d = d.with_columns(_m=pl.col(c).max().over('i1'),
                       _2=pl.when(pl.len().over('i1') > 1).then(pl.col(c).top_k(2).min().over('i1')).otherwise(-20.0))
    d = d.with_columns(**{
        f'ce_rank_s_{tag}': pl.col(c).rank('ordinal', descending=True).over('i1').cast(pl.UInt16),
        f'ce_gap_s_{tag}': pl.when(pl.col(c) >= pl.col('_m')).then(pl.col(c) - pl.col('_2')).otherwise(pl.col(c) - pl.col('_m')),
        f'ce_n0_s_{tag}': (pl.col(c) > 0).sum().over('i1').cast(pl.UInt16),
        f'ce_max_s_{tag}': pl.col('_m')}).drop('_m', '_2')
    if 'r' in STACK_TAG:
        d = d.with_columns(_rm=pl.col(c).max().over('ir'),
                           _r2=pl.when(pl.len().over('ir') > 1).then(pl.col(c).top_k(2).min().over('ir')).otherwise(-20.0))
        d = d.with_columns(**{
            f'ce_rank_r_{tag}': pl.col(c).rank('ordinal', descending=True).over('ir').cast(pl.UInt16),
            f'ce_gap_r_{tag}': pl.when(pl.col(c) >= pl.col('_rm')).then(pl.col(c) - pl.col('_r2')).otherwise(pl.col(c) - pl.col('_rm')),
            f'ce_max_other_r_{tag}': pl.when(pl.col(c) >= pl.col('_rm')).then(pl.col('_r2')).otherwise(pl.col('_rm')),
            f'ce_n0_r_{tag}': (pl.col(c) > 0).sum().over('ir').cast(pl.UInt16)}).drop('_rm', '_r2')
    return d


def stack_feats():
    return FEATS + COMP_FEATS + [f'{f}_{t}' for t in CE_TAGS for f in CE_FEATS_PER]


def eval_pairs():
    """stack_train U val U closure (all S1 claiming any of their candidate records): (i1, ir) of the train split."""
    sp = splits()
    s2 = pl.read_parquet(stage2_path('train'), columns=['i1', 'ir'])
    ev = sp.filter(pl.col('split').is_in(['stack_train', 'val'])).select('i1')
    recs = s2.join(ev, on='i1', how='semi').select('ir').unique()
    E = pl.concat([ev, s2.join(recs, on='ir', how='semi').select('i1')]).unique()
    return s2.join(E, on='i1', how='semi'), ev.height, E.height


def ce_stage(train_only=False):
    """Train every CE in CE_TAGS on ce_train stage-2 candidates, score the evaluation set and all of test (cached)."""
    sp = splits()
    ev_pairs, n_ev, n_E = eval_pairs()
    log(f'evaluation set: {n_ev:,} S1 (stack_train+val) + closure -> {n_E:,} S1, {ev_pairs.height:,} pairs')
    ce_tr = pl.read_parquet(stage2_path('train'), columns=['i1', 'ir']).join(
        sp.filter(pl.col('split') == 'ce_train').select('i1'), on='i1', how='semi')
    ce_tr = ce_tr.join(truth().with_columns(pl.lit(1, dtype=pl.Int8).alias('y')), on=['i1', 'ir'], how='left').with_columns(pl.col('y').fill_null(0))
    if ce_tr.height > MAX_CE_TRAIN_PAIRS:
        keep = ce_tr.select('i1').unique().sample(fraction=MAX_CE_TRAIN_PAIRS / ce_tr.height, seed=SEED)
        ce_tr = ce_tr.join(keep, on='i1', how='semi')
    log(f'CE training pairs {ce_tr.height:,} (pos rate {ce_tr["y"].mean():.3f})')
    for tag in CE_TAGS:
        train_ce(tag, ce_tr, micro_bs=256 if tag.startswith('e5s') else 128)
    if train_only:
        return
    te_pairs = pl.read_parquet(stage2_path('test'), columns=['i1', 'ir'])
    CE_TR, CE_TE = {}, {}
    for tag in CE_TAGS:
        CE_TR[tag] = score_ce(tag, 'train', ev_pairs, 'eval')
        CE_TE[tag] = score_ce(tag, 'test', te_pairs, 'all')
    return ev_pairs, te_pairs, CE_TR, CE_TE


def run():
    t0 = time.time()
    sp = splits()
    ev_pairs, te_pairs, CE_TR, CE_TE = ce_stage()
    log(f'CE stage done ({(time.time() - t0) / 60:.1f} min)')

    # ---------------- assemble
    def assemble(split, pairs, ce):
        f = pl.read_parquet(feat_path(split)).join(pairs, on=['i1', 'ir'], how='semi')
        f = f.join(pl.read_parquet(l1_path(split)), on=['i1', 'ir'], how='left')
        for tag in CE_TAGS:
            f = f.join(ce[tag].rename({f'ce_{tag}': f'ce_{tag}'}), on=['i1', 'ir'], how='left')
            f = ce_side_feats(f, tag)
        return f
    SF = stack_feats()
    D = assemble('train', ev_pairs, CE_TR).join(sp.select('i1', 'split', 'country'), on='i1', how='left')
    ST = D.filter(pl.col('split') == 'stack_train')
    X = ST.select(SF).to_numpy().astype(np.float32); y = ST['y'].to_numpy(); g = ST['i1'].to_numpy()
    oof = np.zeros(len(y), np.float32); iters = []
    mdir = OUT / 'models' / f'stack_{RUN}_{"_".join(CE_TAGS)}'; mdir.mkdir(parents=True, exist_ok=True)
    for k, (a, b) in enumerate(GroupKFold(5).split(X, y, g)):
        mf = mdir / f'fold{k}.txt'
        if mf.exists():
            m = lgb.Booster(model_file=str(mf)); iters.append(m.current_iteration())
        else:
            m = lgb.train(STACK_PARAMS, lgb.Dataset(X[a], y[a]), 4000, valid_sets=[lgb.Dataset(X[b], y[b])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            m.save_model(str(mf), num_iteration=m.best_iteration); iters.append(m.best_iteration)
        oof[b] = m.predict(X[b], num_iteration=iters[-1])
    fin = mdir / 'final.txt'
    STACK = lgb.Booster(model_file=str(fin)) if fin.exists() else lgb.train(STACK_PARAMS, lgb.Dataset(X, y), int(np.mean(iters) * 1.1))
    STACK.save_model(str(fin))
    log(f'stacker iters {iters}, OOF AUC {roc_auc_score(y, oof):.5f}')
    imp = sorted(zip(STACK.feature_importance('gain'), SF), reverse=True); tot = sum(v for v, _ in imp)
    log('stacker top gain:', [(f, round(float(v) / tot, 3)) for v, f in imp[:25]])

    # p for every evaluation pair: OOF for stack_train rows, final model for val + closure rows
    p_all = STACK.predict(D.select(SF).to_numpy().astype(np.float32)).astype(np.float32)
    D = D.with_columns(p=pl.Series(p_all))
    D = D.join(ST.select('i1', 'ir').with_columns(p_oof=pl.Series(oof)), on=['i1', 'ir'], how='left').with_columns(
        p=pl.coalesce('p_oof', 'p')).drop('p_oof')
    TR = truth()
    S_ST = sp.filter(pl.col('split') == 'stack_train')['i1']; S_V = sp.filter(pl.col('split') == 'val')['i1']
    T_ST = TR.join(pl.DataFrame({'i1': S_ST}), on='i1', how='semi'); T_V = TR.join(pl.DataFrame({'i1': S_V}), on='i1', how='semi')

    # ---------------- decision rule chosen on stack_train OOF (exclusivity over the whole evaluation set)
    Dx = exclusive(D.select('i1', 'ir', 'p'))
    (t_best, f_best), curve = tune_threshold(Dx, T_ST, S_ST, excl=False, grid=np.round(np.arange(0.30, 0.96, 0.01), 3))
    f_ef = macro_f05(rule_expected_f(D.select('i1', 'ir', 'p')), T_ST, S_ST)
    rule = 'threshold+exclusive' if f_best >= f_ef else 'expected-F+exclusive'
    log(f'stack_train OOF: threshold+exclusive best t={t_best} F={f_best:.5f} | expected-F+exclusive F={f_ef:.5f} -> {rule}')
    decide = (lambda d: rule_threshold(d, t_best)) if rule.startswith('threshold') else (lambda d: rule_expected_f(d))

    by = sp.filter(pl.col('split') == 'val').select('i1', pl.col('country').alias('group'))
    pred_v = decide(D.select('i1', 'ir', 'p'))
    res_v = macro_f05(pred_v, T_V, S_V, by=by)
    orc = candidate_report(D.select('i1', 'ir'), T_V, S_V, 'val stage-2 candidates')
    log(f'VAL (ghost, test-like) macro F0.5: {res_v} | oracle {orc["oracle F0.5"]} | cand/S1 {orc["cand / S1"]}')
    ce_only = {}
    for tag in CE_TAGS:
        dd = D.select('i1', 'ir', pl.col(f'ce_{tag}').alias('p')).with_columns(p=1 / (1 + (-pl.col('p')).exp()))
        (tt, _), _ = tune_threshold(exclusive(dd), T_ST, S_ST, excl=False)
        ce_only[tag] = macro_f05(rule_threshold(dd, tt), T_V, S_V)
    dd = D.select('i1', 'ir', pl.col('p1').alias('p'))
    (tt, _), _ = tune_threshold(exclusive(dd), T_ST, S_ST, excl=False)
    l1_only = macro_f05(rule_threshold(dd, tt), T_V, S_V)
    log(f'VAL ablations: CE-only {ce_only} | level-1 (features only) {l1_only:.5f}')

    # ---------------- test
    F_TE = assemble('test', te_pairs, CE_TE)
    F_TE = F_TE.with_columns(p=pl.Series(STACK.predict(F_TE.select(SF).to_numpy().astype(np.float32)).astype(np.float32)))
    pred_te = decide(F_TE.select('i1', 'ir', 'p'))
    info = {'val': res_v, 'val_oracle': orc, 'rule': rule, 'threshold': t_best, 'stack_train_oof_f05': f_best, 'expected_f_oof': f_ef,
            'ce_only_val': ce_only, 'level1_only_val': l1_only, 'ce_tags': CE_TAGS, 'stack_iters': iters,
            'test_pairs': F_TE.height, 'test_matches': pred_te.height}
    F_TE.select('i1', 'ir', 'p').write_parquet(OUT / f'test_p_{RUN}_{"_".join(CE_TAGS)}.parquet')
    D.select('i1', 'ir', 'p', 'y', 'split').write_parquet(OUT / f'eval_p_{RUN}_{"_".join(CE_TAGS)}.parquet')
    # inputs for the collective second pass (Test 2)
    keep = ['i1', 'ir', 'p', 'p1'] + [f'ce_{t}' for t in CE_TAGS]
    F_TE.select(keep).write_parquet(OUT / f'test_pfull_{RUN}_{"_".join(CE_TAGS)}.parquet')
    D.select(keep + ['y', 'split']).write_parquet(OUT / f'eval_pfull_{RUN}_{"_".join(CE_TAGS)}.parquet')
    write_submission(pred_te, te_pairs, info)


def id_lists(pairs_i, col, m1, mr):
    g = (pairs_i.join(mr, on='ir').sort(['i1', 'rid']).group_by('i1', maintain_order=True)
                .agg(pl.col('rid').unique(maintain_order=True).str.join(',').alias(col)))
    return m1.join(g, on='i1', how='left').with_columns(pl.col(col).fill_null('')).sort('i1').select(pl.col('s1').alias('source1_entity_id'), col)


def write_tsv(df, path):
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\t'.join(df.columns) + '\n')
        for a, b in df.iter_rows():
            fh.write(f'{a}\t{b}\n')


def write_submission(pred_te, cand_te, info):
    m1, mr = id_maps('test')
    MATCH = id_lists(pred_te, 'matched_entity_ids', m1, mr)
    CAND = id_lists(cand_te, 'candidate_entity_ids', m1, mr)
    write_tsv(MATCH, FINAL_OUT / 'matching_results.tsv'); write_tsv(CAND, FINAL_OUT / 'candidate_pairs.tsv')
    hist = OUT / 'submission_history' / time.strftime('%Y%m%d_%H%M%S'); hist.mkdir(parents=True, exist_ok=True)
    for f in ['matching_results.tsv', 'candidate_pairs.tsv']:
        shutil.copy(FINAL_OUT / f, hist / f)
    c1 = nt('test', '1', ['country']).with_row_index('i1')
    per = (c1.join(pred_te.group_by('i1').len('n'), on='i1', how='left').join(cand_te.group_by('i1').len('c'), on='i1', how='left')
             .fill_null(0).group_by('country').agg(pl.len().alias('S1'), pl.col('n').mean().round(3).alias('pred/S1'),
                                                   (pl.col('n') == 0).mean().round(4).alias('empty share'), pl.col('c').mean().round(3).alias('cand/S1'))
             .sort('country'))
    info['test_by_country'] = per.to_dicts()
    save_json(info, hist / 'run_info.json')
    log('TEST by country:\n' + str(per))
    log(f'submission written: {FINAL_OUT} (history {hist})')


if __name__ == '__main__':
    if '--ce-train-only' in sys.argv:
        ce_stage(train_only=True)
    elif '--ce-only' in sys.argv:
        ce_stage()
    else:
        run()
