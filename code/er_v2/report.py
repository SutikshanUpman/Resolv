"""Full metrics report of the v2 run, computed from cached artifacts only (CPU; no retraining).

Writes OUT/metrics/metrics_v2.json and prints every table. Sections:
  data/ghost, blocking (sparse / dense / union / stage-2, per country, per group size, candidate-size distributions),
  model diagnostics (stage-2, level-1, CE, stacker importances + training logs),
  validation (macro F0.5 by country / group size / singletons, macro P/R, pair metrics, loss decomposition,
  threshold sensitivity, calibration), error analysis (FP/FN categories + examples, sibling evidence for misses),
  test diagnostics (score distributions per country, uncertain band, candidate / prediction sizes),
  ordering / id diagnostics (does file order or id number carry the match? train truth vs test predictions).
"""
import json, gc, time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from .config import OUT, FEAT_VERSION
from .data import splits, truth, nt
from .metrics import macro_f05, exclusive, candidate_report
from .blocking_sparse import stage1_dir
from .dense import dense_path
from .stage2 import stage2_path, RR_PATH
from .features import feat_path

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(60)
M = {}
MD = OUT / 'metrics'; MD.mkdir(exist_ok=True)
T_THR = 0.69


def show(title, df):
    print(f'\n### {title}\n{df}')
    return df


def f05_rows(pred, T, S1, meta):
    """per-S1 tp/np/nt/f frame for S1 set."""
    base = meta.select(pl.col('i1').cast(pl.UInt32), 'country', 'n')
    pred = pred.select('i1', 'ir').join(base.select('i1'), on='i1', how='semi')
    tr = T.join(base.select('i1'), on='i1', how='semi')
    tp = pred.join(tr, on=['i1', 'ir']).group_by('i1').len('tp')
    d = (base.join(pred.group_by('i1').len('np'), on='i1', how='left').join(tp, on='i1', how='left').fill_null(0)
             .rename({'n': 'nt'}))
    P = pl.col('tp') / pl.col('np'); R = pl.col('tp') / pl.col('nt')
    return d.with_columns(f=pl.when((pl.col('nt') == 0) & (pl.col('np') == 0)).then(1.0)
                           .when((pl.col('nt') == 0) | (pl.col('np') == 0) | (pl.col('tp') == 0)).then(0.0)
                           .otherwise(1.25 * P * R / (0.25 * P + R)),
                           prec=pl.when(pl.col('np') > 0).then(P).otherwise(None), rec=pl.when(pl.col('nt') > 0).then(R).otherwise(None))


def run():
    t0 = time.time()
    sp = splits(); TR = truth()
    V = sp.filter(pl.col('split') == 'val'); S_V = V['i1']
    T_V = TR.join(V.select('i1'), on='i1', how='semi')
    M['ghost'] = json.load(open(OUT / 'ghost_info.json'))

    # ------------------------------------------------ blocking: every stage on ALL train S1, per country
    c1 = nt('train', '1', ['country']).with_row_index('i1')
    tr_c = TR.join(c1, on='i1')
    rows = []
    def recall_by_country(cand, stage):
        hit = cand.join(TR, on=['i1', 'ir']).join(c1, on='i1').group_by('country').len('hit')
        n_c = cand.group_by('i1').len('k').join(c1, on='i1').group_by('country').agg(pl.col('k').sum().alias('cand'))
        tot = tr_c.group_by('country').len('true')
        s1n = c1.group_by('country').len('S1')
        r = tot.join(hit, on='country').join(n_c, on='country').join(s1n, on='country').with_columns(
            stage=pl.lit(stage), recall=(pl.col('hit') / pl.col('true')).round(5), cand_per_s1=(pl.col('cand') / pl.col('S1')).round(2))
        rows.append(r.select('stage', 'country', 'S1', 'true', 'hit', 'recall', 'cand_per_s1'))
    sp1 = pl.scan_parquet(stage1_dir('train') / 'part_*.parquet').select('i1', 'ir', 'brank').collect()
    for k in [10, 50]:
        recall_by_country(sp1.filter(pl.col('brank') <= k).select('i1', 'ir'), f'sparse top-{k}')
    del sp1; gc.collect()
    de = pl.read_parquet(dense_path('train'), columns=['i1', 'ir', 'drank_s'])
    for k in [5, 10, 40]:
        recall_by_country(de.filter(pl.col('drank_s') <= k).select('i1', 'ir'), f'dense S1->R top-{k}')
    recall_by_country(de.select('i1', 'ir'), 'dense both directions')
    del de; gc.collect()
    s2 = pl.read_parquet(stage2_path('train'), columns=['i1', 'ir', 'p_rr', 'rr_rank'])
    recall_by_country(s2.select('i1', 'ir'), 'stage-2 final (all train S1)')
    B = show('Blocking recall on ALL ghost-train S1 (1.77M) per country', pl.concat(rows))
    M['blocking_all_train'] = B.to_dicts()
    M['blocking_val'] = pl.read_csv(OUT / 'blocking_report_val.csv').to_dicts()
    M['blocking_rr_tune'] = pl.read_csv(OUT / 'blocking_report_rr_tune.csv').to_dicts()
    G = pl.read_csv(OUT / 'stage2_grid_rr_tune.csv')
    front = G.sort('cand / S1').filter(pl.col('oracle F0.5') >= pl.col('oracle F0.5').cum_max())
    M['stage2_frontier'] = show('Stage-2 Pareto frontier (rr_tune)', front.select('stage', 'cand / S1', 'pair recall', 'oracle F0.5')).to_dicts()
    M['stage2_cfg'] = {k: v for k, v in json.load(open(OUT / 'models' / 'stage2_cfg_s2a.json')).items() if k != 'RR_FEATS'}

    # blocking misses on val: what kind of records are lost
    miss = T_V.join(s2.select('i1', 'ir'), on=['i1', 'ir'], how='anti')
    rf = pl.concat([nt('train', s, ['name', 'addr', 'name_indic', 'has_addr']) for s in '23']).with_row_index('ir')
    s1r = nt('train', '1', ['name', 'addr']).with_row_index('i1')
    mm = miss.join(rf, on='ir')
    M['blocking_miss_val'] = {'missed_pairs': miss.height, 'true_pairs': T_V.height,
                              'share_record_no_address': round(float((~mm['has_addr']).mean()), 4),
                              'share_record_indic_name': round(float(mm['name_indic'].mean()), 4)}
    print('\nval blocking misses:', M['blocking_miss_val'])
    show('Examples of val pairs lost by blocking', mm.sample(8, seed=3).join(s1r, on='i1', suffix='_s1').select('name_s1', 'addr_s1', 'name', 'addr'))

    # candidate-size distributions
    def size_dist(c, meta, tag):
        n = meta.select('i1', 'country').join(c.group_by('i1').len('n'), on='i1', how='left').with_columns(pl.col('n').fill_null(0))
        out = n.group_by('country').agg(pl.len().alias('S1'), pl.col('n').mean().round(3).alias('mean'), pl.col('n').median().alias('median'),
                                        pl.col('n').quantile(0.9).alias('p90'), pl.col('n').max().alias('max'),
                                        *[(pl.col('n') == k).mean().round(4).alias(f'={k}') for k in range(0, 9)]).sort('country')
        return show(f'Candidate-set size per S1 ({tag})', out)
    M['cand_size_val'] = size_dist(s2.join(V.select('i1'), on='i1', how='semi'), V, 'val').to_dicts()
    s2t = pl.read_parquet(stage2_path('test'), columns=['i1', 'ir', 'p_rr'])
    c1t = nt('test', '1', ['country']).with_row_index('i1')
    M['cand_size_test'] = size_dist(s2t, c1t, 'TEST = candidate_pairs.tsv').to_dicts()
    del s2; gc.collect()

    # ------------------------------------------------ model diagnostics
    rr = lgb.Booster(model_file=str(RR_PATH))
    imp = sorted(zip(rr.feature_importance('gain'), rr.feature_name()), reverse=True); tot = sum(v for v, _ in imp)
    M['stage2_importance'] = [(json.load(open(OUT / 'models' / 'stage2_cfg_s2a.json'))['RR_FEATS'][int(f.split('_')[-1])] if f.startswith('Column_') else f,
                               round(v / tot, 4)) for v, f in imp[:15]]
    print('\nstage-2 re-ranker top gain:', M['stage2_importance'])
    M['level1'] = json.load(open(OUT / 'models' / f'level1_{FEAT_VERSION}' / 'info.json'))
    for tag in ['e5b', 'e5s']:
        p = OUT / f'ce_{tag}' / 'train_log.json'
        if p.exists():
            L = json.load(open(p)); M[f'ce_{tag}_train'] = {'pairs': L['pairs'], 'minutes': round(L['minutes'], 1), 'base': L['base'],
                                                            'loss_first_last': [L['hist'][0], L['hist'][-1]], 'loss_curve': L['hist'][::10]}
            print(f'CE {tag}:', {k: v for k, v in M[f"ce_{tag}_train"].items() if k != "loss_curve"})
    hist = sorted((OUT / 'submission_history').iterdir())
    M['run_info_v2A'] = json.load(open(OUT / 'submission_history' / '20260926_215731' / 'run_info.json'))
    M['decision_study'] = json.load(open(OUT / f'decision_study_{FEAT_VERSION}_e5b.json'))

    # ------------------------------------------------ validation deep dive (final model = e5b stack)
    D = pl.read_parquet(OUT / f'eval_p_{FEAT_VERSION}_e5b.parquet')
    Dv = D.join(V.select('i1'), on='i1', how='semi')
    y, p = Dv['y'].to_numpy(), Dv['p'].to_numpy()
    M['val_pair_level'] = {'pairs': len(y), 'pos_rate': round(float(y.mean()), 4), 'auc': round(roc_auc_score(y, p), 6),
                           'logloss': round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 5), 'brier': round(brier_score_loss(y, p), 5)}
    print('\nval pair-level:', M['val_pair_level'])
    Dx = exclusive(D.select('i1', 'ir', 'p'))
    pred = Dx.filter(pl.col('p') >= T_THR)
    fr = f05_rows(pred, T_V, S_V, V)
    fr = fr.with_columns(grp=pl.col('nt').clip(0, 7).cast(pl.String))
    M['val_macro'] = {'F05': round(fr['f'].mean(), 5), 'macro_precision': round(fr['prec'].mean(), 5), 'macro_recall': round(fr['rec'].mean(), 5),
                      'pair_precision': round(fr['tp'].sum() / fr['np'].sum(), 5), 'pair_recall': round(fr['tp'].sum() / fr['nt'].sum(), 5),
                      'S1_perfect_share': round(float((fr['f'] == 1).mean()), 5)}
    print('val macro:', M['val_macro'])
    M['val_by_country'] = show('Val F0.5 by country', fr.group_by('country').agg(pl.len().alias('S1'), pl.col('f').mean().round(5).alias('F0.5'),
                               pl.col('prec').mean().round(5).alias('macroP'), pl.col('rec').mean().round(5).alias('macroR')).sort('country')).to_dicts()
    M['val_by_group_size'] = show('Val F0.5 by true group size (7 = 7+)', fr.group_by('grp').agg(pl.len().alias('S1'), pl.col('f').mean().round(5).alias('F0.5'),
                                  pl.col('prec').mean().round(5).alias('macroP'), pl.col('rec').mean().round(5).alias('macroR'),
                                  (1 - pl.col('f')).sum().round(1).alias('F-loss (sum 1-f)')).sort('grp')).to_dicts()

    # loss decomposition: where does 1 - F0.5 come from?
    cand_v = D.join(V.select('i1'), on='i1', how='semi').select('i1', 'ir')
    blk_miss = T_V.join(cand_v, on=['i1', 'ir'], how='anti').group_by('i1').len('blk_miss')
    fr2 = fr.join(blk_miss, on='i1', how='left').with_columns(pl.col('blk_miss').fill_null(0))
    fp_ = pl.col('np') - pl.col('tp'); fn_ = pl.col('nt') - pl.col('tp')
    cat = (pl.when(pl.col('f') == 1).then(pl.lit('perfect'))
             .when(pl.col('nt') == 0).then(pl.lit('singleton predicted non-empty'))
             .when(pl.col('np') == 0).then(pl.lit('non-singleton predicted empty'))
             .when((fp_ > 0) & (fn_ > 0)).then(pl.lit('both FP and FN'))
             .when(fp_ > 0).then(pl.lit('FP only (extra wrong record)'))
             .otherwise(pl.lit('FN only (missed record)')))
    fr2 = fr2.with_columns(cat=cat, fn_blocking_only=(fn_ > 0) & (fn_ == pl.col('blk_miss')))
    dec = fr2.group_by('cat').agg(pl.len().alias('S1'), (1 - pl.col('f')).sum().round(1).alias('sum(1-f)'),
                                  pl.col('fn_blocking_only').mean().round(3).alias('FN all due to blocking'))
    dec = dec.with_columns((pl.col('sum(1-f)') / fr2.height).round(5).alias('F0.5 points lost'))
    M['val_loss_decomposition'] = show('Val loss decomposition (F0.5 points lost = sum(1-f)/N)', dec.sort('F0.5 points lost', descending=True)).to_dicts()

    # threshold sensitivity (val + OOF) and calibration
    S_ST = sp.filter(pl.col('split') == 'stack_train')['i1']; T_ST = TR.join(pl.DataFrame({'i1': S_ST}), on='i1', how='semi')
    sens = []
    for t in [0.4, 0.5, 0.55, 0.6, 0.65, 0.69, 0.72, 0.75, 0.8, 0.85, 0.9]:
        sens.append({'t': t, 'val': round(macro_f05(Dx.filter(pl.col('p') >= t).select('i1', 'ir'), T_V, S_V), 5),
                     'stack_train_oof': round(macro_f05(Dx.filter(pl.col('p') >= t).select('i1', 'ir'), T_ST, S_ST), 5)})
    M['threshold_sensitivity'] = show('Threshold sensitivity (exclusive assignment)', pl.DataFrame(sens)).to_dicts()
    cal = Dv.with_columns(bin=(pl.col('p') * 10).floor().clip(0, 9)).group_by('bin').agg(pl.len().alias('pairs'), pl.col('p').mean().round(4).alias('mean p'),
                                                                                      pl.col('y').mean().round(4).alias('match rate')).sort('bin')
    M['val_calibration'] = show('Val calibration (stacker p deciles)', cal).to_dicts()

    # ------------------------------------------------ error analysis (val, final rule)
    F = pl.read_parquet(feat_path('train'), columns=['i1', 'ir', 'a_miss', 'num_first_eq', 'n_jacc', 'indic2', 'm_s1_sq', 'm_s1_addr', 'dcos'])
    fp = pred.join(V.select('i1'), on='i1', how='semi').join(TR, on=['i1', 'ir'], how='anti').select('i1', 'ir')
    fn = T_V.join(pred.select('i1', 'ir'), on=['i1', 'ir'], how='anti')
    fn_c = fn.join(cand_v, on=['i1', 'ir'], how='semi')
    def cats(df):
        d = df.join(F, on=['i1', 'ir'], how='left')
        return d.select(pl.len().alias('pairs'), (pl.col('a_miss') == 1).mean().round(3).alias('addr missing'),
                        (pl.col('num_first_eq') == 0).mean().round(3).alias('house no differs'), (pl.col('n_jacc') == 0).mean().round(3).alias('no name overlap'),
                        (pl.col('indic2') == 1).mean().round(3).alias('indic record'), (pl.col('m_s1_sq') > 1).mean().round(3).alias('S1 name shared'),
                        (pl.col('m_s1_addr') > 1).mean().round(3).alias('S1 addr shared')).to_dicts()[0]
    M['val_fp'] = cats(fp); M['val_fn_matcher'] = cats(fn_c)
    M['val_fn_matcher']['pairs_total_fn'] = fn.height; M['val_fn_matcher']['fn_blocking'] = fn.height - fn_c.height
    print('\nFP profile:', M['val_fp']); print('FN (in candidates) profile:', M['val_fn_matcher'])
    # sibling evidence: for a missed record, does an identical name_core exist among the S1's OTHER true records (i.e. could
    # record-record clustering recover it?) and among records of OTHER candidate S1s of the same record?
    nc = pl.concat([nt('train', s, ['name_core', 'has_addr']) for s in '23']).with_row_index('ir')
    fnx = fn_c.join(nc, on='ir')
    sib = T_V.join(nc, on='ir').rename({'ir': 'ir2', 'name_core': 'nc2', 'has_addr': 'ha2'})
    ev = fnx.join(sib, on='i1').filter(pl.col('ir2') != pl.col('ir')).group_by('i1', 'ir').agg(
        (pl.col('nc2') == pl.col('name_core').first()).any().alias('same_name_sibling'),
        ((pl.col('nc2') == pl.col('name_core').first()) & pl.col('ha2')).any().alias('same_name_sibling_with_addr'))
    fnx = fnx.join(ev, on=['i1', 'ir'], how='left').with_columns(pl.col('same_name_sibling', 'same_name_sibling_with_addr').fill_null(False))
    M['val_fn_sibling_evidence'] = {
        'fn_in_candidates': fnx.height,
        'record_has_no_address': round(float((~fnx['has_addr']).mean()), 4),
        'identical name_core among same S1 true siblings': round(float(fnx['same_name_sibling'].mean()), 4),
        '... and that sibling has an address': round(float(fnx['same_name_sibling_with_addr'].mean()), 4)}
    print('FN sibling evidence:', M['val_fn_sibling_evidence'])
    show('FP examples (val)', fp.sample(10, seed=5).join(s1r, on='i1').join(rf.select('ir', 'name', 'addr'), on='ir', suffix='_r')
         .join(Dx, on=['i1', 'ir']).select('name', 'addr', 'name_r', 'addr_r', pl.col('p').round(3)))
    show('FN examples (val, in candidates)', fn_c.sample(10, seed=5).join(s1r, on='i1').join(rf.select('ir', 'name', 'addr'), on='ir', suffix='_r')
         .join(D.select('i1', 'ir', 'p'), on=['i1', 'ir'], how='left').select('name', 'addr', 'name_r', 'addr_r', pl.col('p').round(3)))
    del F, D, Dv, Dx; gc.collect()

    # ------------------------------------------------ test diagnostics
    te = pl.read_parquet(OUT / f'test_p_{FEAT_VERSION}_e5b.parquet').join(c1t, on='i1')
    tx = exclusive(te.select('i1', 'ir', 'p', 'country'))
    M['test_p_distribution'] = show('TEST stacker p distribution per country (all candidate pairs)', te.group_by('country').agg(
        pl.len().alias('pairs'), (pl.col('p') >= 0.9).mean().round(4).alias('p>=0.9'), ((pl.col('p') >= 0.3) & (pl.col('p') < 0.9)).mean().round(4).alias('0.3<=p<0.9 (uncertain)'),
        (pl.col('p') < 0.1).mean().round(4).alias('p<0.1'), pl.col('p').mean().round(4).alias('mean p')).sort('country')).to_dicts()
    pv = pl.read_parquet(OUT / f'eval_p_{FEAT_VERSION}_e5b.parquet').join(V.select('i1', 'country'), on='i1')
    M['val_p_distribution'] = show('VAL stacker p distribution per country (same columns, for comparison)', pv.group_by('country').agg(
        pl.len().alias('pairs'), (pl.col('p') >= 0.9).mean().round(4).alias('p>=0.9'), ((pl.col('p') >= 0.3) & (pl.col('p') < 0.9)).mean().round(4).alias('0.3<=p<0.9 (uncertain)'),
        (pl.col('p') < 0.1).mean().round(4).alias('p<0.1'), pl.col('p').mean().round(4).alias('mean p')).sort('country')).to_dicts()
    conflict = te.filter(pl.col('p') >= 0.5).group_by('ir').len().filter(pl.col('len') > 1).height / max(te.filter(pl.col('p') >= 0.5)['ir'].n_unique(), 1)
    confv = pv.filter(pl.col('p') >= 0.5).group_by('ir').len().filter(pl.col('len') > 1).height / max(pv.filter(pl.col('p') >= 0.5)['ir'].n_unique(), 1)
    M['record_conflict_rate'] = {'test: records with >=2 S1 at p>=0.5': round(conflict, 5), 'val-eval set: same': round(confv, 5)}
    print('\nrecord conflict rate:', M['record_conflict_rate'])
    predt = tx.filter(pl.col('p') >= T_THR)
    ps = c1t.join(predt.group_by('i1').len('n'), on='i1', how='left').with_columns(pl.col('n').fill_null(0))
    M['test_pred_size'] = show('TEST predicted matches per S1 (distribution)', ps.group_by('country').agg(
        pl.col('n').mean().round(3).alias('mean'), *[(pl.col('n') == k).mean().round(4).alias(f'={k}') for k in range(0, 8)], (pl.col('n') >= 8).mean().round(4).alias('>=8')).sort('country')).to_dicts()
    vs = V.select('i1', 'country', 'n')
    M['val_true_size'] = show('VAL true matches per S1 (for comparison)', vs.group_by('country').agg(
        pl.col('n').mean().round(3).alias('mean'), *[(pl.col('n') == k).mean().round(4).alias(f'={k}') for k in range(0, 8)], (pl.col('n') >= 8).mean().round(4).alias('>=8')).sort('country')).to_dicts()

    # ------------------------------------------------ ordering / id diagnostics (is there positional structure?)
    from scipy.stats import spearmanr
    ids1 = nt('train', '1', ['entity_id']).with_row_index('i1'); idsr = pl.concat([nt('train', s, ['entity_id']) for s in '23']).with_row_index('ir')
    t_s = TR.sample(200_000, seed=1).join(ids1, on='i1').join(idsr, on='ir', suffix='_r')
    num = lambda c: t_s[c].str.slice(3).cast(pl.Int64).to_numpy()
    ids1t = nt('test', '1', ['entity_id']).with_row_index('i1'); idsrt = pl.concat([nt('test', s, ['entity_id']) for s in '23']).with_row_index('ir')
    hp = predt.filter(pl.col('p') >= 0.95).sample(200_000, seed=1).join(ids1t, on='i1').join(idsrt, on='ir', suffix='_r')
    numt = lambda c: hp[c].str.slice(3).cast(pl.Int64).to_numpy()
    M['order_id_diagnostics'] = {
        'train truth: spearman(S1 row, record row)': round(spearmanr(t_s['i1'].to_numpy(), t_s['ir'].to_numpy()).correlation, 5),
        'train truth: spearman(S1 id number, record id number)': round(spearmanr(num('entity_id'), num('entity_id_r')).correlation, 5),
        'test confident preds: spearman(S1 row, record row)': round(spearmanr(hp['i1'].to_numpy(), hp['ir'].to_numpy()).correlation, 5),
        'test confident preds: spearman(S1 id number, record id number)': round(spearmanr(numt('entity_id'), numt('entity_id_r')).correlation, 5)}
    print('order/id diagnostics:', M['order_id_diagnostics'])

    json.dump(M, open(MD / 'metrics_v2.json', 'w'), indent=1, default=str)
    print(f'\nwrote {MD / "metrics_v2.json"} in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    run()
