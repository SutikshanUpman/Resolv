"""Recall diagnostics on val (v2 caches, nothing is refitted): where every true pair is lost, and what the lost pairs look like.

Status of each true (S1, record) pair of the 100k val S1:
  not_in_union : never retrieved by sparse or dense
  cut          : retrieved, dropped by the stage-2 cut (rank > kmax or p_rr < floor)
  matcher      : reached the matcher, not predicted (stacker p < t, or the record went to another S1)
  hit          : predicted
For the cut losses: stage-2 rank / p_rr, S1 at the 8-cap, record without address, sibling evidence
(the record's name_core equals that of a confident (p_rr >= 0.9) record of the same S1), acronym match.
For the matcher losses: the true S1's rank among the record's claimants, and own-vs-rival sibling evidence from stacker p.
Also simulates a few cut rules on val (information only; the rule is chosen on rr_tune in stage 2).
"""
import sys, json
import numpy as np
import polars as pl
import lightgbm as lgb
from .config import OUT, FEAT_VERSION, log, save_json
from .data import subset_i1, truth
from .metrics import exclusive, candidate_report
from .stage2 import union_for, side_tables, fast_features, RR_FEATS, RR_PATH, stage2_path

TAG = 'e5b'
T = 0.69
SMALL = ['de', 'la', 'le', 'les', 'du', 'des', 'd', 'l', 'et', 'and', 'the', 'of', 'en', 'sur', 'a', 'au', 'aux']


def initials(col):
    """First letters of the alphabetic, non-stopword tokens of name_core ('fun boutique primaire' -> 'fbp')."""
    return (pl.col(col).fill_null('').str.split(' ')
            .list.eval(pl.element().filter((pl.element().str.len_chars() > 0) & ~pl.element().is_in(SMALL)
                                           & pl.element().str.contains(r'^[a-z]')).str.slice(0, 1))
            .list.join(''))


def acr_expr():
    """acronym match in either direction (the short side is 2-6 characters)."""
    a = (pl.col('name_sq_r').str.len_chars().is_between(2, 6)) & (pl.col('name_sq_r') == pl.col('ini'))
    b = (pl.col('name_sq').str.len_chars().is_between(2, 6)) & (pl.col('name_sq') == pl.col('ini_r'))
    return (a | b) & (pl.col('ini').str.len_chars() >= 2) & (pl.col('ini_r').str.len_chars() >= 2)


def sib_eq(d, p, thr, key='name_core_r'):
    """Per pair: number of OTHER confident records (p >= thr) of the same S1 whose key equals this record's key."""
    core = d.filter(pl.col(p) >= thr).select('i1', pl.col('ir').alias('ir_c'), pl.col(key).alias('k'))
    cnt = core.group_by('i1', 'k').len('n_same')
    out = d.join(cnt, left_on=['i1', key], right_on=['i1', 'k'], how='left').with_columns(pl.col('n_same').fill_null(0))
    # remove the pair itself when it is part of the core
    return out.with_columns((pl.col('n_same') - (pl.col(p) >= thr).cast(pl.UInt32)).clip(0).alias(f'sib_{p}'))


def run():
    V = pl.DataFrame({'i1': np.sort(subset_i1('val').to_numpy())}).with_columns(pl.col('i1').cast(pl.UInt32))
    TV = truth().join(V, on='i1', how='semi').with_columns(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32))
    s1f, rf, rs = side_tables('train')
    log(f'val S1 {V.height:,}, true pairs {TV.height:,}')

    # stage-2 scores for every val union pair (the cached stage-2 file only holds the kept pairs)
    f = fast_features(union_for('train', V['i1'].to_numpy()), s1f, rf, rs)
    m = lgb.Booster(model_file=str(RR_PATH))
    f = f.with_columns(p_rr=pl.Series(m.predict(f.select(RR_FEATS).to_numpy().astype(np.float32)).astype(np.float32)))
    f = f.with_columns(rr_rank=pl.col('p_rr').rank('ordinal', descending=True).over('i1').cast(pl.UInt16))
    f = f.select('i1', 'ir', 'p_rr', 'rr_rank', 'a_miss', 'drank_r', 'dcos', 'd_gap_r')
    f = (f.join(s1f.select(pl.col('i1').cast(pl.UInt32), 'name_core', 'name_sq'), on='i1', how='left')
          .join(rf.select(pl.col('ir').cast(pl.UInt32), pl.col('name_core').alias('name_core_r'), pl.col('name_sq').alias('name_sq_r'),
                          pl.col('name_sk').alias('name_sk_r'), pl.col('addr_norm').is_null().alias('r_noaddr')), on='ir', how='left')
          .with_columns(ini=initials('name_core'), ini_r=initials('name_core_r')))
    f = f.with_columns(acr=acr_expr())
    f = sib_eq(f, 'p_rr', 0.9)
    s2 = pl.read_parquet(stage2_path('train'), columns=['i1', 'ir']).join(V, on='i1', how='semi')
    ncap = s2.group_by('i1').len('n_cand')
    ev = pl.read_parquet(OUT / f'eval_p_{FEAT_VERSION}_{TAG}.parquet').select('i1', 'ir', 'p')
    pred = exclusive(ev).filter(pl.col('p') >= T).select('i1', 'ir')
    ntrue = TV.group_by('i1').len('n_true')

    st = (TV.join(f, on=['i1', 'ir'], how='left')
            .join(s2.with_columns(in_s2=pl.lit(True)), on=['i1', 'ir'], how='left')
            .join(pred.with_columns(hit=pl.lit(True)), on=['i1', 'ir'], how='left')
            .join(ev.rename({'p': 'p_stack'}), on=['i1', 'ir'], how='left')
            .join(ncap, on='i1', how='left').join(ntrue, on='i1', how='left')
            .with_columns(status=pl.when(pl.col('p_rr').is_null()).then(pl.lit('not_in_union'))
                          .when(pl.col('in_s2').is_null()).then(pl.lit('cut'))
                          .when(pl.col('hit').is_null()).then(pl.lit('matcher')).otherwise(pl.lit('hit'))))
    out = {'true_pairs': TV.height}
    out['status'] = {k: v for k, v in st.group_by('status').len().iter_rows()}
    log('status of true val pairs:', out['status'])

    cut = st.filter(pl.col('status') == 'cut')
    prof = lambda d: {
        'pairs': d.height,
        'record no address': round(d['r_noaddr'].mean(), 4),
        'S1 at the 8-cap': round((d['n_cand'].fill_null(0) >= 8).mean(), 4),
        'rr_rank 9-10': round(d['rr_rank'].is_between(9, 10).mean(), 4),
        'rr_rank 11-20': round(d['rr_rank'].is_between(11, 20).mean(), 4),
        'rr_rank > 20': round((d['rr_rank'] > 20).mean(), 4),
        'rr_rank <= 8 (cut by the p floor)': round((d['rr_rank'] <= 8).mean(), 4),
        'p_rr >= 0.3': round((d['p_rr'] >= 0.3).mean(), 4),
        'p_rr 0.05-0.3': round(d['p_rr'].is_between(0.05, 0.3).mean(), 4),
        'p_rr < 0.05': round((d['p_rr'] < 0.05).mean(), 4),
        'sibling name_core match (p_rr>=0.9 core)': round((d['sib_p_rr'] >= 1).mean(), 4),
        '... and record has no address': round(((d['sib_p_rr'] >= 1) & d['r_noaddr']).mean(), 4),
        'acronym match': round(d['acr'].mean(), 4),
        'record-side dense rank <= 3': round((d['drank_r'] <= 3).mean(), 4),
        'S1 group size >= 7': round((d['n_true'] >= 7).mean(), 4)}
    out['cut_profile'] = prof(cut)
    log('CUT losses:', json.dumps(out['cut_profile'], indent=1))

    # simulated cut rules on val (information only)
    TVc = TV; Vi = V['i1']
    rules = {
        'v2: rank<=8 & p>=0.01': (pl.col('rr_rank') <= 8) & (pl.col('p_rr') >= 0.01),
        'v2 | p>=0.3': ((pl.col('rr_rank') <= 8) & (pl.col('p_rr') >= 0.01)) | (pl.col('p_rr') >= 0.3),
        'v2 | p>=0.15': ((pl.col('rr_rank') <= 8) & (pl.col('p_rr') >= 0.01)) | (pl.col('p_rr') >= 0.15),
        'v2 | sibling&noaddr': ((pl.col('rr_rank') <= 8) & (pl.col('p_rr') >= 0.01)) | ((pl.col('sib_p_rr') >= 1) & pl.col('r_noaddr')),
        'v2 | p>=0.15 | sibling&noaddr': ((pl.col('rr_rank') <= 8) & (pl.col('p_rr') >= 0.01)) | (pl.col('p_rr') >= 0.15)
                                         | ((pl.col('sib_p_rr') >= 1) & pl.col('r_noaddr')),
        'v2 | p>=0.15 | sibling&noaddr | acronym': ((pl.col('rr_rank') <= 8) & (pl.col('p_rr') >= 0.01)) | (pl.col('p_rr') >= 0.15)
                                                   | ((pl.col('sib_p_rr') >= 1) & pl.col('r_noaddr')) | pl.col('acr'),
        'rank<=10 & p>=0.005': (pl.col('rr_rank') <= 10) & (pl.col('p_rr') >= 0.005),
    }
    sims = [candidate_report(f.select('i1', 'ir'), TVc, Vi, 'union')]
    for name, e in rules.items():
        sims.append(candidate_report(f.filter(e).select('i1', 'ir'), TVc, Vi, name))
    out['cut_rule_simulation_val'] = sims
    with pl.Config(tbl_rows=20, tbl_width_chars=220, fmt_str_lengths=60):
        log('cut rules on val (info only):\n' + str(pl.DataFrame(sims).drop('S1')))

    # matcher losses: rank of the true S1 among the record's claimants (stacker p) and sibling evidence own vs rival
    evx = ev.with_columns(r_rank=pl.col('p').rank('ordinal', descending=True).over('ir').cast(pl.UInt16),
                          r_n=pl.len().over('ir').cast(pl.UInt16))
    evx = (evx.join(rf.select(pl.col('ir').cast(pl.UInt32), pl.col('name_core').alias('name_core_r'),
                              pl.col('addr_norm').is_null().alias('r_noaddr')), on='ir', how='left'))
    core = exclusive(ev).filter(pl.col('p') >= 0.95).select('i1', 'ir')
    core = core.join(rf.select(pl.col('ir').cast(pl.UInt32), pl.col('name_core').alias('k')), on='ir').group_by('i1', 'k').len('n_same')
    evx = evx.join(core, left_on=['i1', 'name_core_r'], right_on=['i1', 'k'], how='left').with_columns(pl.col('n_same').fill_null(0))
    evx = evx.with_columns(self_core=((pl.col('p') >= 0.95) & (pl.col('p') >= pl.col('p').max().over('ir'))).cast(pl.UInt32))
    evx = evx.with_columns(sib=(pl.col('n_same') - pl.col('self_core')).clip(0))
    # rival = best OTHER claimant of the record by stacker p
    best = (evx.sort('p', descending=True).group_by('ir', maintain_order=True)
               .agg(pl.col('i1').first().alias('b1'), pl.col('sib').first().alias('s1b'), pl.col('sib').slice(1, 1).first().alias('s2b')))
    mt = (st.filter(pl.col('status') == 'matcher').select('i1', 'ir', 'p_stack', 'r_noaddr')
            .join(evx.select('i1', 'ir', 'r_rank', 'r_n', 'sib'), on=['i1', 'ir'], how='left').join(best, on='ir', how='left')
            .with_columns(sib_rival=pl.when(pl.col('b1') == pl.col('i1')).then(pl.col('s2b')).otherwise(pl.col('s1b')).fill_null(0)))
    out['matcher_profile'] = {
        'pairs': mt.height, 'record no address': round(mt['r_noaddr'].mean(), 4),
        'true S1 ranked first for the record': round((mt['r_rank'] == 1).mean(), 4),
        'record has >= 2 claimants': round((mt['r_n'] >= 2).mean(), 4),
        'p_stack 0.3-0.69': round(mt['p_stack'].is_between(0.3, 0.69).mean(), 4),
        'p_stack < 0.3': round((mt['p_stack'] < 0.3).mean(), 4),
        'own sibling name_core match (stacker core)': round((mt['sib'] >= 1).mean(), 4),
        'own sibling match AND rival none (resolvable)': round(((mt['sib'] >= 1) & (mt['sib_rival'] == 0)).mean(), 4),
        'rival sibling match AND own none (misleading)': round(((mt['sib'] == 0) & (mt['sib_rival'] >= 1)).mean(), 4)}
    log('MATCHER losses:', json.dumps(out['matcher_profile'], indent=1))
    (OUT / 'metrics').mkdir(exist_ok=True)
    save_json(out, OUT / 'metrics' / 'diag_recall.json')
    log('wrote', OUT / 'metrics' / 'diag_recall.json')


if __name__ == '__main__':
    run()
