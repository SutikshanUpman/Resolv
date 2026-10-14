"""Decision-rule study on the stacker's evaluation predictions (choice on stack_train OOF, report on val only).

Rules compared (all with global exclusive assignment: a record is kept only for its highest-scoring S1):
  A  global threshold
  B  per-country thresholds for countries seen in training; any unseen label (France) falls back to the global one
  C  threshold + margin over the record's runner-up claimant
  D  expected-F0.5 lists
Writes the chosen rule's test submission if --write is given.
"""
import sys, json
import numpy as np
import polars as pl
from .config import OUT, FEAT_VERSION, log, save_json
from .data import splits, truth, nt
from .metrics import macro_f05, exclusive, rule_expected_f

TAG = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'e5b'
GRID = np.round(np.arange(0.30, 0.96, 0.01), 3)


def with_runner_up(d):
    return d.with_columns(_r2=pl.when(pl.len().over('ir') > 1).then(pl.col('p').top_k(2).min().over('ir')).otherwise(0.0))


def best_t(dx, T, S, grid=GRID):
    res = [(float(t), macro_f05(dx.filter(pl.col('p') >= t).select('i1', 'ir'), T, S)) for t in grid]
    return max(res, key=lambda x: x[1])


def run(write=False):
    sp = splits()
    D = pl.read_parquet(OUT / f'eval_p_{FEAT_VERSION}_{TAG}.parquet').join(sp.select('i1', 'country'), on='i1', how='left')
    TR = truth()
    S_ST = sp.filter(pl.col('split') == 'stack_train')['i1']; S_V = sp.filter(pl.col('split') == 'val')['i1']
    T_ST = TR.join(pl.DataFrame({'i1': S_ST}), on='i1', how='semi'); T_V = TR.join(pl.DataFrame({'i1': S_V}), on='i1', how='semi')
    by = sp.filter(pl.col('split') == 'val').select('i1', pl.col('country').alias('group'))
    Dx = exclusive(with_runner_up(D.select('i1', 'ir', 'p', 'country')))
    out = {}
    # A: global threshold
    tA, fA = best_t(Dx, T_ST, S_ST)
    out['A_global'] = {'t': tA, 'oof': fA, 'val': macro_f05(Dx.filter(pl.col('p') >= tA).select('i1', 'ir'), T_V, S_V, by=by)}
    # B: per-country thresholds (tuned on that country's stack_train S1), unseen -> global
    tc = {}
    for c in sp['country'].unique().sort().to_list():
        Sc = sp.filter((pl.col('split') == 'stack_train') & (pl.col('country') == c))['i1']
        Tc = TR.join(pl.DataFrame({'i1': Sc}), on='i1', how='semi')
        tc[c] = best_t(Dx.join(pl.DataFrame({'i1': Sc}), on='i1', how='semi'), Tc, Sc)[0]
    thr = lambda: pl.col('country').replace_strict(tc, default=tA, return_dtype=pl.Float64)
    predB = lambda d: d.filter(pl.col('p') >= thr()).select('i1', 'ir')
    out['B_per_country'] = {'t': tc, 'oof': macro_f05(predB(Dx), T_ST, S_ST), 'val': macro_f05(predB(Dx), T_V, S_V, by=by)}
    # C: threshold + margin over the runner-up claimant
    bestC = (None, -1)
    for m in [0.0, 0.05, 0.1, 0.2, 0.3]:
        for t in np.round(np.arange(tA - 0.15, tA + 0.06, 0.01), 3):
            f = macro_f05(Dx.filter((pl.col('p') >= t) & (pl.col('p') - pl.col('_r2') >= m)).select('i1', 'ir'), T_ST, S_ST)
            if f > bestC[1]:
                bestC = ((float(t), m), f)
    (tC, mC), fC = bestC
    predC = lambda d: d.filter((pl.col('p') >= tC) & (pl.col('p') - pl.col('_r2') >= mC)).select('i1', 'ir')
    out['C_margin'] = {'t': tC, 'margin': mC, 'oof': fC, 'val': macro_f05(predC(Dx), T_V, S_V, by=by)}
    # D: expected-F lists
    out['D_expected_f'] = {'oof': macro_f05(rule_expected_f(D.select('i1', 'ir', 'p')), T_ST, S_ST),
                           'val': macro_f05(rule_expected_f(D.select('i1', 'ir', 'p')), T_V, S_V, by=by)}
    for k, v in out.items():
        log(k, json.dumps(v, default=lambda x: round(x, 5) if isinstance(x, float) else str(x)))
    best = max(out, key=lambda k: out[k]['oof'])
    log('best rule on stack_train OOF:', best)
    save_json(out, OUT / f'decision_study_{FEAT_VERSION}_{TAG}.json')
    if write:
        from .stack import write_submission
        te = pl.read_parquet(OUT / f'test_p_{FEAT_VERSION}_{TAG}.parquet').join(
            nt('test', '1', ['country']).with_row_index('i1'), on='i1', how='left')
        tx = exclusive(with_runner_up(te))
        pred = {'A_global': lambda d: d.filter(pl.col('p') >= tA).select('i1', 'ir'), 'B_per_country': predB, 'C_margin': predC,
                'D_expected_f': lambda d: rule_expected_f(te.select('i1', 'ir', 'p'))}[best](tx)
        from .stage2 import stage2_path
        write_submission(pred, pl.read_parquet(stage2_path('test'), columns=['i1', 'ir']), {'rule': best, **out[best], 'study': out})
    return out


if __name__ == '__main__':
    run(write='--write' in sys.argv)
