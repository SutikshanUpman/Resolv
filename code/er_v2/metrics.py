"""Exact PS metric (per-S1 F0.5, macro-averaged over ALL S1 incl. singletons), candidate-set diagnostics, decision rules."""
import numpy as np
import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids, by: pl.DataFrame = None):
    """pred/truth: (i1, ir). s1_ids: every S1 of the evaluation set. by: optional (i1, group) -> per-group scores too."""
    base = pl.DataFrame({'i1': pl.Series(s1_ids).unique().cast(pl.UInt32)})
    pred = pred.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32)).unique().join(base, on='i1', how='semi')
    truth = truth.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32)).join(base, on='i1', how='semi')
    tp = pred.join(truth, on=['i1', 'ir']).group_by('i1').len('tp')
    d = (base.join(pred.group_by('i1').len('np'), on='i1', how='left').join(truth.group_by('i1').len('nt'), on='i1', how='left')
             .join(tp, on='i1', how='left').fill_null(0))
    P = pl.col('tp') / pl.col('np'); R = pl.col('tp') / pl.col('nt')
    f = (pl.when((pl.col('nt') == 0) & (pl.col('np') == 0)).then(1.0)
           .when((pl.col('nt') == 0) | (pl.col('np') == 0) | (pl.col('tp') == 0)).then(0.0)
           .otherwise(1.25 * P * R / (0.25 * P + R))).alias('f')
    d = d.with_columns(f)
    if by is None:
        return d['f'].mean()
    g = d.join(by.select(pl.col('i1').cast(pl.UInt32), 'group'), on='i1', how='left')
    out = {'all': d['f'].mean()}
    out.update({k: v for k, v in g.group_by('group').agg(pl.col('f').mean()).iter_rows()})
    out['singletons'] = d.filter(pl.col('nt') == 0)['f'].mean()
    return out


def candidate_report(cand: pl.DataFrame, truth: pl.DataFrame, s1_ids, stage: str) -> dict:
    """Pair recall / precision and the oracle F0.5 (a perfect matcher restricted to the candidates)."""
    base = pl.DataFrame({'i1': pl.Series(s1_ids).unique().cast(pl.UInt32)})
    cand = cand.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32)).join(base, on='i1', how='semi')
    tr = truth.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32)).join(base, on='i1', how='semi')
    hit = cand.join(tr, on=['i1', 'ir'])
    return {'stage': stage, 'S1': base.height, 'candidates': cand.height, 'cand / S1': round(cand.height / base.height, 3),
            'pair recall': round(hit.height / max(tr.height, 1), 4), 'pair precision': round(hit.height / max(cand.height, 1), 4),
            'oracle F0.5': round(macro_f05(hit, tr, s1_ids), 5)}


# ---------------------------------------------------------------- decision rules (p = final match probability)
def exclusive(d: pl.DataFrame, p='p') -> pl.DataFrame:
    """Each S2/S3 record belongs to at most one S1: keep it only for the S1 that scores it highest."""
    return d.filter(pl.col(p) >= pl.col(p).max().over('ir'))


def rule_threshold(d, t, p='p', excl=True):
    d = exclusive(d, p) if excl else d
    return d.filter(pl.col(p) >= t).select('i1', 'ir')


def rule_expected_f(d, p='p', excl=True, min_p=0.0):
    """Per S1, the top-k list that maximises expected F0.5 (empty list scores P(singleton) = prod(1-p))."""
    d = exclusive(d, p) if excl else d
    d = d.filter(pl.col(p) >= min_p).sort(['i1', p], descending=[False, True])
    d = d.with_columns(k=(pl.int_range(pl.len()).over('i1') + 1), cs=pl.col(p).cum_sum().over('i1'), et=pl.col(p).sum().over('i1'),
                       p0=(1 - pl.col(p).clip(0, 1 - 1e-6)).log().sum().over('i1').exp())
    d = d.with_columns(ef=1.25 * pl.col('cs') / (0.25 * pl.col('et') + pl.col('k')))
    d = d.with_columns(best_k=pl.col('k').sort_by('ef', descending=True).first().over('i1'), best_ef=pl.col('ef').max().over('i1'))
    return d.filter((pl.col('best_ef') > pl.col('p0')) & (pl.col('k') <= pl.col('best_k'))).select('i1', 'ir')


def tune_threshold(d, truth, s1_ids, p='p', excl=True, grid=None):
    grid = np.round(np.arange(0.2, 0.96, 0.02), 3) if grid is None else grid
    dd = exclusive(d, p) if excl else d
    res = [(float(t), macro_f05(dd.filter(pl.col(p) >= t).select('i1', 'ir'), truth, s1_ids)) for t in grid]
    return max(res, key=lambda x: x[1]), res
