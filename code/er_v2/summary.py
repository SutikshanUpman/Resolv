"""Compact report of one test run (reads the run's eval / test predictions; run with the same ER_* env as the run).

val: macro F0.5 / P / R by country, group size and singletons, the loss decomposition by error type, and where the
missed true pairs sit (never retrieved / stage-2 cut / matcher). test: predicted matches and empty share per country,
and the share of acronym-like records that are matched (the France acronym check).
"""
import sys, json, glob, os
import numpy as np
import polars as pl
from .config import OUT, FINAL_OUT, FEAT_VERSION, STAGE2_VERSION, L1_TAG, STACK_TAG, log, save_json
from .data import splits, truth, nt, id_maps
from .metrics import exclusive
from .stage2 import stage2_path, union_path

TAG = 'e5b'


def per_s1(pred, T, S1):
    base = pl.DataFrame({'i1': pl.Series(S1).cast(pl.UInt32)})
    pred = pred.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32))
    T = T.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32)).join(base, on='i1', how='semi')
    tp = pred.join(T, on=['i1', 'ir']).group_by('i1').len('tp')
    d = (base.join(pred.group_by('i1').len('np'), on='i1', how='left').join(T.group_by('i1').len('nt'), on='i1', how='left')
             .join(tp, on='i1', how='left').fill_null(0))
    P = pl.col('tp') / pl.col('np'); R = pl.col('tp') / pl.col('nt')
    return d.with_columns(
        f=pl.when((pl.col('nt') == 0) & (pl.col('np') == 0)).then(1.0).when((pl.col('nt') == 0) | (pl.col('np') == 0) | (pl.col('tp') == 0)).then(0.0)
          .otherwise(1.25 * P * R / (0.25 * P + R)),
        prec=pl.when(pl.col('np') > 0).then(P).otherwise(None), rec=pl.when(pl.col('nt') > 0).then(R).otherwise(None),
        cat=pl.when((pl.col('nt') == 0) & (pl.col('np') == 0)).then(pl.lit('singleton ok'))
             .when(pl.col('nt') == 0).then(pl.lit('singleton predicted non-empty'))
             .when(pl.col('np') == 0).then(pl.lit('non-singleton predicted empty'))
             .when((pl.col('tp') == pl.col('np')) & (pl.col('tp') == pl.col('nt'))).then(pl.lit('perfect'))
             .when(pl.col('tp') == pl.col('np')).then(pl.lit('missed record(s) only'))
             .when(pl.col('tp') == pl.col('nt')).then(pl.lit('extra wrong record only'))
             .otherwise(pl.lit('both')))


def run(threshold=None, ev_name=None):
    hist = sorted(glob.glob(str(OUT / 'submission_history' / '*')))[-1]
    info = json.load(open(os.path.join(hist, 'run_info.json')))
    t = threshold if threshold is not None else info.get('threshold', 0.69)
    sp = splits(); V = sp.filter(pl.col('split') == 'val')
    ev = pl.read_parquet(OUT / (ev_name or f'eval_p_{FEAT_VERSION}{L1_TAG}{STACK_TAG}_{TAG}.parquet'))
    t1 = info.get('t1', t) if threshold is None else t     # rank-aware rule (collective runs): best pair of each S1 at t1
    pred = (exclusive(ev.select('i1', 'ir', 'p')).with_columns(k=pl.col('p').rank('ordinal', descending=True).over('i1'))
              .filter(((pl.col('k') == 1) & (pl.col('p') >= t1)) | (pl.col('p') >= t)).select('i1', 'ir'))
    TV = truth().join(V.select('i1'), on='i1', how='semi')
    d = per_s1(pred, TV, V['i1']).join(V.select(pl.col('i1').cast(pl.UInt32), 'country'), on='i1')
    N = d.height
    out = {'run': hist, 'feat': FEAT_VERSION, 'stage2': STAGE2_VERSION, 'threshold': t, 't1': t1,
           'val_F05': round(d['f'].mean(), 5), 'macro_P': round(d['prec'].mean(), 5), 'macro_R': round(d['rec'].mean(), 5),
           'by_country': {c: round(v, 5) for c, v in d.group_by('country').agg(pl.col('f').mean()).iter_rows()},
           'singletons': round(d.filter(pl.col('nt') == 0)['f'].mean(), 5),
           'by_group_size': {str(g): round(v, 5) for g, v in d.with_columns(g=pl.col('nt').clip(0, 7)).group_by('g').agg(pl.col('f').mean()).sort('g').iter_rows()},
           'loss_by_type': {c: round(v / N, 5) for c, v in d.group_by('cat').agg((1 - pl.col('f')).sum()).sort('cat').iter_rows() if v > 0}}
    # where the missed true pairs are
    un = pl.scan_parquet(union_path('train')).select('i1', 'ir').join(TV.lazy(), on=['i1', 'ir'], how='semi').collect()
    s2 = pl.read_parquet(stage2_path('train'), columns=['i1', 'ir']).join(TV, on=['i1', 'ir'], how='semi')
    hit = pred.join(TV, on=['i1', 'ir'], how='semi')
    out['true_pairs'] = {'total': TV.height, 'not_in_union': TV.height - un.height, 'cut_at_stage2': un.height - s2.height,
                         'rejected_by_matcher': s2.height - hit.height, 'hit': hit.height,
                         'candidate_recall': round(s2.height / TV.height, 5), 'final_recall': round(hit.height / TV.height, 5)}
    # test
    m = pl.read_csv(FINAL_OUT / 'matching_results.tsv', separator='\t', quote_char=None, infer_schema=False)
    c1 = nt('test', '1', ['entity_id', 'country'])
    m = m.join(c1, left_on='source1_entity_id', right_on='entity_id')
    m = m.with_columns(n=pl.when(pl.col('matched_entity_ids').fill_null('') == '').then(0)
                         .otherwise(pl.col('matched_entity_ids').str.count_matches(',') + 1))
    out['test'] = {c: {'pred/S1': round(a, 3), 'empty': round(b, 4)} for c, a, b in
                   m.group_by('country').agg(pl.col('n').mean().alias('a'), (pl.col('n') == 0).mean().alias('b')).sort('country').iter_rows()}
    rr = pl.concat([nt('test', s, ['entity_id', 'name', 'country']) for s in '23'])
    matched = m.filter(pl.col('n') > 0).select(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids')['matched_entity_ids']
    rr = rr.with_columns(acr=pl.col('name').str.strip_chars().str.replace_all(r'[\.\s]', '').str.contains(r'^[A-Z]{2,5}$'),
                         matched=pl.col('entity_id').is_in(matched.implode()))
    out['test_acronym_records_matched_share'] = {c: round(v, 4) for c, v in rr.filter('acr').group_by('country').agg(pl.col('matched').mean()).sort('country').iter_rows()}
    print(json.dumps(out, indent=1))
    save_json(out, OUT / 'metrics' / f'summary_{STAGE2_VERSION}_{FEAT_VERSION}{L1_TAG}{STACK_TAG}{"_coll" if ev_name else ""}.json')
    return out


if __name__ == '__main__':
    run(float(sys.argv[1]) if len(sys.argv) > 1 else None)
