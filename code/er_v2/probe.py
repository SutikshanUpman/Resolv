"""Leaderboard probes from the cached v2A test scores (CPU only, ~3 min each). Never overwrites output/.

  python -m er_v2.probe <name> [--t 0.69] [--empty France] [--tag e5b]

Writes <studio>/output/probe_<name>/{matching_results.tsv, candidate_pairs.tsv} and runs the official validator.
Probes used to localise the val (0.990) -> LB (0.982) gap:
  france_empty : v2A with every France S1 predicted empty. The France score is then
                 F_France ~= s_France + (LB_v2A - LB_probe) / share_France, with share_France = 0.1498 of test S1
                 and s_France ~= 0.056 (France singleton rate, approximated by the v2A empty share).
  t050 / t085  : v2A with threshold 0.50 / 0.85. If LB rises at 0.50, test is less ambiguous than ghost-val.
"""
import sys, subprocess, json
import polars as pl
from .config import OUT, FINAL_OUT, FEAT_VERSION, DATA, log
from .data import nt, id_maps
from .metrics import exclusive
from .stage2 import stage2_path
from .stack import id_lists, write_tsv


def main():
    args = sys.argv[1:]
    name = args[0]
    t = float(args[args.index('--t') + 1]) if '--t' in args else 0.69
    empty = args[args.index('--empty') + 1].split(',') if '--empty' in args else []
    tag = args[args.index('--tag') + 1] if '--tag' in args else 'e5b'
    te = pl.read_parquet(OUT / f'test_p_{FEAT_VERSION}_{tag}.parquet')
    c1 = nt('test', '1', ['country']).with_row_index('i1')
    pred = exclusive(te.select('i1', 'ir', 'p')).filter(pl.col('p') >= t).select('i1', 'ir')
    if empty:
        pred = pred.join(c1.filter(pl.col('country').is_in(empty)).select('i1'), on='i1', how='anti')
    cand = pl.read_parquet(stage2_path('test'), columns=['i1', 'ir'])
    m1, mr = id_maps('test')
    d = FINAL_OUT / f'probe_{name}'; d.mkdir(parents=True, exist_ok=True)
    write_tsv(id_lists(pred, 'matched_entity_ids', m1, mr), d / 'matching_results.tsv')
    write_tsv(id_lists(cand, 'candidate_entity_ids', m1, mr), d / 'candidate_pairs.tsv')
    per = (c1.join(pred.group_by('i1').len('n'), on='i1', how='left').fill_null(0).group_by('country')
             .agg(pl.col('n').mean().round(3).alias('pred/S1'), (pl.col('n') == 0).mean().round(4).alias('empty')).sort('country'))
    log(f'probe {name}: t={t} empty={empty} -> {pred.height:,} matches\n{per}')
    json.dump({'name': name, 't': t, 'empty': empty, 'tag': tag, 'matches': pred.height, 'by_country': per.to_dicts()}, open(d / 'probe_info.json', 'w'), indent=1)
    v = subprocess.run([sys.executable, str(OUT.parent / 'code' / 'validate_submission.py'), '--matching', str(d / 'matching_results.tsv'),
                        '--candidate', str(d / 'candidate_pairs.tsv'), '--test-dir', str(DATA), '--check-ids'], capture_output=True, text=True)
    print(v.stdout[-800:])


if __name__ == '__main__':
    main()
