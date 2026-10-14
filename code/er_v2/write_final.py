"""Writes the final submission from the collective model's test probabilities (the last step of the pipeline).

Decision (chosen on stack_train OOF, reported on val): global exclusive assignment (each S2/S3 record is kept only for
its highest-scoring S1), then keep pairs with p >= threshold. The candidate file is the stage-2 v3 candidate set.

    ER_STAGE2=s3 ER_FEAT=f3 python -m er_v2.write_final test_p_f3br_qw_coll_e5b.parquet 0.70 <out_dir>
    (Test 3 / leaderboard 0.986: test_p_f3br_qw_coll_e5b.parquet, threshold 0.70)
"""
import sys
from pathlib import Path
import polars as pl
from .config import OUT, log
from .metrics import exclusive
from .stage2 import stage2_path


def main(test_p, t, out_dir):
    from . import stack
    stack.FINAL_OUT = Path(out_dir); stack.FINAL_OUT.mkdir(parents=True, exist_ok=True)
    d = pl.read_parquet(OUT / test_p).select('i1', 'ir', 'p')
    pred = exclusive(d).filter(pl.col('p') >= t).select('i1', 'ir')
    log(f'{test_p}: {pred.height:,} matched pairs at threshold {t}')
    stack.write_submission(pred, pl.read_parquet(stage2_path('test'), columns=['i1', 'ir']),
                           {'rule': 'threshold+exclusive', 'threshold': t, 'source': test_p})


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3])
