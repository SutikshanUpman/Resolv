"""Builds the minimal bundle needed to run the final pass (chain_final.sh) on another studio.

Keeps the amlc/ layout, so on the new studio the code runs unchanged after `tar -xf` into
/teamspace/studios/this_studio/. Parquet tables are slimmed to the columns the final pass reads.

    python -m er_v2.make_transfer_bundle          # -> /teamspace/studios/this_studio/final_bundle.tar + MANIFEST.txt
"""
import hashlib, shutil, tarfile, time
from pathlib import Path
import polars as pl

HOME = Path('/teamspace/studios/this_studio')
SRC = HOME / 'amlc'
B = HOME / 'final_bundle' / 'amlc'
NORM_COLS = ['entity_id', 'country', 'name', 'addr', 'name_core', 'name_sk', 'addr_norm']
COPY = ['output/eval_p_f3_e5b.parquet', 'output/test_p_f3_e5b.parquet',
        'output/qwen_scores_train.parquet', 'output/qwen_scores_test.parquet',
        'output/eval_pfull_f3br_e5b.parquet', 'output/test_pfull_f3br_e5b.parquet',
        'output/splits_train_v2.parquet', 'output/truth_train_v2.parquet',
        'code/validate_submission.py', 'code/run_bg.sh']
SLIM = {'output/blocking/train_stage2_s3.parquet': ['i1', 'ir', 'p_rr'],
        'output/blocking/test_stage2_s3.parquet': ['i1', 'ir']}


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''):
            h.update(c)
    return h.hexdigest()


def main():
    t = time.time()
    if B.parent.exists():
        shutil.rmtree(B.parent)
    (B / 'output' / 'norm').mkdir(parents=True); (B / 'output' / 'blocking').mkdir(parents=True)
    shutil.copytree(SRC / 'code' / 'er_v2', B / 'code' / 'er_v2', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(SRC / 'output' / 'qwen_rr06', B / 'output' / 'qwen_rr06', ignore=shutil.ignore_patterns('checkpoint'))
    for rel in COPY:
        shutil.copy2(SRC / rel, B / rel)
    for rel, cols in SLIM.items():
        pl.read_parquet(SRC / rel, columns=cols).write_parquet(B / rel)
    for p in sorted((SRC / 'output' / 'norm').glob('*_v2.parquet')):
        if p.name.startswith('train_full'):
            continue                                   # the un-ghosted train S1 table is not read by the final pass
        cols = [c for c in NORM_COLS if c in pl.read_parquet_schema(p)]
        pl.read_parquet(p, columns=cols).write_parquet(B / 'output' / 'norm' / p.name)
    files = sorted(x for x in B.rglob('*') if x.is_file())
    lines = [f'{md5(f)}  {f.stat().st_size:>12,}  {f.relative_to(B.parent)}' for f in files]
    (B.parent / 'MANIFEST.txt').write_text('\n'.join(lines) + '\n')
    tar = HOME / 'final_bundle.tar'
    with tarfile.open(tar, 'w') as tf:                 # parquet / safetensors are already compressed: plain tar is fastest
        tf.add(B.parent, arcname='final_bundle')
    size = sum(f.stat().st_size for f in files)
    print('\n'.join(lines))
    print(f'\n{len(files)} files, {size / 2**30:.2f} GiB -> {tar} ({tar.stat().st_size / 2**30:.2f} GiB), md5 {md5(tar)}, {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
