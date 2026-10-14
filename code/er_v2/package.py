"""Builds <team>_submission.zip exactly in the structure the problem statement requires:

<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/er_v2/*.py, README.md, requirements.txt}
└── Documentation_template.md
"""
import sys, zipfile, subprocess, importlib.metadata as md
from pathlib import Path
from .config import FINAL_OUT, OUT, TEAM_NAME, log

PKGS = ['polars', 'pyarrow', 'numpy', 'scikit-learn', 'scipy', 'lightgbm', 'rapidfuzz', 'anyascii', 'torch', 'transformers',
        'tokenizers', 'safetensors', 'joblib', 'psutil', 'huggingface-hub']


def requirements():
    lines = [f'# pinned from the reference run (Python {sys.version.split()[0]}, Lightning AI A100)']
    for p in PKGS:
        try:
            lines.append(f'{p}=={md.version(p)}')
        except md.PackageNotFoundError:
            pass
    return '\n'.join(lines) + '\n'


def build(doc_path, team=TEAM_NAME):
    src = Path(__file__).resolve().parent
    z = FINAL_OUT.parent / f'{team}_submission.zip'
    base = 'code/business_entity_resolution'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in ['matching_results.tsv', 'candidate_pairs.tsv']:
            zf.write(FINAL_OUT / f, f'output/{f}')
        for f in sorted(src.glob('*.py')) + sorted(src.glob('*.sh')):
            zf.write(f, f'{base}/src/er_v2/{f.name}')
        val = src.parent / 'validate_submission.py'
        if val.exists():
            zf.write(val, f'{base}/src/utils/validate_submission.py')
        nb = src.parents[1] / 'MLChallenge_v2.ipynb'
        if nb.exists():
            zf.write(nb, f'{base}/src/MLChallenge_v2.ipynb')
        zf.write(src / 'README.md', f'{base}/README.md')
        zf.writestr(f'{base}/requirements.txt', requirements())
        zf.write(doc_path, 'Documentation_template.md')
    names = zipfile.ZipFile(z).namelist()
    need = ['output/matching_results.tsv', 'output/candidate_pairs.tsv', f'{base}/README.md', f'{base}/requirements.txt', 'Documentation_template.md']
    ok = all(n in names for n in need) and any(n.startswith(f'{base}/src/') for n in names)
    log(f'{z} ({z.stat().st_size / 2**20:.1f} MiB, {len(names)} files) PACKAGE STRUCTURE: {"PASS" if ok else "FAIL"}')
    return z


if __name__ == '__main__':
    build(sys.argv[1], *(sys.argv[2:3]))
