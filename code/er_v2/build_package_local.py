"""Assembles the final hackathon package on the laptop, in the structure the problem statement requires:

<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md

    python er_v2/build_package_local.py test3  <matching.tsv> <candidate.tsv> <notebook.ipynb>
    python er_v2/build_package_local.py stageA <matching.tsv> <candidate.tsv> <notebook.ipynb>
Output: final_packages/<variant>/The_Epoch_Warriors_submission/ (folder) + The_Epoch_Warriors_submission.zip
"""
import sys, shutil, zipfile, hashlib
from pathlib import Path

TEAM = 'The_Epoch_Warriors'
REQ = """# pinned from the reference runs (Python 3.12.11; Lightning AI A100-40GB / H100-80GB, CUDA 12.8)
polars==1.44.2
pyarrow==25.0.1
numpy==1.26.4
scikit-learn==1.3.2
scipy==1.11.4
lightgbm==4.7.0
rapidfuzz==3.14.6
anyascii==0.3.3
torch==2.8.0+cu128
transformers==5.17.0
tokenizers==0.23.2
safetensors==0.8.0
joblib==1.6.0
psutil==7.2.2
huggingface-hub==1.33.0
# notebook only
matplotlib>=3.8
nbformat>=5.9
"""
STAGE_A_README = """
**Final submission = Stage A.** After step 4, score the wider uncertainty band with the same Qwen model (existing
scores are reused), rerun the collective pass with a global threshold, and write the files:

```bash
ER_STAGE2=s3 ER_FEAT=f3 ER_QW_BAND=0.02,0.98 ER_QW_MODEL=qwen_rr06 ER_QW_OUT=_wide ER_QW_REUSE= python -m er_v2.qwen_ce score
ER_STAGE2=s3 ER_FEAT=f3 ER_L1=b ER_STACK=r ER_QW=1 ER_QW_FILES=_wide:qw ER_QW_TAG=_qwA ER_RULE=global python -m er_v2.collective
ER_STAGE2=s3 ER_FEAT=f3 python -m er_v2.write_final test_p_f3br_qwA_coll_e5b.parquet 0.70 $ER_ROOT/../output
```
(`chain_final.sh` runs this Stage A together with the Stage B experiment, which was not selected.)
"""


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''):
            h.update(c)
    return h.hexdigest()


def variant_texts(which, root):
    readme = (root / 'er_v2' / 'README.md').read_text(encoding='utf-8')
    doc = (root / 'submission' / 'Documentation_template.md').read_text(encoding='utf-8')
    if which == 'stageA':
        readme = readme.replace('**Submitted result:** public leaderboard **0.986**; ghost validation (test-like, 100k train S1) macro F0.5 **0.99153**.',
                                '**Submitted result: Stage A.** Ghost validation (test-like, 100k train S1) macro F0.5 **0.99158** '
                                '(Test 3: 0.99153, public leaderboard 0.986).')
        readme = readme.replace('**5. Validate:**', STAGE_A_README + '\n**5. Validate:**')
        doc = doc.replace('**Final result: public leaderboard 0.986** (v2: 0.982). Test-like validation macro F0.5: **0.99153**, with macro precision 0.9980 and recall 0.9777.',
                          '**Final submission: Stage A**, which extends Qwen coverage to the band 0.02–0.98. Test-like validation macro F0.5: **0.99158**, with macro precision 0.9980, recall 0.9778 and singletons 0.9952. The previous best, Test 3, scored **0.986** on the public leaderboard (v2: 0.982).')
        doc = doc.replace('| **+ Qwen3-Reranker-0.6B (final)** | **0.99153** | **0.986** |',
                          '| + Qwen3-Reranker-0.6B (Test 3) | 0.99153 | 0.986 |\n'
                          '| **+ Qwen score on the wide band 0.02–0.98 (Stage A, final)** | **0.99158** | – |\n'
                          '| + Qwen with French-style training noise (Stage B, not selected: lower singletons, 0.9946) | 0.99160 | – |')
        doc = doc.replace('- **F_0.5 Score (macro):** 0.99153 on 100,000 held-out train S1 under the test-like distribution (India 0.9926, US 0.9908; singletons 0.9950). Public leaderboard: **0.986**.',
                          '- **F_0.5 Score (macro):** 0.99158 on 100,000 held-out train S1 under the test-like distribution (India 0.9928, US 0.9908; singletons 0.9952). Test 3, which differs only in Qwen coverage, scored **0.986** on the public leaderboard.')
    return readme, doc


def main(which, matching, candidate, notebook):
    root = Path.cwd()
    out = root / 'final_packages' / which; pkg = out / f'{TEAM}_submission'
    if out.exists():
        shutil.rmtree(out)
    code = pkg / 'code' / 'business_entity_resolution'; src = code / 'src'
    (pkg / 'output').mkdir(parents=True); (src / 'utils').mkdir(parents=True)
    shutil.copy2(matching, pkg / 'output' / 'matching_results.tsv'); shutil.copy2(candidate, pkg / 'output' / 'candidate_pairs.tsv')
    shutil.copytree(root / 'er_v2', src / 'er_v2', ignore=shutil.ignore_patterns('__pycache__', 'README.md'))
    shutil.copy2(root / 'submissions' / 'validate_submission.py', src / 'utils' / 'validate_submission.py')
    shutil.copy2(notebook, src / 'MLChallenge_v3.ipynb')
    shutil.copy2(root / 'MLChallenge_v2.ipynb', src / 'MLChallenge_v2.ipynb')
    readme, doc = variant_texts(which, root)
    (code / 'README.md').write_text(readme, encoding='utf-8'); (code / 'requirements.txt').write_text(REQ, encoding='utf-8')
    (pkg / 'Documentation_template.md').write_text(doc, encoding='utf-8')
    z = out / f'{TEAM}_submission.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pkg.rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(pkg).as_posix())
    names = zipfile.ZipFile(z).namelist()
    need = ['output/matching_results.tsv', 'output/candidate_pairs.tsv', 'code/business_entity_resolution/README.md',
            'code/business_entity_resolution/requirements.txt', 'Documentation_template.md']
    ok = all(n in names for n in need) and any(n.startswith('code/business_entity_resolution/src/') for n in names)
    print(f'{z} ({z.stat().st_size / 2**20:.1f} MiB, {len(names)} files) STRUCTURE: {"PASS" if ok else "FAIL"}')
    print('matching md5', md5(pkg / 'output' / 'matching_results.tsv'), '| candidate md5', md5(pkg / 'output' / 'candidate_pairs.tsv'))


if __name__ == '__main__':
    main(*sys.argv[1:5])
