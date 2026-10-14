"""How resolvable are address-less records through exact-name siblings? (train, ghosted labels)"""
import polars as pl
from er_v2.data import truth, nt

TR = truth()
r = pl.concat([nt('train', s, ['name', 'country', 'has_addr', 'name_core']) for s in '23']).with_row_index('ir')
r = r.join(TR, on='ir', how='left')  # i1 = owning S1; null = distractor / ghost orphan
OUT = {}
for key, desc in [('name', 'exact RAW name'), ('name_core', 'normalised name_core')]:
    oth = r.filter(pl.col('i1').is_not_null()).select('country', key, pl.col('ir').alias('ir2'), pl.col('i1').alias('i1b'),
                                                       pl.col('has_addr').alias('ha2'))

    def profile(sub, tag):
        j = sub.select('ir', 'country', key, 'i1').join(oth, on=['country', key]).filter(pl.col('ir2') != pl.col('ir'))
        agg = j.group_by('ir').agg((pl.col('i1b') == pl.col('i1').first()).any().alias('same'),
                                   (pl.col('i1b') != pl.col('i1').first()).any().alias('other'),
                                   ((pl.col('i1b') == pl.col('i1').first()) & pl.col('ha2')).any().alias('same_addr'))
        a = sub.select('ir').join(agg, on='ir', how='left').with_columns(pl.col('same', 'other', 'same_addr').fill_null(False))
        res = {'n': a.height, 'shares key with sibling of OWN S1': round(a['same'].mean(), 4),
               'that sibling has an address': round(a['same_addr'].mean(), 4),
               'shares key with ANOTHER S1': round(a['other'].mean(), 4),
               'own S1 only (unique resolution)': round((a['same'] & ~a['other']).mean(), 4),
               'no labelled sibling at all': round((~a['same'] & ~a['other']).mean(), 4)}
        print(f'[{desc}] {tag}: {res}')
        OUT[f'{desc} | {tag}'] = res

    profile(r.filter(pl.col('i1').is_not_null() & ~pl.col('has_addr')), 'MATCHED records WITHOUT address')
    profile(r.filter(pl.col('i1').is_not_null() & pl.col('has_addr')), 'MATCHED records WITH address')
    d = r.filter(pl.col('i1').is_null() & ~pl.col('has_addr')).select('ir', 'country', key)
    hit = d.join(oth.select('country', key).unique(), on=['country', key], how='semi').height
    print(f'[{desc}] DISTRACTOR records without address: n={d.height:,} | key matches some labelled record: {hit / d.height:.4f}')
    OUT[f'{desc} | distractors without address matching a labelled key'] = round(hit / d.height, 4)
import json
json.dump(OUT, open('/teamspace/studios/this_studio/amlc/output/metrics/sibling_analysis.json', 'w'), indent=1)
