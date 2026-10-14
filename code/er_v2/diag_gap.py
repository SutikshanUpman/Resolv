"""Diagnostics for the val (0.990) -> LB (0.982) gap and for the missing signal behind 0.999 (CPU, cached artifacts)."""
import json
import numpy as np
import polars as pl
from .config import OUT, FEAT_VERSION
from .data import splits, truth, nt
from .metrics import exclusive

pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(240); pl.Config.set_fmt_str_lengths(55)
R = {}
T_THR = 0.69

# ---------------------------------------------------------------- A. test conflicts (record claimed by >= 2 S1 with p >= 0.5)
c1t = nt('test', '1', ['country', 'name', 'addr']).with_row_index('i1')
rt = pl.concat([nt('test', s, ['name', 'addr']) for s in '23']).with_row_index('ir')
te = pl.read_parquet(OUT / f'test_p_{FEAT_VERSION}_e5b.parquet').join(c1t.select('i1', 'country'), on='i1')
hi = te.filter(pl.col('p') >= 0.5)
cf = hi.group_by('ir').agg(pl.len().alias('k'), pl.col('country').first())
R['test_conflicts_by_country'] = cf.group_by('country').agg((pl.col('k') > 1).mean().round(5).alias('conflict rate'),
                                                          (pl.col('k') > 1).sum().alias('conflicted records')).sort('country').to_dicts()
print('A. test conflict rate by country:', R['test_conflicts_by_country'])
ex = cf.filter(pl.col('k') > 1).sample(6, seed=4).select('ir')
x = hi.join(ex, on='ir').join(c1t.select('i1', pl.col('name').alias('s1_name'), pl.col('addr').alias('s1_addr')), on='i1') \
      .join(rt.select('ir', pl.col('name').alias('r_name'), pl.col('addr').alias('r_addr')), on='ir').sort('ir', 'p', descending=[False, True])
print(x.select('country', 'ir', 'r_name', 'r_addr', 's1_name', 's1_addr', pl.col('p').round(3)))
# same-name / same-address S1 duplicates inside test S1 (is S1 really deduplicated?)
for split in ['train', 'test']:
    s = nt(split, '1', ['country', 'name_sq', 'addr_norm'])
    dup = s.group_by('country', 'name_sq', 'addr_norm').len().filter(pl.col('len') > 1)
    R[f'{split}_S1_same_name_and_addr_groups'] = {'groups': dup.height, 'S1 in them': int(dup['len'].sum()), 'S1 total': s.height}
    print(f'{split}: S1 sharing BOTH normalised name and address:', R[f'{split}_S1_same_name_and_addr_groups'])

# ---------------------------------------------------------------- B. val address-less FNs: what separates the true S1 from competitors?
sp = splits(); V = sp.filter(pl.col('split') == 'val').select('i1')
TR = truth()
D = pl.read_parquet(OUT / f'eval_p_{FEAT_VERSION}_e5b.parquet')
Dx = exclusive(D.select('i1', 'ir', 'p'))
pred = Dx.filter(pl.col('p') >= T_THR).select('i1', 'ir')
T_V = TR.join(V, on='i1', how='semi')
fn = T_V.join(pred, on=['i1', 'ir'], how='anti').join(D.select('i1', 'ir'), on=['i1', 'ir'], how='semi')
r1 = pl.concat([nt('train', s, ['name', 'addr', 'has_addr', 'name_core', 'legal']) for s in '23']).with_row_index('ir')
s1 = nt('train', '1', ['name', 'addr', 'name_core', 'legal']).with_row_index('i1')
fn = fn.join(r1.select('ir', 'has_addr'), on='ir').filter(~pl.col('has_addr')).select('i1', 'ir')
# competitors = other S1 with this record among their candidates
comp = D.select('i1', 'ir', 'p').join(fn.select('ir'), on='ir', how='semi').join(fn.rename({'i1': 'true_i1'}), on='ir')
comp = comp.with_columns(is_true=pl.col('i1') == pl.col('true_i1'))
agg = comp.group_by('ir').agg(pl.col('p').filter(pl.col('is_true')).first().alias('p_true'),
                              pl.col('p').filter(~pl.col('is_true')).max().alias('p_best_other'),
                              (~pl.col('is_true')).sum().alias('n_other'))
R['fn_noaddr_competition'] = agg.select(pl.len().alias('fn address-less'), (pl.col('n_other') == 0).mean().round(4).alias('no competitor in candidates'),
                                        (pl.col('p_true') > pl.col('p_best_other')).mean().round(4).alias('true S1 ranked first'),
                                        pl.col('p_true').median().round(3).alias('median p_true'),
                                        pl.col('p_best_other').median().round(3).alias('median p_best_other')).to_dicts()[0]
print('\nB. val address-less FNs vs competitors:', R['fn_noaddr_competition'])
# raw-name agreement of the record with the true S1 vs the best competitor
best = comp.filter(~pl.col('is_true')).sort('p', descending=True).group_by('ir').first().select('ir', pl.col('i1').alias('comp_i1'))
cmpx = fn.join(best, on='ir', how='left').join(r1.select('ir', pl.col('name').alias('rn'), pl.col('name_core').alias('rc'), pl.col('legal').alias('rl')), on='ir') \
         .join(s1.select('i1', pl.col('name').alias('tn'), pl.col('name_core').alias('tc'), pl.col('legal').alias('tl')), on='i1') \
         .join(s1.select(pl.col('i1').alias('comp_i1'), pl.col('name').alias('cn'), pl.col('name_core').alias('cc'), pl.col('legal').alias('cl')), on='comp_i1', how='left')
has_c = pl.col('comp_i1').is_not_null()
R['fn_noaddr_name_signal'] = cmpx.filter(has_c).select(
    pl.len().alias('with competitor'),
    (pl.col('rn').str.to_lowercase() == pl.col('tn').str.to_lowercase()).mean().round(4).alias('raw==true S1'),
    (pl.col('rn').str.to_lowercase() == pl.col('cn').str.to_lowercase()).mean().round(4).alias('raw==competitor'),
    (pl.col('tn').str.to_lowercase() == pl.col('cn').str.to_lowercase()).mean().round(4).alias('true S1 raw == competitor raw'),
    (pl.col('tc') == pl.col('cc')).mean().round(4).alias('true core == competitor core'),
    ((pl.col('rl') == pl.col('tl')) & (pl.col('rl') != pl.col('cl'))).mean().round(4).alias('legal form matches true only'),
    ((pl.col('rl') == pl.col('cl')) & (pl.col('rl') != pl.col('tl'))).mean().round(4).alias('legal form matches competitor only')).to_dicts()[0]
print('B2. name signal (records with a competitor):', R['fn_noaddr_name_signal'])
print(cmpx.filter(has_c).sample(10, seed=1).select('rn', 'tn', 'cn'))
# S1 id / row-order signal among same-name S1 (does the true owner have a systematic position?)
pos = cmpx.filter(has_c).select((pl.col('i1') < pl.col('comp_i1')).mean().round(4).alias('true S1 row < competitor row'))
print('B3. row order true vs competitor:', pos.to_dicts()[0]); R['fn_noaddr_order'] = pos.to_dicts()[0]

# ---------------------------------------------------------------- C. France vs others on test: ambiguity drivers
r_ha = pl.concat([nt('test', s, ['has_addr', 'country']) for s in '23'])
R['test_record_no_address_share'] = r_ha.group_by('country').agg((~pl.col('has_addr')).mean().round(4).alias('no address')).sort('country').to_dicts()
s1t = nt('test', '1', ['country', 'name_sq'])
cnt = s1t.group_by('country', 'name_sq').len()
R['test_S1_name_shared_share'] = s1t.join(cnt, on=['country', 'name_sq']).group_by('country').agg((pl.col('len') > 1).mean().round(4).alias('S1 normalised name shared')).sort('country').to_dicts()
print('\nC. test record without address:', R['test_record_no_address_share'])
print('C. test S1 normalised name shared:', R['test_S1_name_shared_share'])
s1v = nt('train', '1', ['country', 'name_sq'])
cntv = s1v.group_by('country', 'name_sq').len()
R['ghost_train_S1_name_shared_share'] = s1v.join(cntv, on=['country', 'name_sq']).group_by('country').agg((pl.col('len') > 1).mean().round(4).alias('S1 normalised name shared')).sort('country').to_dicts()
print('C. ghost-train S1 normalised name shared:', R['ghost_train_S1_name_shared_share'])
json.dump(R, open(OUT / 'metrics' / 'diag_gap.json', 'w'), indent=1, default=str)
print('wrote diag_gap.json')
