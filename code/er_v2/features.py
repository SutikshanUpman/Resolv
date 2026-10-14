"""Step 4: full pair features on the final (stage-2) candidates of every train S1 and every test S1.

v1's 43 full features (name/address string similarities, char n-gram TF-IDF cosines, legal-form agreement, script and
domain flags, number / ZIP / state agreement, presence flags) on top of the stage-2 fast features, plus v2 additions:
  * multiplicity: how many S1 / S2+S3 records share this name (squashed, skeleton) or this address, on each side.
    Same-name businesses at different addresses are the main source of false merges, and their frequency is
    visible without labels (computed per split, like IDF).
  * missing-field augmentation (train only): for a random 15 % of train S1 the ZIP / state evidence is blanked, so
    France (no ZIP, no known state dictionary) looks in-distribution instead of looking like "missing = weak pair".
"""
import gc, time
import numpy as np
import polars as pl
import joblib
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, Prefix, Postfix
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from .config import OUT, SEED, FEAT_VERSION, log
from .data import nt, truth
from .stage2 import stage2_path, RR_FEATS_ALL, cp

FULL_COLS = ['name', 'addr', 'name_norm', 'name_core', 'name_sq', 'name_sk', 'legal', 'is_domain', 'name_indic', 'addr_norm',
             'addr_nums', 'zip', 'state', 'has_addr', 'addr_indic', 'country']
AUG_FRAC = 0.15
FEAT_DIR = OUT / 'features'
FEAT_DIR.mkdir(parents=True, exist_ok=True)
TFIDF_PATH = OUT / 'tfidf_char_v2.joblib'

MULT_FEATS = ['m_s1_sq', 'm_r_sq', 'm_s1_sk', 'm_r_sk', 'm_s1_addr', 'm_r_addr', 'mr_s1_sq', 'mr_r_sq', 'mr_s1_addr', 'mr_r_addr']
FULL_FEATS = ['n_tsort', 'n_partial', 'n_lev', 'n_prefix', 'n_postfix', 'n_full_tset', 'n_tfidf', 'n_jacc',
              'n_len1', 'n_len2', 'n_ntok1', 'n_ntok2', 'n_len_diff', 'legal_eq', 'legal_conf', 'legal1', 'legal2', 'dom1', 'dom2',
              'indic1', 'indic2', 'a_tsort', 'a_ratio', 'a_partial', 'a_tfidf', 'a_jacc', 'num_inter', 'num1', 'num2', 'num_jacc',
              'num_first_eq', 'state_miss', 'state_eq', 'a_len1', 'a_len2', 'has_a1', 'has_a2', 'aindic1', 'aindic2', 'p_rel', 'n_cand',
              'n_raw_eq', 'a_raw_eq']
FEATS = RR_FEATS_ALL + ['p_rr', 'rr_rank'] + FULL_FEATS + MULT_FEATS


def feat_path(split):
    return FEAT_DIR / f'{split}_{FEAT_VERSION}.parquet'


def tfidf():
    if TFIDF_PATH.exists():
        return joblib.load(TFIDF_PATH)
    smp = pl.concat([nt('train', s, ['name_core', 'addr_norm']).sample(400_000, seed=SEED) for s in '123'])
    hv_n = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 4), n_features=2**20, alternate_sign=False, norm=None)
    hv_a = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 3), n_features=2**20, alternate_sign=False, norm=None)
    tf_n = TfidfTransformer(sublinear_tf=True).fit(hv_n.transform(smp['name_core'].to_list()))
    tf_a = TfidfTransformer(sublinear_tf=True).fit(hv_a.transform(smp['addr_norm'].fill_null('').to_list()))
    joblib.dump((hv_n, tf_n, hv_a, tf_a), TFIDF_PATH)
    return hv_n, tf_n, hv_a, tf_a


def tfidf_cos(a, b, hv, tf):
    A = tf.transform(hv.transform(a)); B = tf.transform(hv.transform(b))
    return np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)


def side_full(split):
    """Per-record columns + multiplicity counts (within split and country)."""
    s1 = nt(split, '1', FULL_COLS).with_row_index('i1')
    r = pl.concat([nt(split, s, FULL_COLS) for s in '23']).with_row_index('ir')
    def counts(col):
        a = s1.group_by('country', col).len('n1'); b = r.group_by('country', col).len('nr')
        return a, b
    out1, outr = s1, r
    for col, tag in [('name_sq', 'sq'), ('name_sk', 'sk'), ('addr_norm', 'addr')]:
        a, b = counts(col)
        out1 = out1.join(a, on=['country', col], how='left').join(b, on=['country', col], how='left').rename({'n1': f'c1_{tag}', 'nr': f'cr_{tag}'})
        outr = outr.join(a, on=['country', col], how='left').join(b, on=['country', col], how='left').rename({'n1': f'c1_{tag}', 'nr': f'cr_{tag}'})
    fill = [pl.col(c).fill_null(0) for c in out1.columns if c.startswith('c1_') or c.startswith('cr_')]
    return out1.with_columns(fill).sort('i1'), outr.with_columns(fill).sort('ir')


def full_features(c: pl.DataFrame, s1: pl.DataFrame, r: pl.DataFrame, tf) -> pl.DataFrame:
    """c: stage-2 rows (i1, ir, fast features, p_rr, rr_rank). Returns c + full features."""
    hv_n, tf_n, hv_a, tf_a = tf
    d = c.join(s1.drop('country'), on='i1').join(r.drop('country'), on='ir', suffix='_r')
    L = lambda col: d[col].fill_null('').to_list()
    n1, n2, q1, q2, a1, a2 = L('name_core'), L('name_core_r'), L('name_sq'), L('name_sq_r'), L('addr_norm'), L('addr_norm_r')
    a_miss = d['a_miss'].to_numpy().astype(bool)
    def amask(v): v = v.astype(np.float32); v[a_miss] = -1; return v
    new = {
        'n_tsort': cp(n1, n2, fuzz.token_sort_ratio), 'n_partial': cp(n1, n2, fuzz.partial_ratio),
        'n_lev': cp(q1, q2, Levenshtein.normalized_similarity), 'n_prefix': cp(q1, q2, Prefix.normalized_similarity),
        'n_postfix': cp(q1, q2, Postfix.normalized_similarity), 'n_full_tset': cp(L('name_norm'), L('name_norm_r'), fuzz.token_set_ratio),
        'n_tfidf': tfidf_cos(n1, n2, hv_n, tf_n),
        'a_tsort': amask(cp(a1, a2, fuzz.token_sort_ratio)), 'a_ratio': amask(cp(a1, a2, fuzz.ratio)),
        'a_partial': amask(cp(a1, a2, fuzz.partial_ratio)), 'a_tfidf': amask(tfidf_cos(a1, a2, hv_a, tf_a)),
    }
    d = d.with_columns(**{k: pl.Series(v) for k, v in new.items()})
    tok = lambda col: pl.col(col).fill_null('').str.split(' ').list.eval(pl.element().filter(pl.element() != ''))
    inter = lambda x, y: x.list.set_intersection(y).list.len()
    union = lambda x, y: x.list.set_union(y).list.len()
    nt1, nt2, at1, at2 = tok('name_core'), tok('name_core_r'), tok('addr_norm'), tok('addr_norm_r')
    num1, num2 = pl.col('addr_nums'), pl.col('addr_nums_r')
    l1, l2 = num1.list.len().fill_null(0), num2.list.len().fill_null(0)
    no_num = (l1 == 0) | (l2 == 0)
    d = d.with_columns(
        n_jacc=(inter(nt1, nt2) / union(nt1, nt2).clip(1)).cast(pl.Float32),
        n_len1=pl.col('name_core').str.len_chars(), n_len2=pl.col('name_core_r').str.len_chars(),
        n_ntok1=nt1.list.len(), n_ntok2=nt2.list.len(),
        legal_eq=((pl.col('legal') == pl.col('legal_r')) & (pl.col('legal') != '')).cast(pl.Int8),
        legal_conf=((pl.col('legal') != pl.col('legal_r')) & (pl.col('legal') != '') & (pl.col('legal_r') != '')).cast(pl.Int8),
        legal1=(pl.col('legal') != '').cast(pl.Int8), legal2=(pl.col('legal_r') != '').cast(pl.Int8),
        dom1=pl.col('is_domain').cast(pl.Int8), dom2=pl.col('is_domain_r').cast(pl.Int8),
        indic1=pl.col('name_indic').fill_null(False).cast(pl.Int8), indic2=pl.col('name_indic_r').fill_null(False).cast(pl.Int8),
        a_jacc=pl.when(pl.col('a_miss') == 1).then(-1.0).otherwise(inter(at1, at2) / union(at1, at2).clip(1)).cast(pl.Float32),
        num_inter=inter(num1, num2).fill_null(0), num1=l1, num2=l2,
        num_jacc=pl.when(no_num).then(-1.0).otherwise(inter(num1, num2) / union(num1, num2).clip(1)).cast(pl.Float32),
        num_first_eq=pl.when(no_num).then(-1).otherwise((num1.list.first() == num2.list.first()).cast(pl.Int8)),
        state_miss=(pl.col('state').is_null() | pl.col('state_r').is_null()).cast(pl.Int8),
        state_eq=pl.when(pl.col('state').is_null() | pl.col('state_r').is_null()).then(-1).otherwise((pl.col('state') == pl.col('state_r')).cast(pl.Int8)),
        a_len1=pl.col('addr_norm').str.len_chars().fill_null(0), a_len2=pl.col('addr_norm_r').str.len_chars().fill_null(0),
        has_a1=pl.col('has_addr').cast(pl.Int8), has_a2=pl.col('has_addr_r').cast(pl.Int8),
        aindic1=pl.col('addr_indic').cast(pl.Int8), aindic2=pl.col('addr_indic_r').cast(pl.Int8),
        p_rel=pl.col('p_rr') / pl.col('p_rr').max().over('i1'), n_cand=pl.len().over('i1'),
        n_raw_eq=(pl.col('name').str.to_lowercase() == pl.col('name_r').str.to_lowercase()).cast(pl.Int8),
        a_raw_eq=(pl.col('addr').str.to_lowercase() == pl.col('addr_r').str.to_lowercase()).fill_null(False).cast(pl.Int8),
        # multiplicity: counts on each side and the S1 / record ratio (how many other businesses share the evidence)
        m_s1_sq=pl.col('c1_sq'), m_r_sq=pl.col('cr_sq'), m_s1_sk=pl.col('c1_sk'), m_r_sk=pl.col('cr_sk'),
        m_s1_addr=pl.col('c1_addr'), m_r_addr=pl.col('cr_addr'),
        mr_s1_sq=pl.col('c1_sq_r'), mr_r_sq=pl.col('cr_sq_r'), mr_s1_addr=pl.col('c1_addr_r'), mr_r_addr=pl.col('cr_addr_r'),
    ).with_columns(n_len_diff=(pl.col('n_len1') - pl.col('n_len2')).abs())
    return d.select(['i1', 'ir'] + FEATS)


def augment_missing(f: pl.DataFrame) -> pl.DataFrame:
    """Blank ZIP/state evidence for a deterministic 15 % of S1 (France-like records)."""
    aug = (pl.col('i1').cast(pl.UInt64).hash(seed=11) % 1000) < int(AUG_FRAC * 1000)
    return f.with_columns(
        pl.when(aug).then(0.0).otherwise(pl.col('b_z')).alias('b_z'),
        pl.when(aug).then(-1).otherwise(pl.col('zip_eq')).cast(pl.Float32).alias('zip_eq'),
        pl.when(aug).then(1).otherwise(pl.col('zip_miss')).cast(pl.Int8).alias('zip_miss'),
        pl.when(aug).then(-1).otherwise(pl.col('state_eq')).cast(pl.Int32).alias('state_eq'),
        pl.when(aug).then(1).otherwise(pl.col('state_miss')).cast(pl.Int8).alias('state_miss'),
        aug.cast(pl.Int8).alias('aug'))


def build(split, chunk_s1=200_000, prev='f2'):
    """Full features for the stage-2 candidates. From f3 on, the string features of pairs already scored in `prev` are
    reused (they depend only on the two records); only new pairs are computed, and the stage-2 columns, p_rel and n_cand
    are taken from / recomputed on the new candidate set."""
    p = feat_path(split)
    if p.exists():
        log(p.name, 'cached'); return
    t = time.time(); tf = tfidf(); s1, r = side_full(split)
    c2 = pl.read_parquet(stage2_path(split))
    old_p = FEAT_DIR / f'{split}_{prev}.parquet'
    reuse = [c for c in FULL_FEATS + MULT_FEATS if c not in ('p_rel', 'n_cand')]
    if FEAT_VERSION >= 'f3' and old_p.exists():
        old = pl.read_parquet(old_p, columns=['i1', 'ir'] + reuse)
        have = c2.join(old, on=['i1', 'ir'], how='inner').with_columns(p_rel=pl.lit(0.0), n_cand=pl.lit(0, dtype=pl.UInt32))
        c2 = c2.join(old.select('i1', 'ir'), on=['i1', 'ir'], how='anti'); del old
        log(f'{split}: reusing {have.height:,} pairs from {prev}, computing {c2.height:,} new pairs')
        outs = [have.select(['i1', 'ir'] + FEATS)]
    else:
        outs = []
    u = c2['i1'].unique().sort().to_numpy()
    for k in range(0, len(u), chunk_s1):
        outs.append(full_features(c2.join(pl.DataFrame({'i1': u[k:k + chunk_s1]}), on='i1', how='semi'), s1, r, tf)); gc.collect()
    f = pl.concat(outs, how='vertical_relaxed').sort(['i1', 'ir'])
    f = f.with_columns(p_rel=pl.col('p_rr') / pl.col('p_rr').max().over('i1'), n_cand=pl.len().over('i1'))
    if split == 'train':
        f = augment_missing(f)
        f = f.join(truth().with_columns(pl.lit(1, dtype=pl.Int8).alias('y')), on=['i1', 'ir'], how='left').with_columns(pl.col('y').fill_null(0))
    else:
        f = f.with_columns(aug=pl.lit(0, dtype=pl.Int8))
    f.write_parquet(p)
    log(f'{p.name}: {f.shape} in {(time.time() - t) / 60:.1f} min')


def run(split=None):
    for s in ([split] if split else ['train', 'test']):
        build(s)


if __name__ == '__main__':
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else None)
