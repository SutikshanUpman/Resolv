"""Step 1: record normalisation (v2).

Every rule cancels a documented noise pattern so that true pairs become near-identical strings:
NFKC + anyascii transliteration, legal-form canonicalisation, honorific / junk / phone-number removal, domain stems,
street vocabulary (US, Indian and French abbreviations), state canonicalisation, de-zeroed house numbers,
digit-free locality tokens and a phonetic consonant skeleton of the name (survives Indic transliteration).
Country is an open set: US/India state dictionaries apply to those labels, every other label takes the generic path.
"""
import re, sys, gc, time, unicodedata
import polars as pl
from anyascii import anyascii
from .config import OUT, NORM_VERSION, RAW_FILES, raw_path, log

NORM_DIR = OUT / 'norm'
NORM_DIR.mkdir(parents=True, exist_ok=True)
INDIC = r'[\p{Devanagari}\p{Bengali}\p{Gujarati}\p{Gurmukhi}\p{Oriya}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}]'


def read_tsv(path):
    """All columns as strings; quote_char=None because a handful of names contain a literal '"'."""
    return pl.read_csv(path, separator='\t', quote_char=None, infer_schema=False)


def _to_ascii(s):
    return anyascii(unicodedata.normalize('NFKC', s))


def translit(series: pl.Series) -> pl.Series:
    """Transliterate only the non-ASCII strings (deduplicated) and map back vectorised."""
    uniq = series.filter(series.str.contains(r'[^\x00-\x7F]').fill_null(False)).unique()
    if uniq.len() == 0:
        return series
    return series.replace(uniq, pl.Series([_to_ascii(x) for x in uniq.to_list()], dtype=pl.String))


# ---------------------------------------------------------------- dictionaries
LEGAL_CANON = {
    'private': 'pvt', 'pvt': 'pvt', 'pte': 'pvt', 'pvtltd': 'pvt', 'limited': 'ltd', 'ltd': 'ltd', 'ltda': 'ltd',
    'incorporated': 'inc', 'inc': 'inc', 'corporation': 'corp', 'corp': 'corp', 'corpn': 'corp',
    'company': 'co', 'co': 'co', 'cie': 'co', 'llc': 'llc', 'llp': 'llp', 'lp': 'lp', 'plc': 'plc',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sas', 'sa': 'sa', 'eurl': 'eurl', 'sci': 'sci', 'snc': 'snc',
    'selarl': 'selarl', 'scop': 'scop', 'scp': 'scp', 'gie': 'gie', 'earl': 'earl', 'gaec': 'gaec', 'sccv': 'sccv',
    'gmbh': 'gmbh', 'pty': 'pty', 'opc': 'opc', 'pc': 'pc', 'pllc': 'pllc', 'ltee': 'ltd',
    # transliterated Indic spellings of the same legal words
    'praivet': 'pvt', 'praaivet': 'pvt', 'praivett': 'pvt', 'prayvet': 'pvt', 'praibhet': 'pvt', 'praivhet': 'pvt',
    'limitedd': 'ltd', 'limitad': 'ltd', 'limited.': 'ltd', 'limitedd.': 'ltd', 'limitett': 'ltd', 'limitted': 'ltd',
    'elelpi': 'llp', 'elelpii': 'llp', 'kampani': 'co', 'kompani': 'co', 'inka': 'inc',
}
TOKEN_CANON = {**LEGAL_CANON, 'ets': 'etablissements', 'etabl': 'etablissements', 'et': 'and'}
LEGAL_VALS = sorted(set(LEGAL_CANON.values()))
HONORIFICS = ['smt', 'sri', 'shri', 'shree', 'sh', 'mr', 'mrs', 'ms', 'dr', 'kumari', 'messrs', 'the']
DROP_NAME = LEGAL_VALS + HONORIFICS

NAME_PRE = [  # applied in order to the lower-cased ASCII name
    (r'\+?\d[\d\s\-]{7,}\d', ' '),                       # phone numbers / long digit runs inside names
    (r'\bm\s*/\s*s\b\.?', ' '),
    (r'\bl\.\s*l\.\s*c\b\.?', ' llc '), (r'\bl\.\s*l\.\s*p\b\.?', ' llp '),
    (r'\bs\.\s*a\.\s*r\.\s*l\b\.?', ' sarl '), (r'\be\.\s*u\.\s*r\.\s*l\b\.?', ' eurl '),
    (r'\bs\.\s*a\.\s*s\.\s*u\b\.?', ' sasu '), (r'\bs\.\s*a\.\s*s\b\.?', ' sas '),
    (r'\bs\.\s*n\.\s*c\b\.?', ' snc '), (r'\bs\.\s*c\.\s*i\b\.?', ' sci '), (r'\bs\.\s*a\.(\s|$)', ' sa '),
    (r'\bp\.?\s*ltd\b', ' pvt ltd '), (r'&', ' and '), (r'\+', ' and '),
]
DOMAIN_RX = r'\b(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:co\.in|com|in|net|org|co|fr|biz|info|us)\b'

US_STATE = {'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca', 'colorado': 'co', 'connecticut': 'ct',
    'delaware': 'de', 'district of columbia': 'dc', 'washington dc': 'dc', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne',
    'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri',
    'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va',
    'washington': 'wa', 'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy', 'puerto rico': 'pr'}
IN_STATE = {'andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br', 'chhattisgarh': 'cg', 'chattisgarh': 'cg', 'ct': 'cg',
    'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr', 'himachal pradesh': 'hp', 'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl',
    'madhya pradesh': 'mp', 'madhyapradesh': 'mp', 'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml', 'mizoram': 'mz', 'nagaland': 'nl',
    'odisha': 'od', 'orissa': 'od', 'or': 'od', 'punjab': 'pb', 'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn', 'tamilnadu': 'tn',
    'telangana': 'ts', 'tg': 'ts', 'tripura': 'tr', 'uttar pradesh': 'up', 'uttarakhand': 'uk', 'uttaranchal': 'uk', 'ut': 'uk',
    'west bengal': 'wb', 'delhi': 'dl', 'nct of delhi': 'dl', 'jammu and kashmir': 'jk', 'ladakh': 'la', 'puducherry': 'py',
    'pondicherry': 'py', 'chandigarh': 'ch', 'andaman and nicobar islands': 'an', 'lakshadweep': 'ld',
    'dadra and nagar haveli': 'dn', 'daman and diu': 'dd'}
STATE_CODES = sorted(set(US_STATE.values()) | set(IN_STATE.values()))

ADDR_MAP = {
    'street': 'st', 'str': 'st', 'strt': 'st', 'saint': 'st', 'avenue': 'ave', 'av': 'ave', 'avn': 'ave', 'avenu': 'ave',
    'road': 'rd', 'drive': 'dr', 'drv': 'dr', 'lane': 'ln', 'boulevard': 'blvd', 'bd': 'blvd', 'boul': 'blvd', 'bvd': 'blvd', 'bld': 'blvd',
    'court': 'ct', 'place': 'pl', 'highway': 'hwy', 'parkway': 'pkwy', 'circle': 'cir', 'terrace': 'ter', 'terr': 'ter',
    'square': 'sq', 'apartment': 'apt', 'apts': 'apt', 'suite': 'ste', 'floor': 'fl', 'flr': 'fl', 'building': 'bldg',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w', 'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    'mount': 'mt', 'fort': 'ft', 'sector': 'sec', 'extension': 'extn', 'ext': 'extn', 'colony': 'col', 'district': 'dist', 'dt': 'dist',
    'village': 'vill', 'vil': 'vill', 'tehsil': 'teh', 'tahsil': 'teh', 'taluk': 'tq', 'taluka': 'tq', 'tk': 'tq',
    'opposite': 'opp', 'near': 'nr', 'post': 'po', 'nagar': 'ngr', 'township': 'twp', 'townshiip': 'twp',
    # French
    'rue': 'rue', 'r': 'rue', 'chemin': 'chem', 'che': 'chem', 'ch': 'chem', 'impasse': 'imp', 'allee': 'all', 'allees': 'all',
    'route': 'rte', 'faubourg': 'fbg', 'residence': 'res', 'batiment': 'bat', 'quai': 'qu', 'q': 'qu', 'cours': 'crs',
    'cite': 'cite', 'lieu': 'ld', 'lieudit': 'ld', 'hameau': 'ham', 'sainte': 'st'}
ADDR_FILL = ['no', 'nos', 'number', 'num', 'hno', 'dno', 'h', 'd', 'door', '', 'bis', 'ter', 'ndeg', 'deg']
STOP_LOC = set(ADDR_MAP.values()) | set(ADDR_FILL) | {'and', 'the', 'of', 'de', 'la', 'le', 'du', 'des', 'les', 'en', 'sur', 'cdp',
                                                      'near', 'behind', 'road', 'main', 'cross', 'block', 'plot', 'flat', 'house'}
POBOX_RX = r'\bp\.?\s*o\.?\s*box\s*\d*'


# ---------------------------------------------------------------- phonetic skeleton
_SK_RULES = [(re.compile(p), r) for p, r in [
    (r'[^a-z ]', ''), (r'tion', 'shn'), (r'sion', 'shn'), (r'ph', 'f'), (r'gh', 'g'), (r'kh', 'k'), (r'bh', 'b'), (r'dh', 'd'),
    (r'th', 't'), (r'sh', 's'), (r'ch', 'k'), (r'ck', 'k'), (r'x', 'ks'), (r'q', 'k'), (r'c(?=[eiy])', 's'), (r'c', 'k'), (r'z', 's'),
    (r'g(?=[eiy])', 's'), (r'j', 's'), (r'm(?=[^aeiouy ])', 'n')]]


def _skel_token(t):
    if not t:
        return ''
    first = 'v' if t[0] == 'w' else t[0]
    rest = re.sub(r'[aeiouywhv]', '', t[1:])
    s = first + rest
    return re.sub(r'(.)\1+', r'\1', s)


def skeleton(name_core: str) -> str:
    """Consonant skeleton, e.g. 'vijn sistms' (from विजन सिस्टम्स) and 'vision systems' both -> 'stns vsn'."""
    s = name_core
    for rx, rep in _SK_RULES:
        s = rx.sub(rep, s)
    toks = sorted(t for t in (_skel_token(x) for x in s.split()) if t)
    return ' '.join(toks)


def _tok_pipeline(e: pl.Expr) -> pl.Expr:
    """lower-cased address text -> list of canonical tokens (ordinals removed, digits split from letters, de-zeroed)."""
    return (e.str.replace_all(r'ndeg', ' ').str.replace_all(r'\b(\d+)(st|nd|rd|th)\b', '${1}')
             .str.replace_all(r'(\d)([a-z])', '${1} ${2}')
             .str.replace_all(r'[^a-z0-9]+', ' ').str.strip_chars().str.split(' ')
             .list.eval(pl.element().str.replace(r'^0+(\d)', '${1}').replace(ADDR_MAP)
                        .filter(~pl.element().is_in(ADDR_FILL))))


def normalize_table(df: pl.DataFrame, src: int) -> pl.DataFrame:
    """Raw source table -> normalised record table (one row per entity_id; raw fields kept for the transformers)."""
    df = df.rename({'business_name': 'name', 'business_address': 'addr'})
    df = df.with_columns(pl.col('name').fill_null('').str.strip_chars(),
                         pl.when(pl.col('addr').str.strip_chars() == '').then(None).otherwise(pl.col('addr').str.strip_chars()).alias('addr'),
                         pl.col('country').fill_null('UNKNOWN'))
    df = df.with_columns(name_a=translit(df['name']), addr_a=translit(df['addr']))

    # ---------- name ----------
    e = pl.col('name_a').str.to_lowercase()
    for pat, rep in NAME_PRE:
        e = e.str.replace_all(pat, rep)
    is_dom = e.str.contains(DOMAIN_RX)
    e = e.str.replace_all(DOMAIN_RX, ' ${1} ').str.replace_all(r'[^a-z0-9]+', ' ').str.strip_chars()
    ntok = e.str.split(' ').list.eval(pl.element().filter(pl.element() != '').replace(TOKEN_CANON))
    df = df.with_columns(is_domain=is_dom, name_indic=pl.col('name').str.contains(INDIC), _nt=ntok)
    df = df.with_columns(
        name_norm=pl.col('_nt').list.join(' '),
        legal=pl.col('_nt').list.eval(pl.element().filter(pl.element().is_in(LEGAL_VALS))).list.unique().list.sort().list.join(' '),
        _ct=pl.col('_nt').list.eval(pl.element().filter(~pl.element().is_in(DROP_NAME))))
    df = df.with_columns(name_core=pl.when(pl.col('_ct').list.len() > 0).then(pl.col('_ct').list.join(' ')).otherwise(pl.col('name_norm')))
    df = df.with_columns(name_sq=pl.col('name_core').str.replace_all(' ', ''))
    uq = df['name_core'].unique()
    df = df.with_columns(name_sk=pl.col('name_core').replace(uq, pl.Series([skeleton(x) for x in uq.to_list()], dtype=pl.String)))

    # ---------- address ----------
    a = pl.col('addr_a').str.to_lowercase().str.replace_all('&', ' and ').str.replace_all(POBOX_RX, ' ').str.replace_all('#', ' ')
    comps = a.str.split(',').list.eval(pl.element().str.replace_all(r'[.\s]+$', '').str.replace_all(r'\s+', ' ').str.strip_chars())
    comps = (pl.when(pl.col('country') == 'US').then(comps.list.eval(pl.element().replace(US_STATE)))
               .when(pl.col('country') == 'India').then(comps.list.eval(pl.element().replace(IN_STATE)))
               .otherwise(comps))
    df = df.with_columns(_a=a, _comps=comps)
    atok = _tok_pipeline(pl.col('_comps').list.join(' '))
    loc = (_tok_pipeline(pl.col('_comps').list.eval(pl.element().filter(~pl.element().str.contains(r'\d'))).list.join(' '))
           .list.eval(pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().is_in(list(STOP_LOC))
                                          & ~pl.element().str.contains(r'\d')))
           .list.unique(maintain_order=True).list.head(6))
    df = df.with_columns(
        addr_norm=pl.when(pl.col('addr').is_null()).then(None).otherwise(atok.list.join(' ')),
        loc=pl.when(pl.col('addr').is_null()).then(None).otherwise(loc),
        state=pl.col('_comps').list.eval(pl.element().filter(pl.element().is_in(STATE_CODES))).list.last(),
        addr_nums=pl.col('_a').str.replace_all(r'\b\d{8,}\b', ' ').str.extract_all(r'\d+')
                  .list.eval(pl.element().str.replace(r'^0+(\d)', '${1}')).list.unique(maintain_order=True),
        zip=pl.col('_a').str.extract_all(r'\b\d{5,6}\b').list.last(),
        has_addr=pl.col('addr').is_not_null(),
        addr_indic=pl.col('addr').str.contains(INDIC).fill_null(False),
        src=pl.lit(src, dtype=pl.Int8))
    return df.select('entity_id', 'src', 'country', 'name', 'addr', 'name_norm', 'name_core', 'name_sq', 'name_sk', 'legal',
                     'is_domain', 'name_indic', 'addr_norm', 'loc', 'addr_nums', 'zip', 'state', 'has_addr', 'addr_indic')


def norm_path(split, s):
    """split in {'train_full', 'test'}; the ghosted train S1 table is written by splits.py as 'train'."""
    return NORM_DIR / f'{split}_s{s}_{NORM_VERSION}.parquet'


def run():
    for split in ['train', 'test']:
        for s in '123':
            tag = 'train_full' if (split == 'train' and s == '1') else split
            p = norm_path(tag, s)
            n_exp = RAW_FILES[f'{split}_source{s}']
            if p.exists() and pl.scan_parquet(p).select(pl.len()).collect().item() == n_exp:
                log(p.name, 'cached'); continue
            t = time.time()
            raw = read_tsv(raw_path(f'{split}_source{s}'))
            assert raw.height == n_exp, (split, s, raw.height)
            nd = normalize_table(raw, int(s)); del raw
            tmp = p.with_suffix('.tmp'); nd.write_parquet(tmp); tmp.rename(p)
            log(f'{p.name}: {nd.shape} in {time.time() - t:.0f}s')
            del nd; gc.collect()


if __name__ == '__main__':
    run()
