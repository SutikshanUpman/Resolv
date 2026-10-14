"""Test 3: Qwen3-Reranker-0.6B (Apache-2.0, 0.6B params) fine-tuned as a second cross-encoder, used on uncertain pairs.

Why: e5-base cannot resolve acronyms ("FBP" = "Fun Boutique Primaire SAS"), French abbreviations / region names or
coined DBA names well, and those dominate the test-only (France) loss. A multilingual LLM reranker brings that
pretrained knowledge; the address-less same-name cases stay unresolvable (no signal in the text).
Model: Qwen3ForCausalLM; score = logit('yes') - logit('no') at the last position of the official reranker prompt
(left padding; only the last hidden state goes through the LM head). Fine-tuned 1 epoch with BCE on hard stage-2
candidates of `ce_train` S1 that are NOT in the evaluation set (so eval / test scores are out-of-sample).
Scored only where the Test-1 stacker is uncertain (UNC_LO <= p <= UNC_HI); elsewhere the feature is missing.

    python -m er_v2.qwen_ce train      # ~20-40 min on the A100
    python -m er_v2.qwen_ce score      # eval + test uncertain pairs -> OUT/qwen_scores_{train,test}.parquet
"""
import os, sys, time, math, json
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
os.environ.setdefault('RAYON_NUM_THREADS', '3')
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
torch.set_num_threads(3)
from .config import OUT, SEED, log, save_json
from .data import nt, splits, truth
from .stage2 import stage2_path

BASE = 'Qwen/Qwen3-Reranker-0.6B'
QDIR = OUT / 'qwen_rr06'
N_TRAIN = int(os.environ.get('ER_QW_N', 400_000))
MAX_CONTENT = 96            # tokens for instruction + both records
BS, LR = 64, 2e-5
UNC_LO, UNC_HI = 0.05, 0.95
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCT = 'Is the Document the same real-world business as the Query?'


def texts(split):
    fmt = lambda: pl.concat_str([pl.col('name').fill_null(''), pl.lit(' | '), pl.col('addr').fill_null('no address'),
                                 pl.lit(' | '), pl.col('country').fill_null('')])
    s1 = nt(split, '1', ['name', 'addr', 'country']).select(fmt().alias('t'))['t'].to_list()
    r = pl.concat([nt(split, s, ['name', 'addr', 'country']) for s in '23']).select(fmt().alias('t'))['t'].to_list()
    return s1, r


class Enc:
    """Token ids: PREFIX + truncated '<Instruct>..<Query>..<Document>..' + SUFFIX, left-padded."""
    def __init__(self, tok):
        self.tok = tok
        self.pre = tok(PREFIX, add_special_tokens=False)['input_ids']
        self.suf = tok(SUFFIX, add_special_tokens=False)['input_ids']
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, qs, ds):
        mids = self.tok([f'<Instruct>: {INSTRUCT}\n<Query>: {q}\n<Document>: {d}' for q, d in zip(qs, ds)],
                        add_special_tokens=False, truncation=True, max_length=MAX_CONTENT)['input_ids']
        seqs = [self.pre + m + self.suf for m in mids]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s); att[i, L - len(s):] = 1
        return ids, att


def load(path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    yes, no = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    return tok, model, yes, no


def score_batch(model, ids, att, yes, no):
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1, :]
    W = model.lm_head.weight[[yes, no]]
    lg = (h @ W.T).float()
    return lg[:, 0] - lg[:, 1]


def train_pairs(n=None, seed=SEED):
    n = n or N_TRAIN
    sp = splits()
    ev_s1 = pl.read_parquet(OUT / 'eval_p_f3_e5b.parquet', columns=['i1'])['i1'].unique()
    ce = sp.filter(pl.col('split') == 'ce_train').select('i1').join(pl.DataFrame({'i1': ev_s1}), on='i1', how='anti')
    c = pl.read_parquet(stage2_path('train'), columns=['i1', 'ir', 'p_rr']).join(ce, on='i1', how='semi')
    c = c.join(truth().with_columns(pl.lit(1, dtype=pl.Int8).alias('y')), on=['i1', 'ir'], how='left').with_columns(pl.col('y').fill_null(0))
    hard = c.filter(pl.col('p_rr').is_between(0.02, 0.98)); easy = c.filter(~pl.col('p_rr').is_between(0.02, 0.98))
    n_h = min(hard.height, int(n * 0.7))
    d = pl.concat([hard.sample(n_h, seed=seed), easy.sample(n - n_h, seed=seed)]).sample(fraction=1.0, shuffle=True, seed=seed)
    log(f'Qwen train pairs {d.height:,} (hard {n_h:,}), pos rate {d["y"].mean():.3f}, from {ce.height:,} ce_train S1 outside the eval set')
    return d


def train():
    if (QDIR / 'config.json').exists():
        log('Qwen reranker cached'); return
    from transformers import get_linear_schedule_with_warmup
    tok, model, yes, no = load(BASE)
    model.float()   # fp32 master weights, bf16 autocast
    d = train_pairs(); s1t, rt = texts('train')
    a_i, b_i, y = d['i1'].to_numpy(), d['ir'].to_numpy(), d['y'].to_numpy().astype(np.float32)
    enc = Enc(tok); n_steps = math.ceil(len(y) / BS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * n_steps), n_steps)
    model.train(); model.config.use_cache = False; t = time.time(); run = []
    from concurrent.futures import ThreadPoolExecutor
    mk = lambda k: enc([s1t[i] for i in a_i[k * BS:(k + 1) * BS]], [rt[j] for j in b_i[k * BS:(k + 1) * BS]])
    with ThreadPoolExecutor(2) as ex:
        fut = ex.submit(mk, 0)
        for k in range(n_steps):
            ids, att = fut.result()
            if k + 1 < n_steps:
                fut = ex.submit(mk, k + 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                s = score_batch(model, ids.cuda(non_blocking=True), att.cuda(non_blocking=True), yes, no)
            loss = F.binary_cross_entropy_with_logits(s, torch.from_numpy(y[k * BS:(k + 1) * BS]).cuda())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); run.append(loss.item())
            if k % 200 == 0 or k == n_steps - 1:
                el = time.time() - t
                log(f'Qwen step {k}/{n_steps} loss {np.mean(run[-200:]):.4f} {(k + 1) * BS / el:,.0f} pairs/s eta {(n_steps - k - 1) * el / (k + 1) / 60:.1f} min')
    model.to(torch.bfloat16).save_pretrained(QDIR); tok.save_pretrained(QDIR)
    save_json({'pairs': len(y), 'minutes': (time.time() - t) / 60, 'final_loss': float(np.mean(run[-200:]))}, QDIR / 'train_log.json')
    log(f'Qwen fine-tuned in {(time.time() - t) / 60:.1f} min')


# ------------------------------------------------------------------ final pass (Stage A / Stage B)
# Stage A: score a wider uncertainty band with the Test-3 model, reusing its existing scores.
# Stage B: continue fine-tuning the Test-3 model on training pairs with generic, training-only noise (street-word
#          abbreviations, case / accents, number prefixes, dropped / reordered address components, acronyms), then score.
BAND = tuple(float(x) for x in os.environ.get('ER_QW_BAND', f'{UNC_LO},{UNC_HI}').split(','))
MODEL_DIR = OUT / os.environ.get('ER_QW_MODEL', 'qwen_rr06')
OUT_TAG = os.environ.get('ER_QW_OUT', '')            # qwen_scores_{split}{OUT_TAG}.parquet
REUSE_TAG = os.environ.get('ER_QW_REUSE', None)      # reuse scores of the SAME model from qwen_scores_{split}{REUSE_TAG}.parquet


@torch.no_grad()
def score():
    tok, model, yes, no = load(MODEL_DIR); model.eval(); enc = Enc(tok)
    for split, f in [('train', 'eval_p_f3_e5b'), ('test', 'test_p_f3_e5b')]:
        outp = OUT / f'qwen_scores_{split}{OUT_TAG}.parquet'
        if outp.exists():
            log(outp.name, 'cached'); continue
        d = pl.read_parquet(OUT / f'{f}.parquet', columns=['i1', 'ir', 'p']).filter(pl.col('p').is_between(*BAND)).select('i1', 'ir')
        old = None
        if REUSE_TAG is not None and (OUT / f'qwen_scores_{split}{REUSE_TAG}.parquet').exists():
            old = pl.read_parquet(OUT / f'qwen_scores_{split}{REUSE_TAG}.parquet')
            d = d.join(old.select('i1', 'ir'), on=['i1', 'ir'], how='anti')
        s1t, rt = texts(split)
        a_i, b_i = d['i1'].to_numpy(), d['ir'].to_numpy()
        order = np.argsort([len(s1t[i]) + len(rt[j]) for i, j in zip(a_i, b_i)], kind='stable')
        out = np.empty(len(order), np.float32); t = time.time(); bs = 256
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            ids, att = enc([s1t[a_i[i]] for i in idx], [rt[b_i[i]] for i in idx])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out[idx] = score_batch(model, ids.cuda(), att.cuda(), yes, no).cpu().numpy()
            if (k // bs) % 500 == 0:
                log(f'Qwen[{MODEL_DIR.name}] {split}: {k:,}/{len(order):,} pairs, {k / max(time.time() - t, 1e-6):,.0f} pairs/s')
        new = d.with_columns(qw=pl.Series(out, dtype=pl.Float32))
        res = new if old is None else pl.concat([old.select('i1', 'ir', pl.col('qw').cast(pl.Float32)), new])
        res.write_parquet(outp)
        log(f'Qwen[{MODEL_DIR.name}] {split}: scored {len(out):,} new pairs in {(time.time() - t) / 60:.1f} min -> {outp.name} ({res.height:,} pairs)')


# ---------- Stage B augmentation (training data only; the same rules for every record, no country logic)
def _street_tables():
    from .normalize import ADDR_MAP
    long2short = {k: v for k, v in ADDR_MAP.items() if len(k) > len(v) and k.isalpha() and v.isalpha()}
    short2long = {}
    for k, v in long2short.items():
        short2long.setdefault(v, k)
    return long2short, short2long


def _augment_addr(a, rng, l2s, s2l):
    import re, unicodedata
    if not a:
        return a
    def word(m):
        w = m.group(0); lw = w.lower()
        if lw in l2s and rng.random() < 0.5:
            r = l2s[lw]
            r = r + '.' if rng.random() < 0.5 else r
        elif lw in s2l and rng.random() < 0.3:
            r = s2l[lw]
        else:
            return w
        return r.upper() if w.isupper() else (r.capitalize() if w[:1].isupper() else r)
    a = re.sub(r'[A-Za-z]+', word, a)
    comps = [c.strip() for c in a.split(',')]
    u = rng.random()
    if len(comps) > 2 and u < 0.15:
        comps = comps[:-1]                                   # drop the coarse area (last component)
    elif len(comps) > 1 and u < 0.30:
        comps = comps[1:] + comps[:1]                        # component reordering
    a = ', '.join(comps)
    if rng.random() < 0.2:
        pre = 'N° ' if rng.random() < 0.5 else '#'
        a = re.sub(r'(?<![\w#])(\d+)', lambda m: pre + m.group(1), a, count=1)
    if rng.random() < 0.3:
        a = unicodedata.normalize('NFKD', a).encode('ascii', 'ignore').decode()
    if rng.random() < 0.3:
        a = a.upper()
    return a


def aug_pairs(n=250_000, seed=SEED + 11):
    """Test-3 training pool, new sample; half of the pairs get generic noise on the record (20 %: also on the S1)."""
    from .diag_recall import initials
    d = train_pairs(n, seed)                       # fresh sample (different seed from Test 3)
    s1 = nt('train', '1', ['name', 'addr', 'country', 'name_core']).with_row_index('i1').with_columns(ini=initials('name_core'))
    r = pl.concat([nt('train', s, ['name', 'addr', 'country', 'name_core']) for s in '23']).with_row_index('ir').with_columns(ini=initials('name_core'))
    d = (d.join(s1.select(pl.col('i1').cast(pl.UInt32), pl.col('name').alias('n1'), pl.col('addr').alias('a1'), pl.col('country').alias('c1'),
                          pl.col('ini').alias('ini1')), on='i1')
          .join(r.select(pl.col('ir').cast(pl.UInt32), pl.col('name').alias('n2'), pl.col('addr').alias('a2'), pl.col('country').alias('c2'),
                         pl.col('ini').alias('ini2')), on='ir'))
    rng = np.random.default_rng(seed); l2s, s2l = _street_tables()
    rows = d.select('n1', 'a1', 'c1', 'n2', 'a2', 'c2', 'ini1', 'ini2', 'y').rows()
    fmt = lambda nm, ad, c: f"{nm or ''} | {ad if ad else 'no address'} | {c or ''}"
    qs, ds, ys, n_acr = [], [], [], [0, 0]
    for n1, a1, c1, n2, a2, c2, ini1, ini2, y in rows:
        if rng.random() < 0.5:
            a2 = _augment_addr(a2, rng, l2s, s2l)
            if rng.random() < 0.2:
                a1 = _augment_addr(a1, rng, l2s, s2l)
            if rng.random() < 0.3 and n2:
                n2 = n2.upper()
            u = rng.random()
            if y == 1 and u < 0.05 and ini1 and 2 <= len(ini1) <= 5:
                n2 = ini1.upper(); n_acr[0] += 1                  # true acronym of the S1
            elif y == 0 and u < 0.16 and ini2 and 2 <= len(ini2) <= 5 and ini2 != ini1:   # ~balanced with the positives
                n2 = ini2.upper(); n_acr[1] += 1                  # acronym of a different business
        qs.append(fmt(n1, a1, c1)); ds.append(fmt(n2, a2, c2)); ys.append(y)
    log(f'Stage B pairs {len(ys):,} (pos {np.mean(ys):.3f}); acronym positives {n_acr[0]:,}, acronym negatives {n_acr[1]:,}')
    for i in range(6):
        log('  example:', qs[i], ' <-> ', ds[i], ' y=', ys[i])
    return qs, ds, np.array(ys, np.float32)


def train_aug(n=250_000, lr=1e-5):
    outd = OUT / 'qwen_rr06_aug'
    if (outd / 'config.json').exists():
        log('Stage B model cached'); return
    from transformers import get_linear_schedule_with_warmup
    tok, model, yes, no = load(QDIR)          # continue from the Test-3 model
    model.float()
    qs, ds, y = aug_pairs(n)
    enc = Enc(tok); n_steps = math.ceil(len(y) / BS)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * n_steps), n_steps)
    model.train(); model.config.use_cache = False; t = time.time(); run = []
    from concurrent.futures import ThreadPoolExecutor
    mk = lambda k: enc(qs[k * BS:(k + 1) * BS], ds[k * BS:(k + 1) * BS])
    with ThreadPoolExecutor(2) as ex:
        fut = ex.submit(mk, 0)
        for k in range(n_steps):
            ids, att = fut.result()
            if k + 1 < n_steps:
                fut = ex.submit(mk, k + 1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                s = score_batch(model, ids.cuda(non_blocking=True), att.cuda(non_blocking=True), yes, no)
            loss = F.binary_cross_entropy_with_logits(s, torch.from_numpy(y[k * BS:(k + 1) * BS]).cuda())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); run.append(loss.item())
            if k % 200 == 0 or k == n_steps - 1:
                el = time.time() - t
                log(f'Stage B step {k}/{n_steps} loss {np.mean(run[-200:]):.4f} {(k + 1) * BS / el:,.0f} pairs/s eta {(n_steps - k - 1) * el / (k + 1) / 60:.1f} min')
    model.to(torch.bfloat16).save_pretrained(outd); tok.save_pretrained(outd)
    save_json({'pairs': len(y), 'minutes': (time.time() - t) / 60, 'final_loss': float(np.mean(run[-200:])), 'init': str(QDIR), 'lr': lr},
              outd / 'train_log.json')
    log(f'Stage B fine-tuned in {(time.time() - t) / 60:.1f} min')


if __name__ == '__main__':
    {'train': train, 'score': score, 'train_aug': train_aug, 'aug_preview': lambda: aug_pairs(2000)}[sys.argv[1]]()
