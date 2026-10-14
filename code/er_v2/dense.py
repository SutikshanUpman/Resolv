"""Step 3b: dense candidate source, a fine-tuned multilingual bi-encoder + exact GPU top-k within country.

`intfloat/multilingual-e5-small` (MIT) is fine-tuned with a symmetric in-batch-negative InfoNCE loss on
(S1 text, matched S2/S3 text) pairs of the `bienc` S1 subset (disjoint from every evaluation subset). Batches are drawn
within one country, so the in-batch negatives are same-country businesses. Text = "name | address" in the raw
script, so the model aligns Devanagari/Bengali/Tamil/Kannada names with their English forms, which sparse keys miss.

Search: every S1 and S2/S3 record is embedded once (fp16, L2-normalised); for each country an exact
chunked matmul + top-k gives S1->R top-K_S1 and R->S1 top-K_R. The union of both directions is the dense candidate set.
"""
import os, math, time, gc
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from .config import OUT, SEED, DENSE_VERSION, log
from .data import nt, subset_i1, truth
from .tok_fast import FastTok

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
BE_NAME = 'intfloat/multilingual-e5-small'
BE_DIR = OUT / f'bienc_{DENSE_VERSION}'
EMB_DIR = OUT / f'emb_{DENSE_VERSION}'
MAX_LEN = 64
K_S1, K_R = 40, 3
TXT_COLS = ['name', 'addr']


def texts(split, s):
    d = nt(split, s, TXT_COLS)
    e = pl.when(pl.col('addr').is_null()).then(pl.col('name')).otherwise(pl.concat_str([pl.col('name'), pl.lit(' | '), pl.col('addr')]))
    return d.select(pl.concat_str([pl.lit('query: '), e]).alias('t'))['t'].to_list()


def mean_pool(out, mask):
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)


def train_bienc(max_pairs=1_000_000, bs=1024, lr=5e-5, temp=0.05):
    from transformers import AutoTokenizer, AutoModel
    if (BE_DIR / 'config.json').exists():
        log('bi-encoder cached'); return
    tok = AutoTokenizer.from_pretrained(BE_NAME)
    model = AutoModel.from_pretrained(BE_NAME).cuda()
    s1t, rt = texts('train', '1'), texts('train', '2') + texts('train', '3')
    cty = nt('train', '1', ['country']).with_row_index('i1').rename({'country': 'c'})
    tr = truth().join(pl.DataFrame({'i1': subset_i1('bienc')}), on='i1', how='semi')
    if tr.height > max_pairs:
        tr = tr.sample(max_pairs, seed=SEED)
    tr = tr.join(cty, on='i1')
    rng = np.random.default_rng(SEED)
    batches = []
    for c, g in tr.group_by('c'):
        a = g.select('i1', 'ir').to_numpy(); a = a[rng.permutation(len(a))]
        batches += [a[k:k + bs] for k in range(0, len(a) - bs + 1, bs)]
    batches = [batches[i] for i in rng.permutation(len(batches))]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    from transformers import get_linear_schedule_with_warmup
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * len(batches)), len(batches))
    model.train(); t = time.time()
    for step, b in enumerate(batches):
        # a record may appear once per batch only if its S1 is unique in the batch (duplicate S1 -> false negatives): mask them
        qa = tok([s1t[i] for i in b[:, 0]], max_length=MAX_LEN, truncation=True, padding=True, return_tensors='pt').to('cuda')
        qb = tok([rt[i] for i in b[:, 1]], max_length=MAX_LEN, truncation=True, padding=True, return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ea = F.normalize(mean_pool(model(**qa).last_hidden_state, qa['attention_mask']).float(), dim=-1)
            eb = F.normalize(mean_pool(model(**qb).last_hidden_state, qb['attention_mask']).float(), dim=-1)
        sim = ea @ eb.T / temp
        same = torch.from_numpy(b[:, 0][:, None] == b[:, 0][None, :]).cuda()
        sim = sim.masked_fill(same & ~torch.eye(len(b), dtype=torch.bool, device='cuda'), -1e4)
        lab = torch.arange(len(b), device='cuda')
        loss = (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab)) / 2
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            log(f'bienc step {step}/{len(batches)} loss {loss.item():.4f} {(step + 1) * bs / (time.time() - t):,.0f} pairs/s')
    model.save_pretrained(BE_DIR); tok.save_pretrained(BE_DIR)
    del model; torch.cuda.empty_cache()


@torch.no_grad()
def encode(model, tok, tx, bs=4096):
    order = np.argsort([len(x) for x in tx], kind='stable')
    out = np.empty((len(tx), model.config.hidden_size), np.float16)
    from concurrent.futures import ThreadPoolExecutor
    chunks = [order[k:k + bs] for k in range(0, len(tx), bs)]
    ft = FastTok(tok, MAX_LEN)
    tokf = lambda idx: ft([tx[i] for i in idx])
    with ThreadPoolExecutor(2) as ex:
        for idx, enc in zip(chunks, ex.map(tokf, chunks)):
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model(**enc).last_hidden_state
            out[idx] = F.normalize(mean_pool(h, enc['attention_mask']).float(), dim=-1).half().cpu().numpy()
    return out


def embed_all():
    from transformers import AutoTokenizer, AutoModel
    EMB_DIR.mkdir(exist_ok=True)
    tok = model = None
    for split in ['train', 'test']:
        for side, srcs in [('s1', '1'), ('r', '23')]:
            p = EMB_DIR / f'{split}_{side}.npy'
            if p.exists():
                continue
            if model is None:
                tok = AutoTokenizer.from_pretrained(BE_DIR)
                model = AutoModel.from_pretrained(BE_DIR, dtype=torch.bfloat16).cuda().eval()
            tx = sum([texts(split, s) for s in srcs], [])
            t = time.time(); e = encode(model, tok, tx); np.save(p, e)
            log(f'embedded {split}/{side}: {e.shape} in {(time.time() - t) / 60:.1f} min')
    if model is not None:
        del model; torch.cuda.empty_cache()


@torch.no_grad()
def topk_search(Q, D, k, q_chunk=2048, d_block=2_000_000):
    """Exact inner-product top-k of every row of Q against D (both fp16 torch tensors on GPU)."""
    vals, idxs = [], []
    for a in range(0, Q.shape[0], q_chunk):
        q = Q[a:a + q_chunk]
        bv = bi = None
        for b in range(0, D.shape[0], d_block):
            s = q @ D[b:b + d_block].T
            v, i = s.topk(min(k, s.shape[1]), dim=1)
            i = i + b
            if bv is None:
                bv, bi = v, i
            else:
                v2 = torch.cat([bv, v], 1); i2 = torch.cat([bi, i], 1)
                bv, j = v2.topk(k, dim=1); bi = i2.gather(1, j)
        vals.append(bv.float().cpu()); idxs.append(bi.cpu())
    return torch.cat(vals).numpy(), torch.cat(idxs).numpy()


def dense_path(split):
    return OUT / 'blocking' / f'{split}_dense_{DENSE_VERSION}_k{K_S1}_{K_R}.parquet'


def search_split(split):
    p = dense_path(split)
    if p.exists():
        log(p.name, 'cached'); return
    E1 = np.load(EMB_DIR / f'{split}_s1.npy'); ER = np.load(EMB_DIR / f'{split}_r.npy')
    c1 = nt(split, '1', ['country'])['country']
    cr = pl.concat([nt(split, s, ['country']) for s in '23'])['country']
    outs = []
    for c in c1.unique().sort().to_list():
        g1 = np.where((c1 == c).to_numpy())[0]; gr = np.where((cr == c).to_numpy())[0]
        Q = torch.from_numpy(E1[g1]).cuda(); D = torch.from_numpy(ER[gr]).cuda()
        t = time.time()
        v, i = topk_search(Q, D, K_S1)
        a = pl.DataFrame({'i1': np.repeat(g1, K_S1).astype(np.uint32), 'ir': gr[i.ravel()].astype(np.uint32),
                          'dcos': v.ravel().astype(np.float32), 'drank_s': np.tile(np.arange(1, K_S1 + 1), len(g1)).astype(np.uint16)})
        v2, i2 = topk_search(D, Q, K_R)
        b = pl.DataFrame({'i1': g1[i2.ravel()].astype(np.uint32), 'ir': np.repeat(gr, K_R).astype(np.uint32),
                          'dcos_r': v2.ravel().astype(np.float32), 'drank_r': np.tile(np.arange(1, K_R + 1), len(gr)).astype(np.uint16)})
        m = a.join(b, on=['i1', 'ir'], how='full', coalesce=True)
        m = m.with_columns(dcos=pl.coalesce('dcos', 'dcos_r')).drop('dcos_r')
        outs.append(m)
        log(f'dense {split}/{c}: S1 {len(g1):,} R {len(gr):,} -> {m.height:,} pairs in {time.time() - t:.0f}s')
        del Q, D; torch.cuda.empty_cache()
    pl.concat(outs).write_parquet(p)


def run():
    train_bienc()
    embed_all()
    for split in ['train', 'test']:
        search_split(split)


if __name__ == '__main__':
    run()
