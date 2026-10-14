"""Step 6: fine-tuned multilingual cross-encoder (CE) over candidate pairs.

Input: "name | address" of both records in their raw script (the model aligns transliterations, digits and
abbreviations with full cross-attention). Single-logit head, BCE, one epoch over the stage-2 candidates of the
`ce_train` S1 subset (hard negatives by construction). bf16, SDPA attention, length-bucketed batches, batched Rust
tokenisation in a prefetch thread. Checkpoints (model + optimiser + scheduler + step) every CKPT_MIN minutes with
automatic resume; scoring is chunked and cached, so an interrupted run continues where it stopped.
"""
import os, math, time, json, gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from .config import OUT, SEED, log
from .data import nt
from .tok_fast import FastTok

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
CKPT_MIN = 10
MODELS = {'e5b': 'intfloat/multilingual-e5-base', 'e5s': 'intfloat/multilingual-e5-small', 'e5l': 'intfloat/multilingual-e5-large'}


def side_texts(split):
    fmt = pl.when(pl.col('addr').is_null()).then(pl.col('name')).otherwise(pl.concat_str([pl.col('name'), pl.lit(' | '), pl.col('addr')]))
    s1 = nt(split, '1', ['name', 'addr']).select(fmt.alias('t'))['t'].to_list()
    r = pl.concat([nt(split, s, ['name', 'addr']) for s in '23']).select(fmt.alias('t'))['t'].to_list()
    return s1, r


def _batches(order, bs):
    return [order[k:k + bs] for k in range(0, len(order), bs)]


def train_ce(tag, pairs: pl.DataFrame, lr=3e-5, micro_bs=128, max_len=128, epochs=1):
    """pairs: (i1, ir, y) of the train split."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    ce_dir = OUT / f'ce_{tag}'; ck_dir = ce_dir / 'checkpoint'; ck_dir.mkdir(parents=True, exist_ok=True); ck = ck_dir / 'last.pt'
    if (ce_dir / 'config.json').exists():
        log(f'CE {tag} cached'); return ce_dir
    name = MODELS[tag.split('_')[0]]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, attn_implementation='sdpa').cuda()
    s1t, rt = side_texts('train')
    pairs = pairs.sort(['i1', 'ir'])
    a_i, b_i, y = pairs['i1'].to_numpy(), pairs['ir'].to_numpy(), pairs['y'].to_numpy().astype(np.float32)
    # length-bucketed batches: sort by length within random mega-chunks, then shuffle the batch order
    rng = np.random.default_rng(SEED)
    lens = np.array([len(s1t[i]) + len(rt[j]) for i, j in zip(a_i, b_i)])
    batches = []
    for ep in range(epochs):
        perm = rng.permutation(len(y))
        for c in range(0, len(perm), micro_bs * 100):
            chunk = perm[c:c + micro_bs * 100]; chunk = chunk[np.argsort(lens[chunk], kind='stable')]
            batches += _batches(chunk, micro_bs)
    order = rng.permutation(len(batches)); batches = [batches[i] for i in order]
    n_steps = len(batches)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)
    start, hist, prev = 0, [], 0.0
    if ck.exists():
        st = torch.load(ck, map_location='cuda', weights_only=False)
        if st['n_steps'] == n_steps:
            model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sch.load_state_dict(st['sch'])
            start, hist, prev = st['step'], st['hist'], st['elapsed_min']
            log(f'CE {tag}: resuming at step {start}/{n_steps}')
    ft = FastTok(tok, max_len)
    enc = lambda idx: ft([s1t[a_i[i]] for i in idx], [rt[b_i[i]] for i in idx])
    model.train(); t = time.time(); last_ck = time.time(); run_loss = []
    def save(step):
        tmp = ck.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'step': step, 'n_steps': n_steps,
                    'hist': hist, 'elapsed_min': prev + (time.time() - t) / 60}, tmp)
        tmp.replace(ck)
    with ThreadPoolExecutor(2) as ex:
        fut = ex.submit(enc, batches[start]) if start < n_steps else None
        for step in range(start, n_steps):
            b = fut.result()
            if step + 1 < n_steps:
                fut = ex.submit(enc, batches[step + 1])
            b = {k: v.cuda(non_blocking=True) for k, v in b.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = model(**b).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[batches[step]]).cuda())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); run_loss.append(loss.item())
            if step % 200 == 0 or step == n_steps - 1:
                el = time.time() - t; done = step + 1 - start; hist.append((step, float(np.mean(run_loss[-200:]))))
                log(f'CE {tag} step {step}/{n_steps} loss {hist[-1][1]:.4f} {done * micro_bs / el:,.0f} pairs/s '
                    f'eta {(n_steps - step - 1) * el / done / 60:.1f} min')
            if time.time() - last_ck > CKPT_MIN * 60:
                save(step + 1); last_ck = time.time()
    save(n_steps)
    model.save_pretrained(ce_dir); tok.save_pretrained(ce_dir)
    json.dump({'hist': hist, 'pairs': len(y), 'minutes': prev + (time.time() - t) / 60, 'base': name, 'lr': lr, 'max_len': max_len},
              open(ce_dir / 'train_log.json', 'w'))
    del model, opt; torch.cuda.empty_cache()
    return ce_dir


def score_ce(tag, split, pairs: pl.DataFrame, name, bs=1024, max_len=128, chunk=2_000_000):
    """CE logits for pairs (i1, ir) of a split -> DataFrame (i1, ir, ce_<tag>).

    The cache is keyed by PAIR: every parquet part under ce_scores/<split>_*_<tag>/ is read, only the pairs not scored
    yet are scored (written to ce_scores/<split>_<name>_<tag>/ with a run-unique part name), so a changed candidate set
    reuses all overlapping scores and can never be matched to stale rows by chunk index."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    root = OUT / 'ce_scores'; d = root / f'{split}_{name}_{tag}'; d.mkdir(parents=True, exist_ok=True)
    col = f'ce_{tag}'
    pairs = pairs.select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32)).unique().sort(['i1', 'ir'])
    def cached():
        parts = [p for dd in root.glob(f'{split}_*_{tag}') if dd.is_dir() for p in dd.glob('part_*.parquet')]
        if not parts:
            return pl.DataFrame(schema={'i1': pl.UInt32, 'ir': pl.UInt32, col: pl.Float32})
        return (pl.concat([pl.read_parquet(p).select(pl.col('i1').cast(pl.UInt32), pl.col('ir').cast(pl.UInt32), pl.col(col).cast(pl.Float32))
                           for p in parts]).unique(['i1', 'ir']))
    have = cached()
    todo = pairs.join(have.select('i1', 'ir'), on=['i1', 'ir'], how='anti')
    log(f'CE {tag} {split}/{name}: {pairs.height:,} pairs, {pairs.height - todo.height:,} cached, {todo.height:,} to score')
    if todo.height:
        ce_dir = OUT / f'ce_{tag}'
        tok = AutoTokenizer.from_pretrained(ce_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ce_dir, attn_implementation='sdpa', dtype=torch.bfloat16).cuda().eval()
        s1t, rt = side_texts(split)
        run_id = time.strftime('%H%M%S')
        n_chunks = math.ceil(todo.height / chunk)
        for k in range(n_chunks):
            t = time.time(); sub = todo.slice(k * chunk, chunk)
            a_i, b_i = sub['i1'].to_numpy(), sub['ir'].to_numpy()
            lens = np.array([len(s1t[i]) + len(rt[j]) for i, j in zip(a_i, b_i)]); order = np.argsort(lens, kind='stable')
            out = np.empty(len(order), np.float32)
            ft = FastTok(tok, max_len)
            enc = lambda idx: ft([s1t[a_i[i]] for i in idx], [rt[b_i[i]] for i in idx])
            bl = _batches(order, bs)
            with torch.no_grad(), ThreadPoolExecutor(3) as ex:
                for idx, b in zip(bl, ex.map(enc, bl)):
                    b = {kk: v.cuda(non_blocking=True) for kk, v in b.items()}
                    out[idx] = model(**b).logits.squeeze(-1).float().cpu().numpy()
            sub.with_columns(pl.Series(col, out)).write_parquet(d / f'part_{run_id}_{k:04d}.parquet')
            log(f'CE {tag} scored {split}/{name} chunk {k + 1}/{n_chunks}: {sub.height / (time.time() - t):,.0f} pairs/s')
        del model; torch.cuda.empty_cache()
        have = cached()
    return pairs.join(have, on=['i1', 'ir'], how='left')
