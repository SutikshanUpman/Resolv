"""Fast batched tokenisation through the Rust `tokenizers` backend.

The transformers 5 Python wrapper tokenises ~15k texts/s on this machine, while the raw Rust `encode_batch` does ~120k/s
(parallel), so the GPU would otherwise sit idle. Returns padded int64 tensors (input_ids, attention_mask).
"""
import numpy as np
import torch


class FastTok:
    def __init__(self, hf_tokenizer, max_len):
        self.tk = hf_tokenizer.backend_tokenizer if hasattr(hf_tokenizer, 'backend_tokenizer') else hf_tokenizer._tokenizer
        self.tk.enable_truncation(max_len)
        self.tk.no_padding()
        self.pad_id = hf_tokenizer.pad_token_id if hf_tokenizer.pad_token_id is not None else 0

    def __call__(self, a, b=None):
        encs = self.tk.encode_batch(list(a) if b is None else list(zip(a, b)))
        ids = [e.ids for e in encs]
        L = max(len(x) for x in ids)
        out = np.full((len(ids), L), self.pad_id, np.int64)
        am = np.zeros((len(ids), L), np.int64)
        for i, x in enumerate(ids):
            out[i, :len(x)] = x; am[i, :len(x)] = 1
        return {'input_ids': torch.from_numpy(out), 'attention_mask': torch.from_numpy(am)}
