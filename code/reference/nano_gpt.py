"""nanoGPT --- Chapter 11 reference implementation.

A self-contained character-level GPT in PyTorch. Every load-bearing component
of a modern decoder-only LLM is here, and each maps to a section of Chapter 11:

    RoPE                      Section 11.5.2  apply_rope / precompute_rope
    scaled dot-product attn   Section 11.2   CausalSelfAttention.forward
    causal masking            Section 11.7   is_causal=True
    multi-head wiring         Section 11.4   head reshape in CausalSelfAttention
    FlashAttention            Section 11.9.1  scaled_dot_product_attention
    pre-LayerNorm residual    Section 11.6   Block.forward
    FFN sublayer              Section 11.6   MLP
    weight tying              Section 11.6   GPT.__init__
    KV-caching                Section 11.10.1  CausalSelfAttention cache path

Usage:

    python3 nano_gpt.py                 # 2 layers, 64 dim, 100 steps, CPU
    python3 nano_gpt.py --big           # 4 layers, 4 heads, 192 dim, 2000 steps
    python3 nano_gpt.py --sample-only   # generate from an existing checkpoint

The default run is deliberately tiny: it finishes in well under a minute on a
laptop CPU and the loss should fall from ~4.17 (random init over 65 characters,
= log 65) to below 2.5 within 100 steps (about 2.3 on a laptop CPU). That drop is the whole point of
the smoke test --- it confirms the plumbing is connected. Producing prose takes
--big and a few minutes on a GPU.

The corpus is Tiny Shakespeare. If input.txt is not present the script fetches
it (urllib first, then curl, which works where Python lacks CA certificates, as
on many macOS installs). If both fail it says so and falls back to a small
embedded sample so the smoke test still runs; the Chapter 11 loss numbers do not
apply to that sample.
"""

import argparse
import math
import os
import shutil
import subprocess
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")
DATA_PATH = "input.txt"

FALLBACK = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.
""" * 200


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _download(url, path):
    """Fetch url -> path. Try urllib first (with certifi's CA bundle when it is
    installed), then fall back to curl, which uses the operating system's
    certificate store. Python builds on macOS often ship without CA
    certificates, so urllib alone fails there with CERTIFICATE_VERIFY_FAILED."""
    errors = []
    try:
        import ssl
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, context=ctx, timeout=30) as r, open(path, "wb") as fh:
            fh.write(r.read())
        return True
    except Exception as exc:
        errors.append(f"urllib: {exc}")
    if shutil.which("curl"):
        rc = subprocess.run(["curl", "-sSfL", "-o", path, url]).returncode
        if rc == 0:
            return True
        errors.append(f"curl: exit code {rc}")
    if os.path.exists(path):
        os.remove(path)                                # never keep a partial file
    print("  download failed:\n    " + "\n    ".join(errors))
    return False


def load_text():
    if not os.path.exists(DATA_PATH):
        print(f"downloading {DATA_URL} -> {DATA_PATH}")
        if not _download(DATA_URL, DATA_PATH):
            print("\n  WARNING: training on a small embedded sample instead of Tiny Shakespeare.")
            print("  The loss numbers in Chapter 11 (~4.2 -> below 2.5, vocab 65) assume the")
            print("  real corpus; on this sample the model memorises the text and the loss")
            print("  falls toward 0. To fix, fetch the file yourself and re-run:")
            print(f"    curl -sSLo {DATA_PATH} {DATA_URL}\n")
            return FALLBACK
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


class CharDataset:
    def __init__(self, text, block_size, device):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split = int(0.9 * len(data))
        self.train, self.val = data[:split], data[split:]
        self.block_size = block_size
        self.device = device

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def batch(self, split, batch_size):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)


# ---------------------------------------------------------------------------
# RoPE --- Section 11.5.2
# ---------------------------------------------------------------------------
def precompute_rope(head_dim, max_len, base=10000.0, device="cpu"):
    """Returns (cos, sin) of shape (max_len, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float()
                               / head_dim))
    pos = torch.arange(max_len, device=device).float()
    ang = torch.outer(pos, inv_freq)                   # (max_len, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin, offset=0):
    """x: (B, H, T, Dh). Rotates each adjacent pair of channels by m*theta."""
    T, half = x.shape[2], x.shape[3] // 2
    c = cos[offset:offset + T].view(1, 1, T, half)
    s = sin[offset:offset + T].view(1, 1, T, half)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# ---------------------------------------------------------------------------
# Attention --- Sections 11.2, 11.3, 11.6, 11.9
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        offset = 0 if cache is None else cache[0].shape[2]
        q = apply_rope(q, cos, sin, offset)
        k = apply_rope(k, cos, sin, offset)

        if cache is not None:                          # KV-cache --- Section 11.10.1
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)

        # PyTorch dispatches this to a FlashAttention kernel when it can.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(cache is None),                 # cached decode attends to all
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache


class MLP(nn.Module):
    """FFN sublayer --- Section 11.6. 4x expansion, GELU."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm residual block --- Section 11.6."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        h, new_cache = self.attn(self.ln1(x), cos, sin, cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPTConfig:
    def __init__(self, vocab_size, n_layer=2, n_head=2, n_embd=64,
                 block_size=128, dropout=0.0):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight         # weight tying --- 11.6

        cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        # Scaled init on residual projections (GPT-2 recipe).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None):
        x = self.tok_emb(idx)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, self.cos, self.sin,
                         None if caches is None else caches[i])
            new_caches.append(c)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Incremental decode using the KV cache --- Section 11.10.1."""
        self.eval()
        caches = None
        cur = idx
        for _ in range(max_new_tokens):
            if idx.shape[1] > self.cfg.block_size:     # cache is bounded
                caches, cur, idx = None, cur[:, -self.cfg.block_size:], \
                    idx[:, -self.cfg.block_size:]
            logits, _, caches = self(cur, caches=caches)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            cur = nxt                                  # feed one token next time
        return idx


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_layer, n_head, n_embd = (4, 4, 192) if args.big else (2, 2, 64)
    steps = args.steps if args.steps else (2000 if args.big else 100)
    batch_size, block_size, lr = (32, 128, 3e-4) if args.big else (16, 128, 3e-3)

    text = load_text()
    data = CharDataset(text, block_size, device)
    cfg = GPTConfig(data.vocab_size, n_layer, n_head, n_embd, block_size)
    model = GPT(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | vocab {data.vocab_size} | "
          f"{n_layer}L {n_head}H {n_embd}D | {n_params/1e6:.3f}M params")
    print(f"random-init loss should be ~log({data.vocab_size}) = "
          f"{math.log(data.vocab_size):.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    if not args.sample_only:
        model.train()
        first = None
        for step in range(1, steps + 1):
            x, y = data.batch("train", batch_size)
            _, loss, _ = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if first is None:
                first = loss.item()
            if step % max(1, steps // 10) == 0 or step == 1:
                print(f"  step {step:5d}   loss {loss.item():.4f}")
        print(f"\nloss {first:.3f} -> {loss.item():.3f}")

    print("\n--- sample ---")
    prompt = "ROMEO:" if "ROMEO:" in text else text[:6]
    idx = torch.tensor([data.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, 300, temperature=0.8, top_k=40)
    print(data.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
