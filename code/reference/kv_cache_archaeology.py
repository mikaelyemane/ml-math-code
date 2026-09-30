"""KV-cache memory, three independent ways --- Chapter 11.

The chapter states that Llama-3-70B costs ~328 KB per token of KV cache, and
~43 GB for a single 128K-context request. Those are load-bearing numbers in any
LLM-serving interview and in any capacity plan, so do not take them on faith.

This script computes both three separate ways that share no arithmetic:

  (a) the closed-form formula from the chapter;
  (b) a layer-by-layer accumulator that builds per-layer byte counts and sums;
  (c) an explicit tensor-shape calculation mirroring what an inference engine's
      KVCache class actually allocates.

All three must agree to the byte. If they ever disagree, the formula is wrong.

    $ python3 kv_cache_archaeology.py

Things to try:

  * Set n_kv_heads = n_heads (plain MHA instead of GQA). The cache multiplies
    by H/G = 8x. This is why every modern open-weight model uses GQA.
  * Set dtype_bytes = 1 (fp8 instead of bf16). The cache halves, and so does
    the dollars-per-request at serving time.

Units: byte counts are reported in decimal (1 GB = 1e9 bytes) to match the
chapter, with binary units alongside, because GPU specs and profiler output are
usually quoted in GiB and the two differ by 7%.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int          # == n_heads for MHA; < n_heads for GQA; 1 for MQA
    head_dim: int
    dtype_bytes: int         # 2 for bf16/fp16, 1 for fp8

    @property
    def d_model(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def gqa_ratio(self) -> float:
        return self.n_heads / self.n_kv_heads


LLAMA_3_70B = ModelSpec(
    name="Llama-3-70B",
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,            # GQA: 8 KV heads shared across 64 query heads
    head_dim=128,
    dtype_bytes=2,           # bf16
)


# ---------------------------------------------------------------------------
# (a) closed form
# ---------------------------------------------------------------------------
def bytes_per_token_closed_form(spec: ModelSpec) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x dtype_bytes."""
    return 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * spec.dtype_bytes


# ---------------------------------------------------------------------------
# (b) layer-by-layer accumulator
# ---------------------------------------------------------------------------
def bytes_per_token_by_layer(spec: ModelSpec) -> int:
    """Build each layer's contribution and sum. No formula, just addition."""
    total = 0
    for _ in range(spec.n_layers):
        k_bytes = spec.n_kv_heads * spec.head_dim * spec.dtype_bytes
        v_bytes = spec.n_kv_heads * spec.head_dim * spec.dtype_bytes
        total += k_bytes + v_bytes
    return total


# ---------------------------------------------------------------------------
# (c) explicit tensor shapes, as an engine would allocate them
# ---------------------------------------------------------------------------
def bytes_total_tensor_shapes(spec: ModelSpec, seq_len: int, batch: int = 1) -> int:
    """Allocate [batch, n_kv_heads, seq_len, head_dim] for K and V, per layer."""
    total = 0
    for _ in range(spec.n_layers):
        for _tensor in ("k", "v"):
            shape = (batch, spec.n_kv_heads, seq_len, spec.head_dim)
            numel = 1
            for dim in shape:
                numel *= dim
            total += numel * spec.dtype_bytes
    return total


def fmt(n_bytes: int) -> str:
    return (f"{n_bytes:,} B  =  {n_bytes / 1e9:.3f} GB (decimal)"
            f"  =  {n_bytes / 2**30:.3f} GiB (binary)")


def report(spec: ModelSpec, contexts=(1, 8192, 131072)) -> None:
    print(f"{spec.name}")
    print(f"  layers {spec.n_layers}   query heads {spec.n_heads}   "
          f"KV heads {spec.n_kv_heads}   head_dim {spec.head_dim}   "
          f"dtype {spec.dtype_bytes} B")
    print(f"  d_model {spec.d_model}   GQA ratio H/G = {spec.gqa_ratio:.0f}x")
    print()

    a = bytes_per_token_closed_form(spec)
    b = bytes_per_token_by_layer(spec)
    c = bytes_total_tensor_shapes(spec, seq_len=1)

    print("  per token")
    print(f"    (a) closed form     {a:,} B")
    print(f"    (b) layer-by-layer  {b:,} B")
    print(f"    (c) tensor shapes   {c:,} B")
    assert a == b == c, f"METHODS DISAGREE: {a} vs {b} vs {c}"
    print(f"    all three agree     {a:,} B  =  {a / 1e3:.2f} KB "
          f"=  {a / 1024:.0f} KiB exactly")
    print()

    print("  by context length")
    for n in contexts:
        total = bytes_total_tensor_shapes(spec, seq_len=n)
        assert total == a * n, "tensor-shape path disagrees with per-token x n"
        print(f"    n = {n:>7,}   {fmt(total)}")
    print()


def main() -> None:
    print("=" * 72)
    report(LLAMA_3_70B)

    print("=" * 72)
    mha = ModelSpec(**{**LLAMA_3_70B.__dict__,
                       "name": "Llama-3-70B, hypothetical MHA (n_kv_heads = n_heads)",
                       "n_kv_heads": LLAMA_3_70B.n_heads})
    report(mha, contexts=(131072,))
    ratio = (bytes_per_token_closed_form(mha)
             / bytes_per_token_closed_form(LLAMA_3_70B))
    print(f"  MHA / GQA cache ratio: {ratio:.0f}x  "
          f"(= H/G = {LLAMA_3_70B.gqa_ratio:.0f})")
    print()

    print("=" * 72)
    fp8 = ModelSpec(**{**LLAMA_3_70B.__dict__,
                       "name": "Llama-3-70B, fp8 KV cache",
                       "dtype_bytes": 1})
    report(fp8, contexts=(131072,))
    print(f"  fp8 / bf16 cache ratio: "
          f"{bytes_per_token_closed_form(fp8) / bytes_per_token_closed_form(LLAMA_3_70B):.2f}x")
    print()

    print("=" * 72)
    print("Sanity checks against the numbers printed in Chapter 11:")
    per_tok = bytes_per_token_closed_form(LLAMA_3_70B)
    at_128k = per_tok * 131072
    checks = [
        ("~328 KB per token", abs(per_tok / 1e3 - 327.68) < 0.01),
        ("exactly 320 KiB per token", per_tok == 320 * 1024),
        ("~43 GB at 128K context", abs(at_128k / 1e9 - 42.95) < 0.05),
        ("exactly 40 GiB at 128K context", at_128k == 40 * 2**30),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    assert all(ok for _, ok in checks), "a printed figure does not reproduce"


if __name__ == "__main__":
    main()
