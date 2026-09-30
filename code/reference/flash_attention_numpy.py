"""FlashAttention forward in pure NumPy --- Chapter 11.

The derivation in the chapter is enough to implement the forward pass in about
fifty lines. This is that implementation, plus a numerical check against vanilla
O(n^2) attention and the working-memory accounting.

Write your own first. The online-softmax recurrence is the entire algorithm:

    m_new = max(m_old, rowmax(S_block))
    l_new = exp(m_old - m_new) * l_old + rowsum(exp(S_block - m_new))
    O_new = (exp(m_old - m_new) * l_old * O_old
             + exp(S_block - m_new) @ V_block) / l_new

Everything else is bookkeeping.

    $ python3 flash_attention_numpy.py

The point is not that this is fast --- in NumPy it is slower than the vanilla
version, because the win comes from keeping tiles in SRAM on a GPU and NumPy has
no SRAM to speak of. The point is that it is *numerically identical* while never
materialising the n x n score matrix, which is what makes long context possible
at a fixed memory budget.
"""

import numpy as np


def vanilla_attention(Q, K, V):
    """Standard attention. Materialises the full n x n score matrix."""
    d = Q.shape[-1]
    S = (Q @ K.T) / np.sqrt(d)                    # (n, n)  <-- the problem
    S = S - S.max(axis=-1, keepdims=True)         # standard max-subtraction
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def flash_attention(Q, K, V, block_q=64, block_k=64):
    """Tiled forward pass with online softmax. Never allocates an (n, n) array.

    Peak score memory is (block_q, block_k) instead of (n, n).
    """
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    O = np.zeros((n, d), dtype=Q.dtype)

    for i in range(0, n, block_q):
        qi = Q[i:i + block_q]                            # (Bq, d)
        rows = qi.shape[0]

        # Running softmax state for this row block.
        m = np.full((rows, 1), -np.inf, dtype=Q.dtype)   # running row max
        l = np.zeros((rows, 1), dtype=Q.dtype)           # running denominator
        acc = np.zeros((rows, d), dtype=Q.dtype)         # running numerator

        for j in range(0, n, block_k):
            kj = K[j:j + block_k]                        # (Bk, d)
            vj = V[j:j + block_k]                        # (Bk, d)

            s = (qi @ kj.T) * scale                      # (Bq, Bk)  <-- only this
            m_new = np.maximum(m, s.max(axis=-1, keepdims=True))

            # Rescale the state carried from previous blocks onto the new max.
            rescale = np.exp(m - m_new)
            p = np.exp(s - m_new)                        # (Bq, Bk)

            l = rescale * l + p.sum(axis=-1, keepdims=True)
            acc = rescale * acc + p @ vj
            m = m_new

        O[i:i + block_q] = acc / l

    return O


def memory_ratio(n, d, block_q=64, block_k=64):
    """Peak score-matrix elements, vanilla vs tiled.

    Vanilla holds n^2 score entries. Tiled holds Bq x Bk scores plus the
    Bq x d and Bk x d tiles it is working on. The chapter's ratio is
    n^2 / (Bq*d + Bk*d); it grows without bound in n, which is the structural
    win --- vanilla is quadratic in n, tiled is constant.
    """
    return n * n / (block_q * d + block_k * d)


def main():
    rng = np.random.default_rng(0)
    n, d = 256, 64
    Q = rng.normal(size=(n, d)).astype(np.float32)
    K = rng.normal(size=(n, d)).astype(np.float32)
    V = rng.normal(size=(n, d)).astype(np.float32)

    ref = vanilla_attention(Q, K, V)
    out = flash_attention(Q, K, V, block_q=64, block_k=64)

    max_abs = float(np.max(np.abs(ref - out)))
    max_rel = max_abs / float(np.max(np.abs(ref)))

    print(f"problem: n = {n}, d = {d}, float32, blocks 64 x 64")
    print(f"  max |flash - vanilla|   {max_abs:.3e}")
    print(f"  max relative error      {max_rel:.3e}")
    print(f"  within 1e-5             {max_abs <= 1e-5}")
    assert max_abs <= 1e-5, "tiled forward does not match vanilla attention"
    print()

    print("block-size independence (the recurrence must not depend on tiling)")
    for bq, bk in ((32, 32), (64, 128), (128, 64), (256, 256), (16, 256)):
        o = flash_attention(Q, K, V, bq, bk)
        e = float(np.max(np.abs(ref - o)))
        print(f"  Bq={bq:>3} Bk={bk:>3}   max err {e:.3e}   ok={e <= 1e-5}")
        assert e <= 1e-5
    print()

    print("peak score memory, vanilla n^2 vs tiled Bq*d + Bk*d")
    print(f"  {'n':>8}  {'vanilla (n^2)':>16}  {'tiled':>10}  {'ratio':>10}")
    for nn in (256, 1024, 4096, 16384, 131072):
        print(f"  {nn:>8,}  {nn * nn:>16,}  {64 * d + 64 * d:>10,}  "
              f"{memory_ratio(nn, d):>10,.0f}x")
    print()
    print("  Vanilla grows quadratically; the tiled footprint does not grow at")
    print("  all. That gap is why 128K context is affordable and why the peak")
    print("  HBM activation footprint drops from O(n^2) to O(nd).")


if __name__ == "__main__":
    main()
