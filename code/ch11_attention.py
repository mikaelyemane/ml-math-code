"""
Ch11 — Scaled Dot-Product Attention & Transformers
Covers: SDPA forward (with causal mask), softmax Jacobian, SDPA backward
        with a finite-difference check, multi-head attention,
        KV-cache single-step decode.
"""
import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax along given axis."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ── Scaled dot-product attention ──────────────────────────────────────────────

def sdpa(Q, K, V, mask=None):
    """
    Attention(Q,K,V) = softmax(QK^T / √d_k + M) V
    Q,K: (n, d_k)   V: (n, d_v)   mask: (n, n) additive (-inf for masked)
    Returns: output (n, d_v), attention weights (n, n)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # (n, n)
    if mask is not None:
        scores = scores + mask
    A = softmax(scores, axis=-1)            # (n, n)
    return A @ V, A


def causal_mask(n):
    """Causal mask: -∞ strictly above the diagonal; M[i,j] = 0 if j≤i, else -∞."""
    M = np.zeros((n, n))
    M[np.triu_indices(n, k=1)] = -np.inf
    return M


# ── Softmax Jacobian ──────────────────────────────────────────────────────────

def softmax_jacobian(a):
    """
    J = diag(a) - a a^T   (book Eq. for softmax Jacobian)
    a: (K,) softmax output.  Returns (K, K) Jacobian.
    """
    return np.diag(a) - np.outer(a, a)


# ── SDPA backward ─────────────────────────────────────────────────────────────

def sdpa_backward(Q, K, V, dO, mask=None):
    """
    Gradients of a loss through O = softmax(QK^T/√d_k + M) V, given dO = dL/dO.
    dV = A^T dO;  dA = dO V^T;  dS = A ⊙ (dA - rowsum(A ⊙ dA));
    dQ = dS K / √d_k;  dK = dS^T Q / √d_k.
    """
    d_k = Q.shape[-1]
    _, A = sdpa(Q, K, V, mask)
    dV = A.T @ dO
    dA = dO @ V.T
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True))
    return dS @ K / np.sqrt(d_k), dS.T @ Q / np.sqrt(d_k), dV


# ── Multi-head attention ──────────────────────────────────────────────────────

def multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads, causal=False):
    """
    MHA: concatenate h attention heads, project with W_o.
    X:  (n, d_model)
    Wq,Wk,Wv: (d_model, d_model)   Wo: (d_model, d_model)
    """
    n, d_model = X.shape
    d_k = d_model // n_heads

    Q = X @ Wq;  K = X @ Wk;  V = X @ Wv    # (n, d_model)
    mask = causal_mask(n) if causal else None

    heads = []
    for h in range(n_heads):
        Qh = Q[:, h*d_k:(h+1)*d_k]
        Kh = K[:, h*d_k:(h+1)*d_k]
        Vh = V[:, h*d_k:(h+1)*d_k]
        out_h, _ = sdpa(Qh, Kh, Vh, mask)
        heads.append(out_h)

    return np.hstack(heads) @ Wo             # (n, d_model)


# ── KV-cache: decode one new token ───────────────────────────────────────────

def kv_cache_step(q_new, K_cache, V_cache, k_new, v_new):
    """
    Append new key/value to cache and compute attention for the new query.
    q_new: (1, d_k)   K_cache,V_cache: (t, d_k),(t, d_v)   k_new,v_new: (1,d_k),(1,d_v)
    Returns: output (1, d_v), updated K/V caches.
    """
    K_new = np.vstack([K_cache, k_new])     # (t+1, d_k)
    V_new = np.vstack([V_cache, v_new])     # (t+1, d_v)
    out, A = sdpa(q_new, K_new, V_new)      # no mask: new token attends to all past + self
    return out, K_new, V_new


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n, d_k, d_v = 6, 8, 8

    Q = rng.normal(size=(n, d_k))
    K = rng.normal(size=(n, d_k))
    V = rng.normal(size=(n, d_v))

    # Full vs causal attention
    out_full,   A_full   = sdpa(Q, K, V)
    out_causal, A_causal = sdpa(Q, K, V, mask=causal_mask(n))

    print("Full attention weights (row = query, col = key):")
    print(A_full.round(3))
    print("\nCausal attention weights (upper triangle = 0):")
    print(A_causal.round(3))
    print("Upper triangle is exactly 0:", np.allclose(np.triu(A_causal, k=1), 0))

    # Softmax Jacobian check
    a = softmax(rng.normal(size=5))
    J_analytical = softmax_jacobian(a)
    # Numerical: d(softmax(s))/ds at s = log(a) (so softmax(s)=a)
    s = np.log(a)
    h = 1e-5
    J_numerical = np.zeros((5, 5))
    for j in range(5):
        sp = s.copy(); sp[j] += h
        sm = s.copy(); sm[j] -= h
        J_numerical[:, j] = (softmax(sp) - softmax(sm)) / (2 * h)
    print(f"\nSoftmax Jacobian max err: {np.max(np.abs(J_analytical - J_numerical)):.2e}")

    # SDPA backward vs. finite differences, loss L = ||O||_F^2 (dL/dO = 2 O)
    mask = causal_mask(n)
    loss = lambda Q_, K_, V_: np.sum(sdpa(Q_, K_, V_, mask)[0] ** 2)
    O, _ = sdpa(Q, K, V, mask)
    grads = sdpa_backward(Q, K, V, 2 * O, mask)
    for name, X_, g in zip("QKV", (Q, K, V), grads):
        g_num = np.zeros_like(X_)
        for idx in np.ndindex(X_.shape):
            Xp = X_.copy(); Xp[idx] += h
            Xm = X_.copy(); Xm[idx] -= h
            args_p = [Q, K, V]; args_m = [Q, K, V]
            args_p["QKV".index(name)] = Xp; args_m["QKV".index(name)] = Xm
            g_num[idx] = (loss(*args_p) - loss(*args_m)) / (2 * h)
        print(f"d{name} max err vs finite difference: {np.max(np.abs(g - g_num)):.2e}")

    # MHA
    d_model, n_heads2 = 16, 4
    X  = rng.normal(size=(n, d_model))
    Wq = rng.normal(size=(d_model, d_model)) * 0.1
    Wk = rng.normal(size=(d_model, d_model)) * 0.1
    Wv = rng.normal(size=(d_model, d_model)) * 0.1
    Wo = rng.normal(size=(d_model, d_model)) * 0.1
    out_mha = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads2, causal=True)
    print(f"\nMHA output shape: {out_mha.shape}  (expected ({n}, {d_model}))")

    # KV cache: decoding the last token from a cache of the first n-1 tokens
    out_kv, _, _ = kv_cache_step(Q[-1:], K[:-1], V[:-1], K[-1:], V[-1:])
    print("KV-cache step matches last causal row:", np.allclose(out_kv, out_causal[-1:]))
