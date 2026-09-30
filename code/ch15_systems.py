"""
Ch15 — Inference Economics: roofline, TPOT, KV-cache memory, disaggregation.
Covers: arithmetic-intensity ratio, prefill vs. decode bound,
        Llama-3-70B KV-cache sizing, prefill-decode disaggregation profitability.
"""
import numpy as np


# ── Roofline & arithmetic intensity ────────────────────────────────────────────

def arithmetic_intensity(flops, bytes_moved):
    """AI = FLOPs / bytes. Compares to GPU's compute:bandwidth ratio."""
    return flops / bytes_moved


def is_compute_bound(ai, peak_flops, peak_bw):
    """Compute-bound iff AI > peak_FLOPs / peak_bandwidth (the GPU's ridge point)."""
    return ai > peak_flops / peak_bw


def roofline(ai, peak_flops, peak_bw):
    """Achievable throughput in FLOP/s under the roofline model."""
    return min(peak_flops, ai * peak_bw)


# ── Prefill and decode bounds for a Transformer ────────────────────────────────

def prefill_tflops_per_token(N_params, P_tokens):
    """
    Prefill cost: ~2N FLOPs per token in the main linear layers,
    plus attention's 2*P*d term per token. Returns TFLOPs/token total over all P.
    Simplified: 6N + 2*P*d per token (gradient + activation + attention),
    inference-only halves the gradient → 2N + 2*P*d per token.
    """
    # FLOPs per token in linear layers ≈ 2N (Chinchilla approximation)
    return 2 * N_params


def tpot_memory_bound(weight_bytes, peak_hbm_bw):
    """
    Decode TPOT (time per output token) when memory-bound:
        TPOT = total_weight_bytes_read / HBM_bandwidth.
    Each decode step reads every model weight once.
    """
    return weight_bytes / peak_hbm_bw


# ── KV-cache memory: Llama-3-70B at 128K context ───────────────────────────────

def kv_cache_bytes(L_layers, K_heads, d_head, P_tokens, dtype_bytes=2,
                   K_groups=None):
    """
    KV-cache size for one request.
    L_layers : transformer block count
    K_heads  : number of key/value heads (note: GQA may have K_heads < attention heads)
    d_head   : per-head dimension
    P_tokens : sequence length
    dtype_bytes: 2 for fp16/bf16, 1 for int8, 0.5 for int4
    K_groups : if set, treats K_heads as the # of KV-shared groups (Llama-3 uses 8)

    Returns bytes for the entire (K + V) cache for one request at length P.
    """
    n_kv_heads = K_heads if K_groups is None else K_groups
    # K and V both stored: factor of 2.
    return 2 * L_layers * n_kv_heads * d_head * P_tokens * dtype_bytes


def llama3_70b_kv():
    """Spec: 80 layers, 64 attn heads, 8 KV groups, d_head=128, bf16."""
    return dict(L_layers=80, K_heads=64, d_head=128, K_groups=8, dtype_bytes=2)


# ── Disaggregation profitability ───────────────────────────────────────────────

def disaggregation_profitable(K_bytes, fabric_bw_GBs, prefill_time_s,
                              margin=10):
    """
    Returns (is_profitable, transfer_time_s, prefill_time_s).
    Profitable iff transfer time is at least `margin`× smaller than prefill.
    """
    transfer_s = K_bytes / (fabric_bw_GBs * 1e9)
    return prefill_time_s / transfer_s > margin, transfer_s, prefill_time_s


# ── Speculative decoding acceptance math (sanity-check) ────────────────────────

def speculative_speedup(alpha, gamma, c=1.0):
    """
    Leviathan-Kalai speedup formula:
        E[tokens per call]   1 - alpha^(gamma+1)
        ──────────────────── = ─────────────────────
        E[wall time per call]  (1 - alpha)(c * gamma + 1)
    where alpha = acceptance rate, gamma = #drafts per round, c = cost ratio.
    """
    num = 1 - alpha ** (gamma + 1)
    den = (1 - alpha) * (c * gamma + 1)
    return num / den


# ── Toy INT8 activation / INT4 weight quantizer ────────────────────────────────

def quantize_int8_perchannel(W, axis=0):
    """Symmetric per-channel INT8 quantizer.
    Returns (W_quant_int8, scales) where W ≈ W_quant_int8 * scales.
    """
    abs_max = np.max(np.abs(W), axis=axis, keepdims=True)
    scales = abs_max / 127.0
    W_int = np.round(W / np.where(scales == 0, 1, scales)).clip(-128, 127)
    return W_int.astype(np.int8), scales


def quantize_int4_perchannel(W, axis=0):
    """Symmetric per-channel INT4 quantizer (packed into int8 for storage).
    Range: [-8, 7] (4-bit signed two's complement).
    """
    abs_max = np.max(np.abs(W), axis=axis, keepdims=True)
    scales = abs_max / 7.0
    W_int = np.round(W / np.where(scales == 0, 1, scales)).clip(-8, 7)
    return W_int.astype(np.int8), scales


def dequantize(W_quant, scales):
    return W_quant.astype(np.float32) * scales


def linear_quant_demo(d_in=256, d_out=128, n=200, rng=None):
    """Compare fp32 vs INT8 vs INT4 linear-layer matmul accuracy on synthetic
    Gaussian inputs and weights. Returns (mse_int8, mse_int4)."""
    if rng is None:
        rng = np.random.default_rng(15)
    W = rng.standard_normal((d_in, d_out)).astype(np.float32)
    X = rng.standard_normal((n, d_in)).astype(np.float32)
    Y_fp32 = X @ W

    W8, s8 = quantize_int8_perchannel(W, axis=0)
    Y_int8 = X @ dequantize(W8, s8)

    W4, s4 = quantize_int4_perchannel(W, axis=0)
    Y_int4 = X @ dequantize(W4, s4)

    mse_int8 = ((Y_fp32 - Y_int8) ** 2).mean()
    mse_int4 = ((Y_fp32 - Y_int4) ** 2).mean()
    return mse_int8, mse_int4, Y_fp32.var()


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(15)

    # 1) Roofline: A100 spec
    A100_TFLOPS_BF16 = 312e12   # 312 TFLOPS bf16
    A100_HBM_BW = 2.0e12        # 2 TB/s HBM bandwidth
    ridge_ai = A100_TFLOPS_BF16 / A100_HBM_BW
    print(f"A100 ridge-point arithmetic intensity: {ridge_ai:.0f} FLOPs/byte")

    # 2) Prefill compute, batch-1, 2K tokens, 70B params, fp16
    N = 70e9
    P = 2048
    prefill_flops = 2 * N * P
    # Weight bytes: 70B params × 2 bytes
    weight_bytes = N * 2
    # Prefill reads every weight once total (not per token), plus 2*P*d^2 activations
    prefill_bytes = weight_bytes + 2 * P * 8192 * 2
    prefill_ai = arithmetic_intensity(prefill_flops, prefill_bytes)
    print(f"Prefill arithmetic intensity (P=2048, 70B): {prefill_ai:.0f} FLOPs/byte "
          f"({'compute-bound' if prefill_ai > ridge_ai else 'memory-bound'})")

    # 3) Decode: 1 new token, reads all 70B weights → memory-bound by definition
    decode_flops_per_token = 2 * N   # ~2N FLOPs
    decode_bytes_per_token = weight_bytes
    decode_ai = arithmetic_intensity(decode_flops_per_token, decode_bytes_per_token)
    print(f"Decode arithmetic intensity (70B): {decode_ai:.1f} FLOPs/byte "
          f"({'compute-bound' if decode_ai > ridge_ai else 'memory-bound'})")
    tpot_ms = tpot_memory_bound(weight_bytes, A100_HBM_BW) * 1000
    print(f"Decode TPOT lower bound (memory-bound, 70B on A100): {tpot_ms:.1f} ms/token")
    print(f"  → max single-stream decode rate: {1000 / tpot_ms:.1f} tok/s")

    # 4) KV-cache for Llama-3-70B at 128K context
    cfg = llama3_70b_kv()
    kv_128k = kv_cache_bytes(**cfg, P_tokens=128 * 1024)
    print(f"\nLlama-3-70B KV-cache at 128K context: "
          f"{kv_128k / 1e9:.1f} GB per request")
    # Published spec: ~43 GB at 128K context (Ch11 / Ch15), decimal GB
    assert 38 < kv_128k / 1e9 < 45, \
        f"KV-cache calc deviates from published 43 GB spec ({kv_128k/1e9:.1f} GB)"
    print("  ✓ matches published Llama-3-70B 43 GB at 128K spec")

    # 5) KV-cache at 2K context (typical chat prefill)
    kv_2k = kv_cache_bytes(**cfg, P_tokens=2048)
    print(f"Llama-3-70B KV-cache at  2K context: "
          f"{kv_2k / 1e6:.1f} MB per request")

    # 6) Disaggregation profitability
    # 200 ms prefill (single A100 estimate), 50 GB/s InfiniBand fabric
    profit, transfer_s, prefill_s = disaggregation_profitable(
        K_bytes=kv_2k, fabric_bw_GBs=50, prefill_time_s=0.2, margin=10
    )
    print(f"\nDisaggregation (P=2K, 50 GB/s fabric, 200 ms prefill):")
    print(f"  KV-transfer time: {transfer_s * 1000:.1f} ms")
    print(f"  Prefill time:     {prefill_s * 1000:.1f} ms")
    print(f"  Profitable?       {profit} (margin >= 10x)")

    # 7) Speculative decoding speedup vs. acceptance rate
    print("\nSpeculative decoding speedup (gamma=4 drafts, c=0.1):")
    for alpha in (0.5, 0.7, 0.8, 0.9):
        s = speculative_speedup(alpha, gamma=4, c=0.1)
        print(f"  alpha = {alpha}: {s:.2f}x")

    # 8) INT8 / INT4 quantization accuracy
    rng = np.random.default_rng(15)
    mse_int8, mse_int4, signal_var = linear_quant_demo(rng=rng)
    print(f"\nLinear-layer quantization (256x128 matmul on Gaussian inputs):")
    print(f"  fp32 baseline signal variance: {signal_var:.3f}")
    print(f"  INT8 weights:  MSE = {mse_int8:.6f}  (SNR = {signal_var/mse_int8:>7.0f}x)")
    print(f"  INT4 weights:  MSE = {mse_int4:.6f}  (SNR = {signal_var/mse_int4:>7.0f}x)")
    print(f"  Storage:  fp32 = 4 B/param, INT8 = 1 B, INT4 = 0.5 B")
    print(f"  Cost-per-MB ratio fp32:INT8:INT4 = 8 : 2 : 1")
