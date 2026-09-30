"""Plan a 1M-DAU LLM product on the back of an envelope --- Chapter 15.

The challenge box in Chapter 15 asks you to size the inference fleet for a chat
product and decide which lever ships when the budget doesn't close. Do it on
paper first. Then run this file to check every number:

    $ python3 ch15_plan_1m_dau.py

Everything here comes from four equations in the chapter:

    step time       T_step(B) = (W + B * M_KV(N)) / B_HBM          (eq:step_time_batched)
    SLO batch cap   B_slo = (B_HBM / R_SLO - W) / M_KV(N)          (eq:bmax_slo)
    memory cap      B_cap = (C_HBM - W - workspace) / M_KV(N)      (eq:bmax_cap)
    cost            $/M tok = C_GPU / (3600 * T_out * U) * 1e6     (eq:cost_per_mtok)

plus the speculative-decoding speedup (eq:speculative_speedup, Chapter 14) and the
compound-routing sum sum_k p_k * ($/M tok)_k.

Change the numbers in PRODUCT and the Model specs and re-run: the point of a
back-of-the-envelope script is that the envelope is editable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

GB = 1e9


# =============================================================================
# The product (from the challenge box)
# =============================================================================

PRODUCT = dict(
    dau=1_000_000,
    sessions_per_user=1,        # the challenge implies one session per user-day
    messages_per_session=5,
    prompt_tokens=500,
    output_tokens=300,
    peak_to_average=3.0,        # diurnal skew
    slo_tokens_per_sec=80.0,    # TPOT SLO: 80 tok/s/user = 12.5 ms/token
    ttft_slo_sec=1.0,
    budget_per_day=1_000.0,
    gpu_cost_per_hour=3.0,
    duty_cycle=0.60,
)


# =============================================================================
# Hardware and models
# =============================================================================

@dataclass(frozen=True)
class GPU:
    name: str = "H100-80G"
    hbm_bytes: float = 80 * GB
    hbm_bw: float = 3.35e12          # bytes/s
    peak_flops: float = 989e12       # dense bf16
    workspace_bytes: float = 5 * GB  # activations + scratch reserved from the KV pool


@dataclass(frozen=True)
class Model:
    name: str
    params: float
    weight_bytes_per_param: float    # INT4 = 0.5
    layers: int
    kv_heads: int
    head_dim: int
    kv_bytes: float = 2.0            # bf16 KV cache; fp8 = 1

    @property
    def weight_bytes(self) -> float:
        return self.params * self.weight_bytes_per_param

    def kv_bytes_per_token(self) -> float:
        # 2 (K and V) * L * G * d_head * bytes   --- the chapter's KV formula
        return 2 * self.layers * self.kv_heads * self.head_dim * self.kv_bytes


LLAMA3_70B_INT4 = Model("Llama-3-70B INT4", 70e9, 0.5, layers=80, kv_heads=8, head_dim=128)
LLAMA3_8B_INT4 = Model("Llama-3-8B INT4 (the '7B' tier)", 8e9, 0.5, layers=32, kv_heads=8, head_dim=128)

# The chapter's hero configuration prices decode at ~1K tokens of live context.
LIVE_CONTEXT_TOKENS = 1024


# =============================================================================
# Chapter 15 equations
# =============================================================================

def t_step(model: Model, gpu: GPU, batch: int, n_ctx: int = LIVE_CONTEXT_TOKENS) -> float:
    """eq:step_time_batched --- one decode step streams W once plus every sequence's KV."""
    return (model.weight_bytes + batch * model.kv_bytes_per_token() * n_ctx) / gpu.hbm_bw


def b_max(model: Model, gpu: GPU, slo: float, n_ctx: int = LIVE_CONTEXT_TOKENS):
    """eq:bmax_slo and eq:bmax_cap; returns (B_max, B_slo, B_cap, which binds)."""
    m_kv = model.kv_bytes_per_token() * n_ctx
    b_slo = math.floor((gpu.hbm_bw / slo - model.weight_bytes) / m_kv)
    b_cap = math.floor((gpu.hbm_bytes - model.weight_bytes - gpu.workspace_bytes) / m_kv)
    return min(b_slo, b_cap), b_slo, b_cap, ("SLO" if b_slo <= b_cap else "memory")


def throughput(model: Model, gpu: GPU, slo: float) -> dict:
    """Aggregate output tokens/s/GPU at the largest batch that holds the SLO."""
    b, b_slo, b_cap, binds = b_max(model, gpu, slo)
    ts = t_step(model, gpu, b)
    # Sanity check that decode is still memory-bound at this batch: FLOPs the
    # step needs vs. the GPU's peak over the same step time.
    compute_use = (2 * model.params * b / ts) / gpu.peak_flops
    return dict(batch=b, b_slo=b_slo, b_cap=b_cap, binds=binds,
                t_step_ms=ts * 1e3, t_out=b / ts, compute_use=compute_use)


def cost_per_mtok(gpu_cost_per_hour: float, t_out: float, duty: float) -> float:
    """eq:cost_per_mtok."""
    return gpu_cost_per_hour / (3600 * t_out * duty) * 1e6


def spec_speedup(alpha: float, gamma: int, c: float) -> float:
    """eq:speculative_speedup: E[emitted] / (1 + c * gamma)."""
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (1 + c * gamma))


# =============================================================================
# The plan
# =============================================================================

def main() -> None:
    p, gpu = PRODUCT, GPU()
    rule = "=" * 76

    # ---- 1. Aggregate output-token rate --------------------------------------
    msgs_per_day = p["dau"] * p["sessions_per_user"] * p["messages_per_session"]
    out_per_day = msgs_per_day * p["output_tokens"]
    avg_rate = out_per_day / 86_400
    peak_rate = avg_rate * p["peak_to_average"]
    print(rule)
    print("1. Output-token rate")
    print(rule)
    print(f"  messages/day        {msgs_per_day:>14,.0f}")
    print(f"  output tokens/day   {out_per_day:>14,.0f}")
    print(f"  average rate        {avg_rate:>14,.0f} tok/s")
    print(f"  peak rate (x{p['peak_to_average']:.0f})     {peak_rate:>14,.0f} tok/s")

    # ---- 2. GPU count ---------------------------------------------------------
    big = throughput(LLAMA3_70B_INT4, gpu, p["slo_tokens_per_sec"])
    gpus_peak = math.ceil(peak_rate / big["t_out"])
    print(f"\n{rule}\n2. GPU count (continuous batching, Llama-3-70B INT4 on {gpu.name})\n{rule}")
    print(f"  B_max: SLO ceiling {big['b_slo']}, memory ceiling {big['b_cap']} -> "
          f"{big['batch']} ({big['binds']} binds)")
    print(f"  T_step({big['batch']}) = {big['t_step_ms']:.2f} ms  ->  T_out = "
          f"{big['t_out']:,.0f} tok/s/GPU  (the chapter's ~1.6K)")
    print(f"  compute check: {big['compute_use']:.0%} of peak FLOPs, so decode is memory-bound as assumed")
    print(f"  GPUs at peak = {peak_rate:,.0f} / {big['t_out']:,.0f} = "
          f"{peak_rate / big['t_out']:.1f}  ->  {gpus_peak} GPUs")

    # ---- 3. Cost against the budget -------------------------------------------
    usd_mtok = cost_per_mtok(p["gpu_cost_per_hour"], big["t_out"], p["duty_cycle"])
    per_day = usd_mtok * out_per_day / 1e6
    fixed_fleet = gpus_peak * 24 * p["gpu_cost_per_hour"]
    print(f"\n{rule}\n3. Cost (eq:cost_per_mtok at $"
          f"{p['gpu_cost_per_hour']:.0f}/GPU-h, U = {p['duty_cycle']:.0%})\n{rule}")
    print(f"  $/M output tokens   ${usd_mtok:.3f}")
    print(f"  $/day               ${per_day:,.0f}   vs budget ${p['budget_per_day']:,.0f}  ->  "
          f"{'UNDER' if per_day <= p['budget_per_day'] else 'OVER'} by "
          f"{abs(per_day / p['budget_per_day'] - 1):.0%}")
    print(f"  (A fleet held at peak size all day: {gpus_peak} x 24 h x ${p['gpu_cost_per_hour']:.0f}"
          f" = ${fixed_fleet:,.0f}/day,")
    print(f"   an effective duty cycle of 1/{p['peak_to_average']:.0f}. U = 60% assumes the fleet"
          f" scales down off-peak.)")

    # ---- 4. Levers ------------------------------------------------------------
    print(f"\n{rule}\n4. Levers (each alone, same SLO, same U)\n{rule}")
    levers = {}

    # (a) route half the traffic to a 7-8B model
    small = throughput(LLAMA3_8B_INT4, gpu, p["slo_tokens_per_sec"])
    small_mtok = cost_per_mtok(p["gpu_cost_per_hour"], small["t_out"], p["duty_cycle"])
    routed = 0.5 * usd_mtok + 0.5 * small_mtok
    levers["route 50% to 8B"] = routed
    print("  (a) Model routing, p = 0.5 to the small tier")
    print(f"      8B: B_max {small['batch']} ({small['binds']} binds), T_out "
          f"{small['t_out']:,.0f} tok/s/GPU, ${small_mtok:.3f}/M tok "
          f"({usd_mtok / small_mtok:.0f}x cheaper; {small['compute_use']:.0%} of peak FLOPs)")
    print(f"      compound: 0.5 x ${usd_mtok:.3f} + 0.5 x ${small_mtok:.3f} = ${routed:.3f}/M tok")

    # (b) fp8 KV cache
    fp8 = replace(LLAMA3_70B_INT4, name="Llama-3-70B INT4, fp8 KV", kv_bytes=1.0)
    big_fp8 = throughput(fp8, gpu, p["slo_tokens_per_sec"])
    fp8_mtok = cost_per_mtok(p["gpu_cost_per_hour"], big_fp8["t_out"], p["duty_cycle"])
    levers["fp8 KV cache"] = fp8_mtok
    print("  (b) FP8 KV cache")
    print(f"      B_max {big['batch']} -> {big_fp8['batch']} ({big_fp8['binds']} still binds), "
          f"T_out {big['t_out']:,.0f} -> {big_fp8['t_out']:,.0f} tok/s/GPU, "
          f"${fp8_mtok:.3f}/M tok ({big_fp8['compute_use']:.0%} of peak FLOPs)")
    print("      The batch doubles, but not because more requests fit: the memory ceiling was")
    print(f"      never binding (it allows {big['b_cap']}). Halving KV bytes halves the per-user"
          " traffic")
    print("      each decode step must stream, so the SLO's byte budget holds twice the users.")

    # (c) speculative decoding
    s = spec_speedup(alpha=0.7, gamma=4, c=0.10)
    spec_mtok = usd_mtok / s
    levers["speculative decoding (bound)"] = spec_mtok
    print("  (c) Speculative decoding, gamma = 4, alpha = 0.7, 7-8B draft (c = 0.10)")
    print(f"      S = (1 - 0.7^5) / (0.3 x 1.4) = {s:.2f}x  ->  at most ${spec_mtok:.3f}/M tok")
    verify_tokens = big["batch"] * 5
    intensity = 2 * verify_tokens / LLAMA3_70B_INT4.weight_bytes_per_param
    ridge = gpu.peak_flops / gpu.hbm_bw
    print(f"      Upper bound only: at B = {big['batch']} a verify pass carries {verify_tokens} tokens,"
          f" ~{intensity:.0f} FLOPs per")
    print(f"      INT4 weight byte against a ridge point of ~{ridge:.0f}. The pass is compute-bound,"
          " so the")
    print("      memory-bound speedup formula overstates the gain ('Speculative Decoding: Revisited').")
    print("\n  lever                              $/M tok     $/day   saving")
    for name, v in sorted(levers.items(), key=lambda kv: kv[1]):
        d = v * out_per_day / 1e6
        print(f"  {name:<32} ${v:>7.3f}  ${d:>7,.0f}   {1 - v / usd_mtok:>5.0%}")

    # ---- 5. Decision and residual risk -----------------------------------------
    both = 0.5 * fp8_mtok + 0.5 * small_mtok
    print(f"\n{rule}\n5. What ships\n{rule}")
    print(f"  FP8 KV first: ${fp8_mtok * out_per_day / 1e6:,.0f}/day, a serving flag with no"
          " router to get wrong.")
    print(f"  Routing second, if needed: FP8 KV + 50% routing = ${both:.3f}/M tok = "
          f"${both * out_per_day / 1e6:,.0f}/day.")

    # Residual risk: prefill is compute, and the decode figures above ignore it.
    prefill_flops = 2 * LLAMA3_70B_INT4.params * p["prompt_tokens"]
    mfu = 0.5
    prefill_sec = prefill_flops / (mfu * gpu.peak_flops)
    peak_msgs = msgs_per_day / 86_400 * p["peak_to_average"]
    print("\n  Residual risk: prefill")
    print(f"    one {p['prompt_tokens']}-token prompt = {prefill_flops:.1e} FLOPs = "
          f"{prefill_sec * 1e3:.0f} ms of an H100 at {mfu:.0%} MFU "
          f"(TTFT budget {p['ttft_slo_sec']:.0f} s: fine)")
    print(f"    at peak, {peak_msgs:,.0f} messages/s x {prefill_sec:.3f} s = "
          f"{peak_msgs * prefill_sec:.0f} GPU-equivalents of prefill compute")
    print("    That is not in the decode-only T_out above. Prefix caching of the conversation")
    print("    history ('The KV-Cache Economy') is what keeps it from doubling the fleet.")


if __name__ == "__main__":
    main()
