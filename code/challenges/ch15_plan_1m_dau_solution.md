# Solution: "Plan a 1M-DAU LLM product on the back of an envelope" (Chapter 15)

**Stop here if you haven't done the envelope yourself.** The challenge only pays off if you do the arithmetic before you check it. `python3 ch15_plan_1m_dau.py` prints every number below.

The spec is:

- 1M DAU, one session each, 5 messages per session.
- 500-token prompts and 300-token outputs.
- Llama-3-70B at INT4 on H100-80G.
- SLOs: 80 tok/s per user and TTFT ≤ 1 s at p95.
- Budget: $1,000/day.

## 1. Peak output-token rate

- 1M users × 5 messages × 300 tokens = **1.5 B output tokens/day**.
- Averaged over 86,400 s, that is **17.4K tok/s**.
- With the 3× diurnal skew, peak is **52K tok/s**.

## 2. GPU count

The chapter's 1.6K tok/s/GPU is not a free parameter. It falls out of `eq:bmax_slo`:

- The 80 tok/s SLO means each decode step may take at most 12.5 ms. At 3.35 TB/s, that step may move 41.9 GB.
- The INT4 weights take 35 GB of that, leaving 6.9 GB for KV traffic.
- At 1K tokens of live context, each user's KV cache is 335 MB, so the budget holds **20 users**.
- The KV pool itself could hold 119 users, so the **SLO binds**, not memory.
- `eq:step_time_batched` then gives T_step(20) = 12.45 ms, so T_out = 20 / 12.45 ms ≈ **1,606 tok/s/GPU**.

That gives 52,083 / 1,606 = 32.4, so **33 GPUs** at peak.

## 3. Cost against the budget

At $3/GPU-h and U = 0.6, `eq:cost_per_mtok` gives:

    3 / (3600 × 1606 × 0.6) × 10⁶ = $0.865 per M output tokens
    × 1,500 M tokens/day = $1,297/day

That is **30% over budget**.

The 60% duty cycle assumes the fleet scales down off-peak. A fleet held at its 33-GPU peak around the clock costs 33 × 24 × $3 = **$2,376/day**. That is an effective duty cycle of 1/3, set by the peak-to-average ratio. Say which case you are in when you present the number.

## 4. The levers

Each lever is priced on its own, at the same SLO and the same U.

| Lever | What changes | $/M tok | $/day | Saving |
|---|---|---|---|---|
| FP8 KV cache | B_max 20 → 40, T_out 1.6K → 3.2K | $0.432 | $648 | 50% |
| Speculative decoding, γ=4, α=0.7, c=0.1 | S = 1.98× (upper bound) | ≥ $0.437 | ≥ $655 | ≤ 50% |
| Route 50% to an 8B tier | 8B serves 22.6K tok/s/GPU, 14× cheaper | $0.463 | $695 | 46% |

**FP8 KV works, but for a different reason than the challenge suggests.**
- The challenge frames it as "double batch size" by fitting more requests in memory. But the memory ceiling was never the constraint: it allowed 119 users and the SLO allowed only 20.
- What FP8 does is halve M_KV in the *denominator* of `eq:bmax_slo`. Each user's KV traffic per decode step halves, so the same 6.9 GB byte budget carries 40 users.
- If you reason "more requests fit, so throughput doubles," you get the right number by accident. With a slightly different SLO you would get the wrong one.

**Speculative decoding's 1.98× is the memory-bound formula, and the premise fails at this batch.**
- The speedup formula from Chapter 14 is S = (1 − α^{γ+1}) / ((1 − α)(1 + cγ)).
- At B = 20 with γ = 4, each verify pass carries 100 tokens. Against INT4 weights that is about 400 FLOPs per weight byte, past the H100's ridge point of about 295.
- So the verify pass is compute-bound, and the realized gain is below 1.98×. "Speculative Decoding: Revisited" makes this point: production stacks drop γ to 1–2, or turn speculation off, at high occupancy.

**Routing is real money but carries a quality risk the other two don't.**
- An 8B INT4 model (Llama-3-8B geometry) fits 282 users under the same SLO and serves 22.6K tok/s/GPU.
- The compound cost is 0.5 × $0.865 + 0.5 × $0.062 = $0.463.
- The saving depends on the router sending the right half. The chapter's routing exercise shows how quickly misrouting erases it.

## 5. What ships

1. **FP8 KV cache first:** $648/day, well inside budget. It is a serving-config change with a known, measurable quality cost: run your eval on FP8 KV before flipping it.
2. **Routing in reserve.** FP8 KV plus 50% routing comes to $0.247/M tok, or about $370/day. That headroom is what you spend when DAU doubles.
3. **Speculative decoding: measure, don't plan on it.** At this occupancy it is a latency tool, not a cost tool.

## Residual risk to write down

- **Prefill isn't in T_out.**
  - A 500-token prompt is 7 × 10¹³ FLOPs, or about 118 ms of an H100 at 60% MFU. TTFT is fine.
  - Across the fleet it is not small: 174 messages/s at peak × 0.118 s ≈ **20 GPU-equivalents** of prefill compute on top of the 33 decode GPUs.
  - Most of a chat prompt is conversation history, so prefix caching ("The KV-Cache Economy") is what keeps this from doubling the fleet. Put the cache hit rate on the dashboard next to duty cycle.
- **Queueing headroom.** The chapter's single-GPU M/M/1 bound puts the p95 duty cycle near ρ ≈ 0.5. A 33-GPU pool behind one scheduler absorbs variability far better than that bound, so ρ ≈ 0.5 is pessimistic here, but U = 0.6 at full B_max is not free either.
- **The 60% duty cycle is an autoscaling promise.** If the fleet can't scale down at night, the budget case is the $2,376/day one, and FP8 KV plus routing is required, not optional.
- **The live context assumption.** Everything above uses the chapter's 1K tokens of live context. Five-turn sessions grow past that. M_KV is linear in context, so B_max shrinks in proportion. Re-run the script with `LIVE_CONTEXT_TOKENS` at your measured p50.
