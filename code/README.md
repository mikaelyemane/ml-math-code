# Code Companion to *Math for ML Engineers*

Three kinds of files live here:

- **`*.py` at the top level**: pure NumPy mirrors of the chapter derivations. No autograd, no PyTorch, no black boxes. Every file runs with `python3 <file>.py` and prints a gradient check or numerical verification.
- **`reference/`**: cleanest reference implementations of the flagship topics — nanoGPT, FlashAttention, KV-cache memory math. PyTorch where appropriate.
- **`challenges/`**: hands-on problems with starter files and solutions. The book has Challenge callout boxes that point to them.

**Dependencies:** `numpy` for the top-level files (`pytest` optional for `ch18_testing.py`). `torch` for `reference/nano_gpt.py`. Python 3.9+.

---

## Files

| File | Chapter | What it implements |
|---|---|---|
| `ch01_linalg.py` | Ch. 1 — Foundations | Economy SVD, Eckart–Young low-rank approx, gradient of quadratic forms, finite-difference Jacobian |
| `ch02_probability.py` | Ch. 2 — Probability | Gaussian MLE, MAP with Gaussian prior, multivariate conditional, Beta-Binomial |
| `ch03_information_theory.py` | Ch. 3 — Information Theory | Entropy, cross-entropy, KL (H(p,q)=H(p)+KL identity), mutual information, Gaussian KL |
| `ch04_linear_models.py` | Ch. 4 — Linear Models | Normal equations, Ridge regression, logistic regression, softmax CE gradient check, ISTA for Lasso |
| `ch05_optimizers.py` | Ch. 5 — Optimization | Gradient descent, momentum, Adam with bias correction, Newton step on a quadratic |
| `ch06_gp_regression.py` | Ch. 6 — Kernel Methods | RBF kernel, GP posterior (Cholesky), 95% credible band |
| `ch07_backprop.py` | Ch. 7 — Shallow Networks | 2-layer ReLU net forward/backward, gradient check, XOR demo |
| `ch08_batch_norm.py` | Ch. 8 — Deep Networks | BatchNorm forward, 3-term backward (dx, dγ, dβ), gradient check |
| `ch09_convolutions.py` | Ch. 9 — Convolutions | 1-D conv as Toeplitz matrix, 2-D conv forward/backward, receptive field |
| `ch10_recurrent.py` | Ch. 10 — Recurrent Networks | RNN forward + BPTT (gradient check, max err ~5e-12), LSTM with all four gates |
| `ch11_attention.py` | Ch. 11 — Attention | SDPA, causal mask, softmax Jacobian, SDPA backward + finite-difference check, multi-head attention, KV-cache step |
| `ch12_vae.py` | Ch. 12 — Generative Models | VAE with exact reparameterised gradients (gradient-checked) and a 200-step ELBO climb on held-out data; GMM-EM; DDPM forward marginal (closed form vs. iterated) and an oracle-noise ancestral sampler at the chapter's T = 1000, with T = 200 alongside to show the terminal-SNR bias |
| `ch13_policy_gradient.py` | Ch. 13 — RL | REINFORCE with running-mean baseline (variance reduction shown), PPO clipped surrogate vs vanilla PG |
| `ch14_frontiers.py` | Ch. 14 — Frontiers | Empirical Fisher, natural gradient, denoising score matching, LoRA savings |
| `ch15_systems.py` | Ch. 15 — Inference Economics | Roofline, prefill/decode arithmetic intensity, Llama-3-70B KV-cache math, disaggregation profitability, speculative-decoding speedup |
| `ch16_interpretability.py` | Ch. 16 — Mechanistic Interpretability | Near-orthogonal packing demo, one-sided vs. two-sided soft-threshold, toy sparse autoencoder with dead-latent mitigation, a hand-built two-head induction circuit (K-composition), denoising/noising activation patching, difference-of-means steering vector with Cauchy–Schwarz verification, chain-of-thought faithfulness ablation |
| `ch17_point_in_time.py` | Ch. 17 — Abstraction & Interfaces | The chapter's naive, as-of and event-time joins on its own tables (reproduces the as-of table and the 1.00 vs 0.33 AUC), the point-in-time contract test, the config-driven feature test |
| `ch18_testing.py` | Ch. 18 — Testing & Versioning | Boundary unit test for `bucketize_age`, name-swap invariance test against a clean and a shortcut model, directional test, content-hash data manifest; runs with or without pytest |
| `ch19_drift_monitoring.py` | Ch. 19 — Technical Debt & Observability | The PSI listing, PSI as the Jeffreys divergence across bin counts, prediction-distribution shift vs. noise floor, the empirical-null threshold with consecutive-day vs. same-weekday pairing |

---

## Gradient checks

Every file that derives a gradient checks it against finite differences:

```
ch01  SVD reconstruction error            1.11e-15   (not a gradient check)
ch01  quadratic-form gradient max err     6.24e-11
ch04  softmax grad max err                1.78e-11
ch07  gradient check, max RELATIVE err    5.54e-12   (W1/b1/W2/b2, untrained net)
ch08  max |dx analytical - numerical|     3.10e-11
ch09  kernel gradient check max err       2.77e-10
ch10  BPTT Wh gradient check max err      5.24e-12
ch11  softmax Jacobian max err            5.38e-12
```
These are the values the scripts actually print on a clean run (deterministic —
every script seeds its RNG). Exact magnitudes depend on the finite-difference
step `h`; all are far below the chapter-derived target of $10^{-5}$.

**A note on how these checks are written.** A gradient check is only worth the
line it prints if it can *fail*. Two ways to write one that cannot:

- **Check at a point where the gradient is zero.** With zero-initialised biases,
  `relu(W1 @ [0,0] + b1)` is zero, so `dW2 = outer(delta2, a1)` is identically
  zero and any implementation "passes".
- **Check a converged network.** Once the model fits, the residual is ~0, so
  every gradient is ~0 and a 10x error is still ~0.

`ch07_backprop.py` hit both and has been rewritten: it checks all four parameter
tensors on a freshly initialised network at a sample with a large residual, and
compares **relative** error so a scale bug cannot hide behind a small magnitude.
Injecting a 10x error, dropping the `relu_grad` factor, flipping a sign, or
perturbing `dW1` by 1% all now fail the check.

## Flagship challenges

| Topic | Book pointer | Files |
|---|---|---|
| 🔍 Find the bug | Ch 5 §Adam | `challenges/ch05_find_the_bug.py` + `ch05_find_the_bug_solution.md` |
| 🏗️ Build nanoGPT in 250 lines | Ch 11 §Transformer block | `reference/nano_gpt.py` |
| 📜 FlashAttention in pure NumPy | Ch 11 §FlashAttention | `reference/flash_attention_numpy.py` |
| 🏛️ KV-cache memory three ways | Ch 11 §KV cache | `reference/kv_cache_archaeology.py` |
| 💸 Plan a 1M-DAU LLM product | Ch 15 §Cost model | `challenges/ch15_plan_1m_dau.py` + `ch15_plan_1m_dau_solution.md` |

## Quick start

```bash
cd code/
python3 ch01_linalg.py
python3 ch11_attention.py
python3 ch16_interpretability.py
python3 ch19_drift_monitoring.py
# ... run any top-level file standalone

python3 reference/flash_attention_numpy.py
python3 reference/kv_cache_archaeology.py
python3 challenges/ch15_plan_1m_dau.py
python3 reference/nano_gpt.py            # tiny smoke test, <1 min on CPU;
                                          # self-downloads Tiny Shakespeare
python3 reference/nano_gpt.py --big      # the real run, minutes on GPU
```
