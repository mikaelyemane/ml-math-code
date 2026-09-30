# Flagship Notebooks

Sixteen Colab-ready notebooks — one per chapter — that mirror the key derivations in
*Math for ML Engineers*. Each notebook uses pure NumPy (plus optional PyTorch where
noted) and produces 2–3 visualisations. No local setup required — open in Google
Colab with one click.

| Notebook | Chapter | Key ideas | Colab |
|----------|---------|-----------|-------|
| `ch01_linear_algebra.ipynb` | Ch. 1 — Foundations | SVD / Eckart–Young low-rank approximation, gradient-check pattern | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch01_linear_algebra.ipynb) |
| `ch02_probability.ipynb` | Ch. 2 — Probability | MAP estimation as shrinkage, MAP↔weight-decay correspondence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch02_probability.ipynb) |
| `ch03_information_theory.ipynb` | Ch. 3 — Information Theory | Entropy/KL/cross-entropy identities, KL as the RLHF/DPO/VAE workhorse | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch03_information_theory.ipynb) |
| `ch04_linear_models.ipynb` | Ch. 4 — Linear Models | Lasso/ISTA exact sparse recovery, soft-thresholding ↔ weight pruning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch04_linear_models.ipynb) |
| `ch05_optimizer_zoo.ipynb` | Ch. 5 — Optimization | GD (5.1), momentum (5.17)–(5.18), Adam (5.36)–(5.37) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch05_optimizer_zoo.ipynb) |
| `ch06_gaussian_processes.ipynb` | Ch. 6 — Kernel Methods | GP posterior fan plot, kernel trick ↔ attention, Bayesian optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch06_gaussian_processes.ipynb) |
| `ch07_xor_backprop.ipynb` | Ch. 7 — Shallow Networks | Manual backprop on XOR, decision-boundary plot, gradient check | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch07_xor_backprop.ipynb) |
| `ch08_mixed_precision.ipynb` | Ch. 8 — Deep Networks | fp16/bf16 overflow and underflow (Mixed-Precision Training section), dynamic loss scaler | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch08_mixed_precision.ipynb) |
| `ch09_conv_from_scratch.ipynb` | Ch. 9 — Convolutions | Conv-as-Toeplitz-matmul, 2-D conv backward, receptive-field growth | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch09_conv_from_scratch.ipynb) |
| `ch10_rnn_lstm_bptt.ipynb` | Ch. 10 — Recurrent Networks | BPTT gradient check, vanishing gradients: RNN vs. LSTM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch10_rnn_lstm_bptt.ipynb) |
| `ch11_nanogpt_attention.ipynb` | Ch. 11 — Attention | SDPA (11.1), causal mask (11.29), RoPE (11.14) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch11_nanogpt_attention.ipynb) |
| `ch12_ddpm_sampler.ipynb` | Ch. 12 — Generative | Forward marginal (12.37), reverse-step mean (12.47), DDIM (12.54) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch12_ddpm_sampler.ipynb) |
| `ch13_dpo_loop.ipynb` | Ch. 13 — RL/Alignment | DPO loss (13.46), implicit reward (13.44), KL monitor (13.40) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch13_dpo_loop.ipynb) |
| `ch14_natural_gradient_lora.ipynb` | Ch. 14 — Frontiers | Fisher information, natural gradient vs. vanilla gradient, LoRA savings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch14_natural_gradient_lora.ipynb) |
| `ch15_roofline_kv_cache.ipynb` | Ch. 15 — Inference Economics | Roofline model, Llama-3-70B KV-cache sizing, GQA, disaggregation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch15_roofline_kv_cache.ipynb) |
| `ch16_superposition_induction.ipynb` | Ch. 16 — Interpretability | Toy superposition phase change, near-orthogonal packing (Thm 16.1), induction circuit + activation patching (Defs. 16.4–16.5) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mikaelyemane/ml-math-code/blob/main/notebooks/ch16_superposition_induction.ipynb) |

## Design principles

- **Mirror the math** — every code cell connects to a numbered equation or named
  result in the book. Chapters with a stable, verified equation numbering (5, 8, 11,
  12, 13, 16) cite it directly; the others cite section ranges rather than guess at
  a number.
- **Self-contained** — NumPy and matplotlib only; PyTorch appears only in Ch. 11's optional GPU section
- **Exercise cells** — each notebook ends with 2 exercises that extend the derivation
- **Failure modes** — each notebook references at least one entry in Appendix D or E
- **One production hook per notebook** — every notebook ties its toy demo to a real
  fact about how modern LLMs are trained or served (LoRA, GQA, weight decay as MAP,
  disaggregated serving, polysemanticity and SAEs, and so on)

## Running locally

```bash
pip install numpy matplotlib jupyter
jupyter notebook notebooks/ch01_linear_algebra.ipynb
```

*(Ch. 6 also uses `scipy.stats` for one plotting cell; Ch. 11's optional GPU section
uses `torch`.)*
