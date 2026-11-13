"""
Ch14 — Frontiers: Natural Gradient & Score Matching
Covers: Fisher information matrix, natural gradient step,
        score matching objective (denoising), LoRA parameter count demo.
"""
import numpy as np


# ── Fisher information ─────────────────────────────────────────────────────────

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def fisher_categorical(logits, n_samples=2000, rng=None):
    """
    Monte Carlo (true) Fisher for a categorical distribution p(y|x; θ) = softmax(logits).
    F ≈ E[∇log p · ∇log p^T]  (outer-product form).
    logits: (K,)  →  F: (K, K)
    """
    if rng is None: rng = np.random.default_rng(0)
    K = len(logits)
    p = softmax(logits)
    F = np.zeros((K, K))
    ys = rng.choice(K, size=n_samples, p=p)
    for y in ys:
        onehot = np.zeros(K); onehot[y] = 1.0
        g = onehot - p                          # ∇log p(y|θ) for softmax model
        F += np.outer(g, g)
    return F / n_samples


def natural_gradient_step(grad, F, lr, eps=1e-4):
    """
    θ ← θ - lr · F⁻¹ g   (natural gradient = steepest descent in KL geometry).
    Adds eps·I for numerical stability (damping).
    """
    K = len(grad)
    F_reg = F + eps * np.eye(K)
    return -lr * np.linalg.solve(F_reg, grad)


# ── Denoising score matching ──────────────────────────────────────────────────

def denoise_score_matching_loss(score_fn, x_clean, sigma, n_samples=200, rng=None):
    """
    DSM objective: E_{x̃}[||s_θ(x̃) + (x̃-x)/σ²||²]
    where x̃ = x + σε,  ε ~ N(0,I).
    score_fn: callable R^d → R^d.
    Returns scalar loss.
    """
    if rng is None: rng = np.random.default_rng(0)
    d = len(x_clean)
    eps    = rng.normal(size=(n_samples, d))
    x_noisy = x_clean + sigma * eps        # (n_samples, d)
    target  = -eps / sigma                 # = -(x̃ - x)/σ² (conditional score of q(x̃|x))
    loss = 0.0
    for i in range(n_samples):
        s = score_fn(x_noisy[i])
        loss += np.sum((s - target[i])**2)
    return loss / n_samples


# ── LoRA parameter count ───────────────────────────────────────────────────────

def lora_params(d, r):
    """
    LoRA adds W = W₀ + BA where B∈R^{d×r}, A∈R^{r×d}.
    Trainable params: 2dr  vs  full d² for W.
    """
    full   = d * d
    lora   = 2 * d * r
    saving = 1 - lora / full
    return {"full": full, "lora": lora, "compression": f"{saving:.1%}"}


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(13)

    # ── Natural gradient vs vanilla gradient on a 3-class toy problem ──
    K = 3
    theta = np.array([1.0, 0.5, -0.5])      # logits
    # True distribution: class 0 has all mass
    y_true = np.array([1.0, 0.0, 0.0])
    p = softmax(theta)
    grad_ce = p - y_true                     # ∇ cross-entropy = p̂ - y

    F = fisher_categorical(theta, n_samples=5000, rng=rng)

    delta_vanilla = -0.1 * grad_ce
    delta_natural = natural_gradient_step(grad_ce, F, lr=0.1)

    print("Vanilla gradient step  Δθ:", delta_vanilla.round(4))
    print("Natural gradient step  Δθ:", delta_natural.round(4))

    theta_v = theta + delta_vanilla
    theta_n = theta + delta_natural
    print(f"p(y=0) before: {p[0]:.4f}")
    print(f"p(y=0) after vanilla: {softmax(theta_v)[0]:.4f}")
    print(f"p(y=0) after natural: {softmax(theta_n)[0]:.4f}  (natural moves faster toward target)")

    # ── Denoising score matching (toy Gaussian data) ──
    x0  = np.array([2.0, -1.0])
    sigma = 0.5
    # For Gaussian N(x0, σ²I), true score = -(x - x0)/σ²
    true_score = lambda x: -(x - x0) / sigma**2
    loss_true  = denoise_score_matching_loss(true_score, x0, sigma, rng=rng)
    zero_score = lambda x: np.zeros_like(x)
    loss_zero  = denoise_score_matching_loss(zero_score, x0, sigma, rng=rng)
    print(f"\nDSM loss  true score: {loss_true:.4f}  (should be small)")
    print(f"DSM loss  zero score: {loss_zero:.4f}  (should be larger)")

    # ── LoRA savings ──
    print("\nLoRA parameter savings:")
    for d, r in [(768, 8), (4096, 8), (4096, 64)]:
        info = lora_params(d, r)
        print(f"  d={d:5d} r={r:3d}: full={info['full']:>10,}  LoRA={info['lora']:>6,}  saved={info['compression']}")
