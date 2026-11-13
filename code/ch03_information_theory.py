"""
Ch03 — Information Theory
Covers: entropy, cross-entropy, KL divergence (non-negativity demo),
        mutual information, H(p,q) = H(p) + KL(p||q).
"""
import numpy as np


def entropy(p, eps=1e-12):
    """H(p) = -Σ pᵢ log pᵢ  (nats, natural log)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    return -np.sum(p * np.log(p + eps))


def cross_entropy(p, q, eps=1e-12):
    """H(p, q) = -Σ pᵢ log qᵢ."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / p.sum(); q = q / q.sum()
    return -np.sum(p * np.log(q + eps))


def kl_divergence(p, q, eps=1e-12):
    """KL(p || q) = Σ pᵢ log(pᵢ/qᵢ).  Always ≥ 0 (non-negativity demo below)."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / p.sum(); q = q / q.sum()
    return np.sum(p * np.log((p + eps) / (q + eps)))


def mutual_information(joint_pxy):
    """
    I(X;Y) = KL(p(x,y) || p(x)p(y)) = H(X) + H(Y) - H(X,Y).
    joint_pxy: 2-D array of joint probabilities.
    """
    pxy = joint_pxy / joint_pxy.sum()
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)
    return entropy(px) + entropy(py) - entropy(pxy.ravel())


def gaussian_kl(mu1, sigma1, mu2, sigma2):
    """
    KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) in closed form.
    KL = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    """
    return (np.log(sigma2 / sigma1)
            + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
            - 0.5)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify H(p,q) = H(p) + KL(p||q)
    p = np.array([0.4, 0.35, 0.25])
    q = np.array([0.3, 0.3,  0.4])
    hpq  = cross_entropy(p, q)
    hp   = entropy(p)
    kl   = kl_divergence(p, q)
    print(f"H(p,q)={hpq:.4f}   H(p)+KL={hp+kl:.4f}   match={np.isclose(hpq, hp+kl)}")

    # Non-negativity: KL ≥ 0 for 1000 random distributions
    rng = np.random.default_rng(7)
    kls = [kl_divergence(rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5)))
           for _ in range(1000)]
    print(f"KL ≥ 0 for all 1000 random pairs: {all(k >= -1e-10 for k in kls)}")

    # Uniform has maximum entropy
    n = 6
    uniform = np.ones(n) / n
    peaked  = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
    print(f"\nH(uniform-{n})={entropy(uniform):.4f}  H(peaked)={entropy(peaked):.4f}")
    print(f"H(uniform) = log({n}) = {np.log(n):.4f}")

    # Gaussian KL
    kl_gauss = gaussian_kl(mu1=1.0, sigma1=1.0, mu2=0.0, sigma2=2.0)
    print(f"\nKL(N(1,1) || N(0,4)) = {kl_gauss:.4f}")

    # Mutual information: independent vs correlated joint
    ind = np.outer([0.5, 0.5], [0.5, 0.5])         # independent
    cor = np.array([[0.45, 0.05], [0.05, 0.45]])    # correlated
    print(f"\nI(X;Y) independent={mutual_information(ind):.4f}")
    print(f"I(X;Y) correlated  ={mutual_information(cor):.4f}")
