"""
Ch02 — Probability & Statistics
Covers: MLE for Gaussian, MAP with Gaussian prior, Bayes posterior,
        multivariate Gaussian PDF and conditional.
"""
import numpy as np


# ── Univariate Gaussian MLE ───────────────────────────────────────────────────

def gaussian_mle(x):
    """MLE estimates: μ̂ = mean(x), σ̂² = mean((x - μ̂)²)."""
    mu = x.mean()
    sigma2 = np.mean((x - mu) ** 2)
    return mu, sigma2


# ── MAP with Gaussian prior on μ ─────────────────────────────────────────────

def gaussian_map(x, mu0, tau2, sigma2):
    """
    Prior:      μ ~ N(μ₀, τ²)
    Likelihood: xᵢ | μ ~ N(μ, σ²)
    Posterior:  μ | x ~ N(μ_post, σ_post²)
    where σ_post² = 1 / (n/σ² + 1/τ²),  μ_post = σ_post²(Σxᵢ/σ² + μ₀/τ²)
    Special case of ridge (MAP with Gaussian prior) with an intercept-only model:
    MAP = (precision-weighted) average of prior and data.
    """
    n = len(x)
    prec_post = n / sigma2 + 1 / tau2
    sigma2_post = 1 / prec_post
    mu_post = sigma2_post * (x.sum() / sigma2 + mu0 / tau2)
    return mu_post, sigma2_post


# ── Multivariate Gaussian ────────────────────────────────────────────────────

def mvn_logpdf(x, mu, Sigma):
    """Log-density of N(mu, Sigma) at x."""
    d = len(mu)
    L = np.linalg.cholesky(Sigma)
    diff = x - mu
    z = np.linalg.solve(L, diff)          # L⁻¹(x - μ)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + z @ z)


def mvn_conditional(mu, Sigma, x2, idx1, idx2):
    """
    Conditional of x₁ | x₂ = x2 when [x₁,x₂] ~ N(mu, Sigma).
    Returns (mu_cond, Sigma_cond).
    idx1, idx2: index arrays for the two blocks.
    """
    mu1, mu2 = mu[idx1], mu[idx2]
    S11 = Sigma[np.ix_(idx1, idx1)]
    S12 = Sigma[np.ix_(idx1, idx2)]
    S22 = Sigma[np.ix_(idx2, idx2)]
    S22_inv = np.linalg.inv(S22)
    mu_cond    = mu1 + S12 @ S22_inv @ (x2 - mu2)
    Sigma_cond = S11 - S12 @ S22_inv @ S12.T
    return mu_cond, Sigma_cond


# ── Bayesian coin flip ────────────────────────────────────────────────────────

def beta_posterior(heads, tails, alpha0=1.0, beta0=1.0):
    """
    Beta-Bernoulli conjugate model.
    Prior: θ ~ Beta(α₀, β₀).  Posterior: θ | data ~ Beta(α₀+heads, β₀+tails).
    """
    return alpha0 + heads, beta0 + tails


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # MLE
    true_mu, true_sigma2 = 3.0, 2.0
    x = rng.normal(true_mu, np.sqrt(true_sigma2), 500)
    mu_hat, s2_hat = gaussian_mle(x)
    print(f"MLE  μ̂={mu_hat:.3f} (true={true_mu})  σ̂²={s2_hat:.3f} (true={true_sigma2})")

    # MAP: strong prior pulls estimate toward μ₀=0
    for tau2 in [0.1, 1.0, 100.0]:
        mu_map, _ = gaussian_map(x, mu0=0.0, tau2=tau2, sigma2=true_sigma2)
        print(f"  MAP τ²={tau2:5.1f}  μ_post={mu_map:.3f}")

    # Multivariate conditional
    mu = np.array([1.0, 2.0, 3.0])
    Sigma = np.array([[2.0, 1.0, 0.5],
                      [1.0, 3.0, 1.0],
                      [0.5, 1.0, 2.0]])
    x2_obs = np.array([2.5, 3.5])
    mu_c, S_c = mvn_conditional(mu, Sigma, x2_obs, idx1=[0], idx2=[1, 2])
    print(f"\nConditional x₁ | x₂=x₂_obs:  μ={mu_c.round(3)}  σ²={S_c.round(3)}")

    # Beta-Bernoulli
    a, b = beta_posterior(heads=14, tails=6, alpha0=1, beta0=1)
    print(f"\nBeta posterior after 14H/6T: Beta({a},{b})  mode={(a-1)/(a+b-2):.3f}  MLE={14/20:.3f}")
