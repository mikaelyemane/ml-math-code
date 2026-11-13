"""Ch. 12 — Generative Models.

Three small, self-contained generative-model demos that mirror the
chapter derivations:

  1. A 2-D VAE with a 1-D latent, trained by gradient ascent on the ELBO
     with exact gradients through the reparameterisation trick (derived
     by hand and checked against finite differences). The ELBO is scored
     on a fixed held-out set, so the printed curve climbs.
  2. A 1-D Gaussian-mixture EM fit on a synthetic two-component mixture.
  3. DDPM with the chapter's schedule (T = 1000, beta linear from 1e-4 to
     0.02): the closed-form forward marginal checked against the iterated
     chain, then ancestral sampling with an oracle noise predictor that
     recovers the data distribution from pure noise. A shortened schedule
     (T = 200) is run alongside to show the terminal-SNR bias the chapter
     warns about: alpha_bar_T is far from 0, so starting from N(0, I) is
     the wrong prior and the samples come out biased toward the origin.
     The chapter's own shortened schedule (beta_t = 0.02 t / T, T = 100)
     is run too, so its alpha_bar_T = 0.362 and sqrt = 0.601 can be checked.
"""
import numpy as np


# =============================================================
# 1) Variational Autoencoder (linear encoder + tanh decoder)
# =============================================================
#
# Data: points on a line in R^2, x = s * (0.8, 0.5) + noise, with s ~ N(0, 0.6^2).
# One latent dimension is enough to explain them, so a working VAE should push
# the ELBO well above the "ignore z" baseline.  Likelihood p(x|z) = N(decode(z),
# SIGMA_X^2 I); additive constants are dropped from the ELBO.

SIGMA_X = 0.1


def make_line_data(rng, n):
    s = rng.normal(0.0, 0.6, size=n)
    return s[:, None] * np.array([0.8, 0.5]) + rng.normal(0.0, 0.05, size=(n, 2))


def init_vae(rng, input_dim=2, latent_dim=1):
    return {
        "W_mu": rng.normal(0, 0.1, (latent_dim, input_dim)),
        "b_mu": np.zeros(latent_dim),
        "W_lv": rng.normal(0, 0.1, (latent_dim, input_dim)),
        "b_lv": np.zeros(latent_dim),
        "W_dec": rng.normal(0, 0.1, (input_dim, latent_dim)),
        "b_dec": np.zeros(input_dim),
    }


def vae_elbo_and_grads(p, X, eps):
    """Batch-mean ELBO and its exact gradient via the reparameterisation trick.

    X: (n, d) data, eps: (n, k) standard-normal draws.  Because
    z = mu + eps * exp(log_var / 2) is a deterministic function of the encoder
    outputs once eps is fixed, the gradient flows through z to the encoder;
    that is the whole point of the trick (Ch. 12, VAE section).
    """
    n = X.shape[0]
    mu = X @ p["W_mu"].T + p["b_mu"]              # (n, k)
    lv = X @ p["W_lv"].T + p["b_lv"]              # (n, k)
    std = np.exp(0.5 * lv)
    z = mu + eps * std                             # reparameterise
    a = z @ p["W_dec"].T + p["b_dec"]              # (n, d)
    xr = np.tanh(a)

    recon = -0.5 * np.sum((X - xr) ** 2, axis=1) / SIGMA_X**2
    kl = 0.5 * np.sum(mu**2 + np.exp(lv) - 1.0 - lv, axis=1)
    elbo = np.mean(recon - kl)

    g_a = (X - xr) / SIGMA_X**2 * (1.0 - xr**2) / n          # dELBO/da
    g_z = g_a @ p["W_dec"]                                    # dELBO/dz
    g_mu = g_z - mu / n
    g_lv = g_z * eps * 0.5 * std - 0.5 * (np.exp(lv) - 1.0) / n
    grads = {
        "W_dec": g_a.T @ z, "b_dec": g_a.sum(0),
        "W_mu": g_mu.T @ X, "b_mu": g_mu.sum(0),
        "W_lv": g_lv.T @ X, "b_lv": g_lv.sum(0),
    }
    return elbo, grads


def vae_grad_check(rng):
    """Finite-difference check of every parameter gradient (fixed eps)."""
    p = init_vae(rng)
    p = {k: v + rng.normal(0, 0.3, v.shape) for k, v in p.items()}
    X = make_line_data(rng, 8)
    eps = rng.standard_normal((8, 1))
    _, g = vae_elbo_and_grads(p, X, eps)
    worst, h = 0.0, 1e-6
    for k in p:
        for idx in np.ndindex(p[k].shape):
            old = p[k][idx]
            p[k][idx] = old + h; fp, _ = vae_elbo_and_grads(p, X, eps)
            p[k][idx] = old - h; fm, _ = vae_elbo_and_grads(p, X, eps)
            p[k][idx] = old
            num = (fp - fm) / (2 * h)
            worst = max(worst, abs(num - g[k][idx]) / max(1.0, abs(num)))
    return worst


def train_vae(rng, n_steps=200, lr=0.01, batch_size=64, eval_every=20):
    """Minibatch gradient ascent on the ELBO with exact reparameterised gradients.

    The ELBO is reported on a fixed held-out set with fixed eps draws, so the
    printed curve reflects the parameters, not sampling noise.
    """
    params = init_vae(rng)
    X_eval = make_line_data(rng, 2000)
    eps_eval = rng.standard_normal((2000, 1))
    history = []
    for step in range(n_steps + 1):
        if step % eval_every == 0:
            history.append((step, vae_elbo_and_grads(params, X_eval, eps_eval)[0]))
        if step == n_steps:
            break
        X = make_line_data(rng, batch_size)
        _, g = vae_elbo_and_grads(params, X, rng.standard_normal((batch_size, 1)))
        for k in params:
            params[k] = params[k] + lr * g[k]
    return params, history


# =============================================================
# 2) Gaussian-Mixture EM (1-D, two components)
# =============================================================

def gmm_em(x, n_iters=50):
    """EM for a 2-component 1-D Gaussian mixture.  Returns (pi, mu, sigma2)."""
    rng = np.random.default_rng(0)
    pi = np.array([0.5, 0.5])
    mu = np.array([float(x.min()) + 0.1, float(x.max()) - 0.1])
    sigma2 = np.array([float(x.var()), float(x.var())])
    log_lik = []
    for _ in range(n_iters):
        # E-step: responsibilities
        diff = x[:, None] - mu[None, :]
        coef = pi / np.sqrt(2 * np.pi * sigma2)
        like = coef * np.exp(-0.5 * diff ** 2 / sigma2)
        gamma = like / like.sum(axis=1, keepdims=True)
        # M-step
        Nk = gamma.sum(axis=0)
        pi = Nk / Nk.sum()
        mu = (gamma * x[:, None]).sum(axis=0) / Nk
        sigma2 = (gamma * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk
        sigma2 = np.maximum(sigma2, 1e-6)
        log_lik.append(np.log(like.sum(axis=1) + 1e-12).sum())
    return pi, mu, sigma2, log_lik


# =============================================================
# 3) DDPM forward process (closed-form marginal verification)
# =============================================================

def ddpm_schedule(T=1000, beta_min=1e-4, beta_max=0.02):
    """Linear noise schedule (Ho et al. 2020)."""
    beta = np.linspace(beta_min, beta_max, T)
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    return beta, alpha, alpha_bar


def ddpm_forward_marginal(x0, alpha_bar_t, rng):
    """One-shot draw from q(x_t | x_0) using the closed-form Gaussian
    derived in the chapter: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) z."""
    z = rng.normal(size=x0.shape)
    return np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * z


def ddpm_forward_iterated(x0, alpha, T_target, rng):
    """Iteratively apply the one-step transition q(x_t | x_{t-1}). Should
    match ddpm_forward_marginal at t = T_target (within Monte Carlo noise)."""
    x = x0.copy()
    for t in range(T_target):
        x = np.sqrt(alpha[t]) * x + np.sqrt(1 - alpha[t]) * rng.normal(size=x.shape)
    return x


def ddpm_reverse_sample(target_mean, beta, alpha, alpha_bar, n_samples, rng):
    """Reverse-time DDPM sampler with a closed-form oracle eps-predictor.

    For data distribution q(x_0) = N(target_mean, I), the optimal Bayes
    predictor of the diffusion noise given x_t is

        eps^*(x_t, t) = sqrt(1 - alpha_bar_t) * (x_t - sqrt(alpha_bar_t) * mu).

    Derivation: (x_0, x_t) is jointly Gaussian, so E[eps | x_t] follows
    from the standard conditional formula (Ch. 2 §Multivariate Gaussian).

    Plugging into the Ho et al. 2020 reverse update

        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_pred)
                  + sigma_t * z,    z ~ N(0, I),

    samples reconstitute the target distribution from pure noise.
    """
    T = len(beta)
    x = rng.normal(size=(n_samples, 2))  # x_T ~ N(0, I)
    for t in reversed(range(T)):
        eps_pred = np.sqrt(1 - alpha_bar[t]) * (x - np.sqrt(alpha_bar[t]) * target_mean)

        mean = (x - (1 - alpha[t]) / np.sqrt(1 - alpha_bar[t]) * eps_pred) / np.sqrt(alpha[t])
        if t > 0:
            sigma = np.sqrt(beta[t])
            x = mean + sigma * rng.normal(size=x.shape)
        else:
            x = mean
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # ---- VAE training ----
    print(f"VAE gradient check (reparameterised ELBO), max rel err: {vae_grad_check(rng):.2e}")
    _, history = train_vae(rng)
    print("VAE ELBO on a fixed held-out set (should climb):")
    for step, v in history:
        print(f"  step {step:3d}: ELBO = {v:9.3f}")

    # ---- EM on a 1-D mixture ----
    n_per = 200
    data = np.concatenate([
        rng.normal(-2.0, 0.6, n_per),
        rng.normal(2.0, 0.4, n_per),
    ])
    pi, mu, sigma2, ll = gmm_em(data, n_iters=50)
    print(f"\nGMM EM:  pi = {pi.round(3)}  mu = {mu.round(3)}  sigma = {np.sqrt(sigma2).round(3)}")
    print(f"         log-likelihood (final): {ll[-1]:.2f}  (init: {ll[0]:.2f})")

    # ---- DDPM forward marginal: closed form vs. iterated ----
    T = 1000
    beta, alpha, alpha_bar = ddpm_schedule(T)
    t_check = 100
    n_check = 20000
    x0 = rng.normal(0.0, 0.5, size=(n_check, 2)) + np.array([1.0, -1.0])
    closed_form = ddpm_forward_marginal(x0, alpha_bar[t_check - 1], rng)
    iterated = ddpm_forward_iterated(x0, alpha, t_check, rng)
    print(f"\nDDPM forward marginal at t={t_check} of T={T} (closed form vs. iterated chain):")
    print(f"  closed-form  mean = {closed_form.mean(axis=0).round(3)},  std = {closed_form.std(axis=0).round(3)}")
    print(f"  iterated     mean = {iterated.mean(axis=0).round(3)},  std = {iterated.std(axis=0).round(3)}")
    print(f"  expected     mean = sqrt(alpha_bar_t)*E[x0] = {(np.sqrt(alpha_bar[t_check-1]) * np.array([1.0, -1.0])).round(3)}")

    # ---- DDPM reverse sampler (oracle noise predictor) ----
    # Data distribution: N([1, -1], I). With an oracle eps-predictor the reverse
    # process recovers it from x_T ~ N(0, I), provided alpha_bar_T ~ 0.
    target_mean = np.array([1.0, -1.0])
    n_gen = 5000
    print(f"\nDDPM reverse sample (oracle noise predictor, {n_gen} samples, target mean {target_mean}):")
    for T_run in (1000, 200):
        b, a_, ab = ddpm_schedule(T_run)
        samples = ddpm_reverse_sample(target_mean, b, a_, ab, n_gen, rng)
        err = np.linalg.norm(samples.mean(axis=0) - target_mean)
        print(f"  T={T_run:4d}  alpha_bar_T={ab[-1]:.1e}  mean={samples.mean(axis=0).round(3)}  "
              f"std={samples.std(axis=0).round(3)}  |mean error|={err:.3f}")
    print("  T=1000 recovers the target; T=200 leaves alpha_bar_T ~ 0.13, so the N(0, I)")
    print("  starting point is the wrong prior and the sample mean is pulled toward 0.")

    # ---- The chapter's shortened schedule: beta_t = 0.02 t / T, T = 100 ----
    T_book = 100
    b, a_, ab = ddpm_schedule(T_book, beta_min=0.02 / T_book, beta_max=0.02)
    samples = ddpm_reverse_sample(target_mean, b, a_, ab, n_gen, rng)
    err = np.linalg.norm(samples.mean(axis=0) - target_mean)
    print(f"\nChapter schedule beta_t = 0.02 t/T, T={T_book}:")
    print(f"  alpha_bar_T = {ab[-1]:.3f}  sqrt(alpha_bar_T) = {np.sqrt(ab[-1]):.3f}  "
          f"sqrt(1 - alpha_bar_T) = {np.sqrt(1 - ab[-1]):.3f}")
    print(f"  oracle sampler mean = {samples.mean(axis=0).round(3)}  |mean error| = {err:.3f}")
