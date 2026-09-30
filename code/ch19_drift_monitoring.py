"""Ch. 19 --- Technical Debt and Observability: drift monitoring with PSI.

Runnable versions of the chapter's two listings (the PSI drift check and the
empirical-null threshold), plus every number the chapter derives from them:

  1. The PSI listing on a synthetic `tenure_days` feature.
  2. PSI is the Jeffreys divergence (eq:psi_jeffreys): the binned value climbs
     toward the exact delta^2 as the bins are refined, which is why a fixed 0.1
     band means something different at every bin count.
  3. Prediction-distribution shift: the same psi() on logged predicted
     probabilities, a real shift against a no-drift control.
  4. The empirical null: consecutive-day pairs vs same-weekday pairs on a daily
     feature with a weekend effect and no incident.

Pure NumPy. Run:  python3 ch19_drift_monitoring.py
"""
import math

import numpy as np

# =============================================================================
# The chapter's listings, verbatim apart from the demo data
# =============================================================================

# PSI rule-of-thumb bands (conventions, not laws):
#   < 0.1 stable | 0.1-0.25 investigate | > 0.25 act


def reference_edges(train_x, n_bins=10):
    """Interior quantile edges of the training-time
    reference, fixed once and reused for every window."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(train_x, qs))


def bin_counts(x, edges):
    idx = np.searchsorted(edges, x, side="right")
    return np.bincount(idx, minlength=len(edges) + 1)


def psi(ref_counts, cur_counts, eps=1e-6):
    """PSI of two histograms on the same edges."""
    r = np.maximum(ref_counts / ref_counts.sum(), eps)
    c = np.maximum(cur_counts / cur_counts.sum(), eps)
    r, c = r / r.sum(), c / c.sum()  # eps: no log(0)
    return float(np.sum((c - r) * np.log(c / r)))


def alert_threshold(windows, lag, pct=99.0, min_pairs=100):
    """Empirical-null PSI threshold for one feature.

    windows: bin counts per window on the reference
             edges, oldest first, from a clean period.
    lag:     windows between the two sides of a pair
             (7 for daily windows, week-over-week).
    """
    null = np.array([psi(windows[i], windows[i + lag])
                     for i in range(len(windows) - lag)])
    if null.size < min_pairs:  # tail percentile = noise
        raise ValueError(f"{null.size} null pairs")
    return float(np.percentile(null, pct))


def band(score):
    return "stable" if score < 0.1 else ("investigate" if score <= 0.25 else "act")


# =============================================================================
# Helpers
# =============================================================================

def std_normal_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def std_normal_ppf(p, tol=1e-12):
    """Inverse CDF by bisection (keeps the file NumPy-only)."""
    lo, hi = np.full_like(p, -40.0), np.full_like(p, 40.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        below = std_normal_cdf(mid) < p
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        if np.max(hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


def rule(title):
    print("\n" + "=" * 72 + "\n" + title + "\n" + "=" * 72)


# =============================================================================
# Demos
# =============================================================================

def demo_listing(rng):
    rule("1. The PSI listing on a synthetic tenure_days feature")
    train = rng.gamma(shape=2.0, scale=180.0, size=50_000)       # training reference
    same = rng.gamma(shape=2.0, scale=180.0, size=5_000)         # a healthy day
    newer = rng.gamma(shape=2.0, scale=140.0, size=5_000)        # an acquisition push: newer users
    edges = reference_edges(train)
    ref = bin_counts(train, edges)
    for name, today in (("healthy day", same), ("acquisition push", newer)):
        score = psi(ref, bin_counts(today, edges))
        print(f"  {name:<18} PSI = {score:.4f}   ({band(score)})")


def demo_jeffreys():
    rule("2. PSI is the Jeffreys divergence (eq:psi_jeffreys)")
    delta = 0.25
    print(f"  N(0,1) vs N({delta},1), equal-probability bins of the reference, exact bin masses")
    print(f"  exact Jeffreys divergence KL(c||r) + KL(r||c) = delta^2 = {delta**2:.4f}")
    for k in (2, 4, 10, 100, 1000):
        e = std_normal_ppf(np.linspace(0, 1, k + 1)[1:-1])
        cdf_r = np.concatenate([[0.0], std_normal_cdf(e), [1.0]])
        cdf_c = np.concatenate([[0.0], std_normal_cdf(e - delta), [1.0]])
        r, c = np.diff(cdf_r), np.diff(cdf_c)
        print(f"    {k:>5} bins   PSI = {np.sum((c - r) * np.log(c / r)):.4f}")
    print("  The binned value approaches delta^2 from below: a fixed 0.1 band is a looser")
    print("  standard for a coarsely binned feature than for a finely binned one.")


def demo_prediction_shift(rng):
    rule("3. Prediction-distribution shift: the same psi() on p(y_hat)")
    n = 5_000
    edges = np.linspace(0, 1, 11)[1:-1]            # ten equal-width probability bins
    ref = rng.beta(2.0, 5.0, n)                     # healthy week, mean 2/7 = 0.286
    cur = rng.beta(2.5, 4.5, n)                     # current week, mean 5/14 = 0.357
    ctl = rng.beta(2.0, 5.0, n)                     # no drift, sampling noise only
    shift = psi(bin_counts(ref, edges), bin_counts(cur, edges))
    noise = psi(bin_counts(ref, edges), bin_counts(ctl, edges))
    print(f"  reference mean {ref.mean():.3f}   current mean {cur.mean():.3f}   ({n:,} scored examples each)")
    print(f"  shifted week   PSI = {shift:.3f}   ({band(shift)})")
    print(f"  no-drift ctrl  PSI = {noise:.3f}   ({band(noise)};"
          f" expected ~2(K-1)/n = {2 * 9 / n:.4f} from sampling noise alone)")
    # How much of that is this one draw?
    reps = np.array([[psi(bin_counts(r, edges), bin_counts(c, edges)) for r, c in
                      ((rng.beta(2, 5, n), rng.beta(2.5, 4.5, n)), (rng.beta(2, 5, n), rng.beta(2, 5, n)))]
                     for _ in range(200)])
    print(f"  over 200 repeats: shift {reps[:, 0].mean():.3f} +/- {reps[:, 0].std():.3f},"
          f"  control {reps[:, 1].mean():.4f} +/- {reps[:, 1].std():.4f}")


def demo_empirical_null(rng):
    rule("4. Empirical null: which pairs you compare sets the threshold")
    weeks, n, weekend_shift = 40, 5_000, 0.6
    print(f"  {weeks} weeks of a daily feature, {n:,} rows/day, N(0,1) on weekdays and")
    print(f"  N({weekend_shift},1) on weekends, no incident. Ten reference-quantile bins.")
    mu = [0.0] * 5 + [weekend_shift] * 2
    train = np.concatenate([rng.normal(mu[d % 7], 1.0, n) for d in range(28)])
    edges = reference_edges(train)
    windows = [bin_counts(rng.normal(mu[d % 7], 1.0, n), edges) for d in range(weeks * 7)]

    dod = np.array([psi(windows[i], windows[i + 1]) for i in range(len(windows) - 1)])
    wow = np.array([psi(windows[i], windows[i + 7]) for i in range(len(windows) - 7)])
    t_dod = alert_threshold(windows, lag=1)
    t_wow = alert_threshold(windows, lag=7)
    print(f"  consecutive-day pairs   median {np.median(dod):.4f}   99th pct {t_dod:.3f}")
    print(f"  same-weekday pairs      median {np.median(wow):.4f}   99th pct {t_wow:.4f}")
    print(f"  threshold ratio         {t_dod / t_wow:.0f}x, same data, different pairing")
    print(f"  day-over-day monitor vs week-over-week threshold: {np.mean(dod > t_wow):.0%} of healthy"
          " comparisons fire")
    print(f"  day-over-day monitor vs the 0.1 band:             {np.mean(dod > 0.1):.0%} fire"
          " (the Fri->Sat and Sun->Mon pairs, 2 of 7)")
    print(f"  week-over-week comparisons ever reaching 0.1:     {int(np.sum(wow >= 0.1))} of {wow.size}")
    try:
        alert_threshold(windows[:60], lag=7)
    except ValueError as err:
        print(f"  min_pairs guard on 60 days of history: ValueError({err})")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo_listing(rng)
    demo_jeffreys()
    demo_prediction_shift(rng)
    demo_empirical_null(rng)
