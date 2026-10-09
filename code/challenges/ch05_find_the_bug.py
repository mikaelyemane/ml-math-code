"""Find the bug --- Chapter 5, Optimization.

A benign convex problem: full-batch least-squares regression, 400 samples,
100 features, no regularisation, no learning-rate schedule, no numerical
cliffs. Adam should walk it down to the noise floor (~5e-5) without drama.

It does not. The loss starts at 24.2, and the very first update sends it to
1.1e4 --- 460x *worse* than doing nothing at all. It thrashes for forty-odd
steps, then grinds back down and eventually reaches the floor anyway, which is
what makes this bug so good at surviving code review.

Exactly one character in `adam_step` is wrong. The bug is in the bias-correction
algebra derived in Section 5.5.3 --- not in the data, the learning rate, or the
loss. The fix is a single character.

    $ python3 ch05_find_the_bug.py

Your job:

  1. Run it. Reproduce the divergence.
  2. Read the step-1 diagnostics it prints. Adam's update is bounded by roughly
     the learning rate, by construction --- that is the whole point of dividing
     by sqrt(v_hat). Check whether it is. The ratio you get back is the clue,
     and it is a suspiciously round number.
  3. Work out which of the two bias corrections could produce exactly that
     ratio at t=1. Fix the character. Re-run.

Solution and full walk-through: ch05_find_the_bug_solution.md.
Do not open it until you have the ratio from step 2.

This is not a toy. The same bug has shipped in production training frameworks
more than once. It is invisible in review --- the line looks right, the shapes
are right, the loss stays finite, and on a short run with a small learning rate
the model still converges. It only bites when the first few steps matter, which
is exactly when you are least likely to be watching.
"""

import numpy as np

BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


def make_problem(n=400, d=100, x_scale=10.0, w_scale=0.05, noise=0.00707, seed=0):
    """Well-conditioned least squares. Irreducible MSE is noise**2 ~ 5e-5."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * x_scale
    w_true = rng.normal(size=d) * w_scale
    y = X @ w_true + noise * rng.normal(size=n)
    return X, y


def loss_and_grad(X, y, w):
    n = X.shape[0]
    resid = X @ w - y
    return float(np.mean(resid**2)), (2.0 / n) * (X.T @ resid)


def adam_step(w, m, v, g, t, lr):
    """One Adam update.

    m, v    first and second raw moment EMAs
    m_hat   bias-corrected first moment
    v_hat   bias-corrected second moment
    t       1-based step count

    One of the four lines below does not match the derivation in Section 5.5.3.
    """
    m = BETA1 * m + (1.0 - BETA1) * g
    v = BETA2 * v + (1.0 - BETA2) * (g * g)

    m_hat = m / (1.0 - BETA2**t)
    v_hat = v / (1.0 - BETA2**t)

    w = w - lr * m_hat / (np.sqrt(v_hat) + EPS)
    return w, m, v, m_hat, v_hat


def train(steps=400, lr=0.01, verbose=True):
    X, y = make_problem()
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)

    initial_loss, _ = loss_and_grad(X, y, w)
    history = []

    for t in range(1, steps + 1):
        _, g = loss_and_grad(X, y, w)
        w, m, v, m_hat, v_hat = adam_step(w, m, v, g, t, lr)

        if verbose and t == 1:
            j = int(np.argmax(np.abs(g)))
            update = lr * m_hat[j] / (np.sqrt(v_hat[j]) + EPS)
            print(f"step-1 diagnostics, largest-gradient coordinate (j={j}):")
            print(f"  g[j]           = {g[j]:+.6e}")
            print(f"  m[j]           = {m[j]:+.6e}    = (1-b1) * g[j]")
            print(f"  m_hat[j]       = {m_hat[j]:+.6e}")
            print(f"  sqrt(v_hat[j]) = {np.sqrt(v_hat[j]):+.6e}")
            print(f"  update         = {update:+.6e}")
            print(f"  |update| / lr  = {abs(update) / lr:.1f}")
            print("                   ^ Adam's step is bounded by ~lr by "
                  "construction. This is not 1.")
            print()

        history.append(float(np.mean((X @ w - y) ** 2)))

    return initial_loss, history


def main():
    initial_loss, history = train()

    print(f"loss before any update: {initial_loss:.4e}")
    print()
    print("loss trajectory")
    for t in (1, 2, 3, 5, 10, 25, 50, 100, 200, 400):
        marker = "  <-- worse than doing nothing" if history[t - 1] > initial_loss else ""
        print(f"  step {t:4d}   {history[t - 1]:.4e}{marker}")
    print()

    peak = max(history)
    final = history[-1]
    overshot = peak > initial_loss
    ripple = max((history[i + 1] / history[i]
                  for i in range(len(history) - 1) if history[i + 1] > history[i]),
                 default=1.0)

    print(f"peak loss        : {peak:.3e}   ({peak / initial_loss:.0f}x the "
          "starting loss)" if overshot else f"peak loss        : {peak:.3e}")
    print(f"final loss       : {final:.3e}   (irreducible noise floor ~5e-5)")
    print(f"largest increase : {ripple:.3f}x")
    print()

    if not overshot and final < 1e-4 and ripple < 1.10:
        print("FIXED. Smooth descent to the noise floor, never worse than the "
              "starting point.")
        print("(The residual few-percent ripple is Adam's own floor "
              "oscillation -- see the")
        print(" 'Adam Does Not Always Converge' pitfall later in the chapter. "
              "It is not a bug.)")
    else:
        print("DIVERGED. On a convex problem with a correctly scaled optimiser, "
              "the loss should")
        print("never exceed its starting value. Re-read adam_step against the "
              "bias-correction")
        print("derivation. Both corrections use the same exponent base right "
              "now -- should they?")


if __name__ == "__main__":
    main()
