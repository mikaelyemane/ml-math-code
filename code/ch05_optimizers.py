r"""Ch. 5 — Optimization.

First-order optimisers on an ill-conditioned quadratic. The example is
deliberately stretched (10x curvature in y vs. x) so that the difference
between plain gradient descent, momentum, and Adam shows up in the loss
at steps 10, 50 and 100; all three have converged by step 500.
A Newton step is added for contrast: with the true Hessian, one step
converges exactly.
"""
import numpy as np


def gd(grad_fn, theta0, lr, n_steps):
    """Plain gradient descent: theta_{t+1} = theta_t - lr * grad(theta_t)."""
    theta = theta0.copy()
    history = [theta.copy()]
    for _ in range(n_steps):
        theta -= lr * grad_fn(theta)
        history.append(theta.copy())
    return theta, history


def momentum(grad_fn, theta0, lr, beta, n_steps):
    """EMA-style momentum: v_t = beta*v + (1-beta)*g; theta -= lr*v.

    Note: this is the EMA / Adam-style form, not the classical heavy-ball
    update (v_t = beta*v + g). The two differ by a (1-beta) factor, which
    is absorbed into an effective learning-rate rescaling.
    """
    theta = theta0.copy()
    v = np.zeros_like(theta)
    history = [theta.copy()]
    for _ in range(n_steps):
        g = grad_fn(theta)
        v = beta * v + (1 - beta) * g
        theta -= lr * v
        history.append(theta.copy())
    return theta, history


def adam(grad_fn, theta0, lr, beta1, beta2, eps, n_steps):
    """Adam (Kingma & Ba 2015) with bias-corrected first and second moments.

    The bias correction by (1 - beta^t) is what makes the early-step
    estimates of m and v unbiased (Section 5 / Adam derivation in the book).
    """
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = [theta.copy()]
    for t in range(1, n_steps + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(theta.copy())
    return theta, history


def newton_step(grad_fn, hess_fn, theta0):
    """One Newton step: theta - H^{-1} g. Converges in 1 step on a quadratic."""
    g = grad_fn(theta0)
    H = hess_fn(theta0)
    return theta0 - np.linalg.solve(H, g)


if __name__ == "__main__":
    # f(x,y) = (x-2)^2 + 10*(y-1)^2, minimum at (2, 1), f=0
    def f(theta):
        x, y = theta
        return (x - 2) ** 2 + 10 * (y - 1) ** 2

    def grad_f(theta):
        x, y = theta
        return np.array([2 * (x - 2), 20 * (y - 1)])

    def hess_f(_theta):
        return np.array([[2.0, 0.0], [0.0, 20.0]])

    theta0 = np.array([0.0, 0.0])

    res_gd, hist_gd = gd(grad_f, theta0, lr=0.05, n_steps=500)
    res_mom, hist_mom = momentum(grad_f, theta0, lr=0.05, beta=0.9, n_steps=500)
    res_adam, hist_adam = adam(grad_f, theta0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=500)
    res_newt = newton_step(grad_f, hess_f, theta0)

    checkpoints = (10, 50, 100)
    print("loss at step   " + "  ".join(f"{k:>9d}" for k in checkpoints))
    for name, hist in [("GD      ", hist_gd), ("Momentum", hist_mom), ("Adam    ", hist_adam)]:
        print(f"{name}      " + "  ".join(f"{f(hist[k]):9.2e}" for k in checkpoints))
    print()

    rows = [
        ("GD      ", res_gd),
        ("Momentum", res_mom),
        ("Adam    ", res_adam),
        ("Newton  ", res_newt),
    ]
    for name, theta in rows:
        print(f"{name}  x={theta[0]:.6f}  y={theta[1]:.6f}  loss={f(theta):.2e}")
