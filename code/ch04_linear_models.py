"""
Ch04 — Linear Models
Covers: normal equations, Ridge regression (=MAP with Gaussian prior),
        logistic regression via gradient descent, softmax cross-entropy gradient.
"""
import numpy as np


# ── Linear regression ─────────────────────────────────────────────────────────

def normal_equations(X, y):
    """w* = (X^T X)^{-1} X^T y — closed-form OLS solution."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def ridge(X, y, lam):
    """
    Ridge / L2 regression: w* = (X^T X + λI)^{-1} X^T y.
    Equivalent to MAP under Gaussian prior w ~ N(0, (1/λ)I).
    """
    n, d = X.shape
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def soft_threshold(x, tau):
    """Element-wise soft-thresholding operator S_tau(x) = sign(x) * max(|x|-tau, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def lasso_ista(X, y, lam, n_iters=500):
    """
    ISTA (Iterative Soft-Thresholding Algorithm) for Lasso:
        min_w  0.5 ||X w - y||^2 + lam * ||w||_1.
    Step size eta = 1 / L where L = lambda_max(X^T X) is the Lipschitz
    constant of the smooth-part gradient. The prox of the L1 norm is
    soft-thresholding (see Chapter 4 derivation).
    """
    n, d = X.shape
    L = np.linalg.norm(X, ord=2) ** 2  # spectral-norm squared
    eta = 1.0 / L
    w = np.zeros(d)
    for _ in range(n_iters):
        grad = X.T @ (X @ w - y)
        w = soft_threshold(w - eta * grad, eta * lam)
    return w


# ── Logistic regression ───────────────────────────────────────────────────────

def sigmoid(z):
    """Numerically stable sigmoid: clips large inputs."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))


def logistic_loss(w, X, y):
    """Binary cross-entropy loss."""
    p = sigmoid(X @ w)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic_grad(w, X, y):
    """∇L(w) = (1/n) X^T (p̂ - y)  — closed form, no loops."""
    p = sigmoid(X @ w)
    return X.T @ (p - y) / len(y)


def logistic_fit(X, y, lr=0.1, n_steps=500):
    w = np.zeros(X.shape[1])
    for _ in range(n_steps):
        w -= lr * logistic_grad(w, X, y)
    return w


# ── Softmax regression ────────────────────────────────────────────────────────

def softmax(Z):
    """Numerically stable row-wise softmax."""
    Z = Z - Z.max(axis=1, keepdims=True)   # subtract max per row
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def softmax_ce_loss(W, X, y):
    """Cross-entropy loss for softmax regression. W shape (d, K)."""
    logits = X @ W                          # (n, K)
    P = softmax(logits)
    n = len(y)
    return -np.mean(np.log(P[np.arange(n), y] + 1e-12))


def softmax_ce_grad(W, X, y):
    """
    ∇_{w_k} L = (1/n) Σᵢ (p̂ᵢₖ - 1[yᵢ=k]) xᵢ
    K is inferred from W.shape[1]. Returns dW of same shape as W.
    """
    n = len(y)
    logits = X @ W
    P = softmax(logits)          # (n, K)
    P[np.arange(n), y] -= 1.0   # subtract 1 from true-class column
    return X.T @ P / n           # (d, K)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(1)

    # Ridge vs OLS on noisy data
    n, d = 50, 10
    w_true = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = X @ w_true + 0.5 * rng.normal(size=n)

    w_ols   = normal_equations(X, y)
    w_ridge = ridge(X, y, lam=1.0)
    print("OLS   ||w - w_true||:", np.linalg.norm(w_ols   - w_true).round(4))
    print("Ridge ||w - w_true||:", np.linalg.norm(w_ridge - w_true).round(4))

    # Lasso / ISTA: sparse recovery with most true coefficients zero.
    d_sparse = 20
    n_sparse = 60
    w_sparse_true = np.zeros(d_sparse)
    w_sparse_true[[0, 5, 12]] = [3.0, -2.0, 1.5]
    X_sp = rng.normal(size=(n_sparse, d_sparse))
    y_sp = X_sp @ w_sparse_true + 0.1 * rng.normal(size=n_sparse)
    w_lasso = lasso_ista(X_sp, y_sp, lam=2.0, n_iters=1000)
    nnz = int(np.sum(np.abs(w_lasso) > 1e-3))
    print(f"Lasso/ISTA recovered nonzeros: {nnz} (true: 3)")
    print(f"Lasso ||w - w_true||: {np.linalg.norm(w_lasso - w_sparse_true):.4f}")

    # Logistic regression on linearly separable data
    X_bin = np.vstack([rng.normal([2, 2], 0.5, (40, 2)),
                       rng.normal([-2, -2], 0.5, (40, 2))])
    y_bin = np.array([1]*40 + [0]*40, dtype=float)
    X_bin = np.hstack([np.ones((80, 1)), X_bin])   # add bias
    w_log = logistic_fit(X_bin, y_bin, lr=0.5, n_steps=200)
    acc   = ((sigmoid(X_bin @ w_log) > 0.5) == y_bin).mean()
    print(f"\nLogistic regression accuracy: {acc:.2%}")

    # Gradient check for softmax CE grad
    K, d2 = 3, 4
    n2 = 20
    X2 = rng.normal(size=(n2, d2))
    y2 = rng.integers(0, K, size=n2)
    W0 = rng.normal(size=(d2, K)) * 0.1

    def loss_fn(w_flat):
        return softmax_ce_loss(w_flat.reshape(d2, K), X2, y2)

    analytical = softmax_ce_grad(W0, X2, y2).ravel()
    h = 1e-5
    numerical  = np.zeros_like(W0.ravel())
    for i in range(W0.size):
        wp = W0.ravel().copy(); wp[i] += h
        wm = W0.ravel().copy(); wm[i] -= h
        numerical[i] = (loss_fn(wp) - loss_fn(wm)) / (2 * h)
    print(f"Softmax grad check max error: {np.max(np.abs(analytical - numerical)):.2e}")
