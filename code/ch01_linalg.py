"""
Ch01 — Linear Algebra Foundations
Covers: SVD, gradient of quadratic forms, matrix-vector Jacobian.
All equations match the book's notation exactly.
"""
import numpy as np


# ── SVD ──────────────────────────────────────────────────────────────────────

def economy_svd(A):
    """Return U, s, Vt with only r = rank(A) columns in U and rows in Vt."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U, s, Vt


def low_rank_approx(A, k):
    """Best rank-k approximation via truncated SVD (Eckart–Young theorem)."""
    U, s, Vt = economy_svd(A)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def condition_number(A):
    """σ_max / σ_min — measures numerical sensitivity to perturbations."""
    s = np.linalg.svd(A, compute_uv=False)
    return s[0] / s[-1]


# ── Gradient of quadratic forms ───────────────────────────────────────────────

def grad_quadratic(A, w, b):
    """
    f(w) = w^T A w - 2 b^T w
    ∇f(w) = (A + A^T) w - 2b   [book Eq. for symmetric A: 2Aw - 2b]
    """
    return (A + A.T) @ w - 2 * b


def grad_squared_norm(A, w, b):
    """
    f(w) = ||Aw - b||^2
    ∇f(w) = 2 A^T (Aw - b)
    """
    return 2 * A.T @ (A @ w - b)


# ── Jacobian ──────────────────────────────────────────────────────────────────

def numerical_jacobian(f, x, h=1e-5):
    """Finite-difference Jacobian of f: R^n -> R^m at x."""
    fx = f(x)
    m, n = fx.size, x.size
    J = np.zeros((m, n))
    for j in range(n):
        xp = x.copy(); xp[j] += h
        xm = x.copy(); xm[j] -= h
        J[:, j] = (f(xp) - f(xm)) / (2 * h)
    return J


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # ── SVD demo on a 4×3 matrix ──
    A = rng.normal(size=(4, 3))
    U, s, Vt = economy_svd(A)
    print("SVD reconstruction error:", np.max(np.abs(A - U @ np.diag(s) @ Vt)))
    print("Singular values:", np.round(s, 4))
    print("Condition number:", round(condition_number(A), 4))

    # ── Rank-1 vs rank-2 approximation ──
    A5x5 = rng.normal(size=(5, 5))
    for k in [1, 2, 3]:
        Ak = low_rank_approx(A5x5, k)
        err = np.linalg.norm(A5x5 - Ak, 'fro')
        tail = np.sqrt(np.sum(np.linalg.svd(A5x5, compute_uv=False)[k:] ** 2))
        print(f"  rank-{k} Frobenius error={err:.4f}  σ_tail={tail:.4f}  match={np.isclose(err, tail)}")

    # ── Gradient check: ∇||Aw-b||^2 = 2A^T(Aw-b) ──
    n, m = 5, 3
    A2 = rng.normal(size=(n, m))
    w0 = rng.normal(size=m)
    b  = rng.normal(size=n)

    analytical = grad_squared_norm(A2, w0, b)
    def scalar_loss(w): return np.sum((A2 @ w - b) ** 2)
    num2 = numerical_jacobian(lambda w: np.array([scalar_loss(w)]), w0).ravel()
    print(f"\nGradient check  max|analytical - numerical|: {np.max(np.abs(analytical - num2)):.2e}")
