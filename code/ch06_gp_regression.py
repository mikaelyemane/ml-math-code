import numpy as np


def rbf_kernel(X1, X2, length_scale, signal_var):
    # K(x,x') = signal_var * exp(-||x-x'||^2 / (2 * l^2))
    X1 = np.atleast_1d(X1)   # shape (n,)
    X2 = np.atleast_1d(X2)   # shape (m,)
    # Outer difference: (n, m)
    sq_dists = (X1[:, None] - X2[None, :]) ** 2
    return signal_var * np.exp(-sq_dists / (2 * length_scale ** 2))


def gp_posterior(X_train, y_train, X_test, kernel_fn, noise_var):
    K_nn = kernel_fn(X_train, X_train) + noise_var * np.eye(len(X_train))
    K_star_n = kernel_fn(X_test, X_train)
    K_star_star = kernel_fn(X_test, X_test)

    L = np.linalg.cholesky(K_nn)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu_star = K_star_n @ alpha

    v = np.linalg.solve(L, K_star_n.T)
    sigma_star = np.sqrt(np.maximum(np.diag(K_star_star) - np.sum(v ** 2, axis=0), 0))

    return mu_star, sigma_star


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_train = 15
    X_train = rng.uniform(-3 * np.pi, 3 * np.pi, n_train)
    y_train = np.sin(X_train) + rng.normal(0, 0.1, n_train)

    X_test = np.linspace(-3 * np.pi, 3 * np.pi, 200)
    y_true = np.sin(X_test)

    kernel_fn = lambda X1, X2: rbf_kernel(X1, X2, length_scale=1.0, signal_var=1.0)
    mu_star, sigma_star = gp_posterior(X_train, y_train, X_test, kernel_fn, noise_var=0.01)

    rmse = np.sqrt(np.mean((mu_star - y_true) ** 2))
    coverage = np.mean(np.abs(y_true - mu_star) <= 2 * sigma_star)

    print(f"Test-grid RMSE (posterior mean vs true sin): {rmse:.4f}")
    print(f"95% coverage (fraction within ±2σ):         {coverage:.4f}")
