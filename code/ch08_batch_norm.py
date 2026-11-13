import numpy as np


def bn_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    x_norm = (x - mu) / np.sqrt(var + eps)
    y = gamma * x_norm + beta
    cache = (x_norm, mu, var, x, gamma, eps)
    return y, cache


def bn_backward(dy, cache):
    x_norm, mu, var, x, gamma, eps = cache
    B = x.shape[0]

    dgamma = np.sum(dy * x_norm, axis=0)
    dbeta = np.sum(dy, axis=0)

    # 3-term formula for dx (Ioffe & Szegedy, 2015)
    dx_norm = dy * gamma
    dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + eps) ** -1.5, axis=0)
    dmu = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2 * (x - mu), axis=0) / B
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mu) / B + dmu / B

    return dx, dgamma, dbeta


def numerical_grad(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy(); x_plus.flat[i] += h
        x_minus = x.copy(); x_minus.flat[i] -= h
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(3)

    B, D = 4, 3
    x = rng.normal(0, 3, (B, D))
    gamma = rng.uniform(0.5, 1.5, D)
    beta = rng.uniform(-1, 1, D)

    y, cache = bn_forward(x, gamma, beta)
    dy = rng.normal(0, 1, y.shape)   # upstream gradient

    dx_anal, dgamma_anal, dbeta_anal = bn_backward(dy, cache)

    # Numerical gradient w.r.t. x
    def loss_x(x_):
        y_, _ = bn_forward(x_, gamma, beta)
        return np.sum(dy * y_)

    dx_num = numerical_grad(loss_x, x)

    err_dx = np.max(np.abs(dx_anal - dx_num))
    print(f"Max |dx  analytical - numerical|: {err_dx:.2e}  (< 1e-5: {err_dx < 1e-5})")

    def loss_gamma(g):
        y_, _ = bn_forward(x, g, beta)
        return np.sum(dy * y_)

    dgamma_num = numerical_grad(loss_gamma, gamma)
    err_dg = np.max(np.abs(dgamma_anal - dgamma_num))
    print(f"Max |dgamma analytical - numerical|: {err_dg:.2e}  (< 1e-5: {err_dg < 1e-5})")
