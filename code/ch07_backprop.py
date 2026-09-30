import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def mse(pred, target):
    return np.mean((pred - target) ** 2)


def forward(x, W1, b1, W2, b2):
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    return z1, a1, z2


def backward(x, y, z1, a1, z2, W1, W2):
    d_out = y.shape[0]                          # output dimension (not batch size)
    delta2 = 2 * (z2 - y) / d_out               # dL/dz2 for L = mean((z2 - y)**2)
    dW2 = np.outer(delta2, a1)
    db2 = delta2
    delta1 = (W2.T @ delta2) * relu_grad(z1)    # chain rule through ReLU
    dW1 = np.outer(delta1, x)
    db1 = delta1
    return dW1, db1, dW2, db2


def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy(); x_plus.flat[i] += h
        x_minus = x.copy(); x_minus.flat[i] -= h
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


def gradient_check(rng, tol=1e-4):
    """Check backward() against finite differences on an UNTRAINED network.

    Two traps this avoids, both of which make a gradient check silently vacuous:

    1. Never check at x = [0, 0]. With b1 initialised to zeros, z1 = b1 = 0 and
       a1 = relu(0) = 0, so dW2 = outer(delta2, a1) is identically zero. Both
       gradients are zero and the check passes for any implementation at all.

    2. Never check a CONVERGED network. Once the net fits XOR, z2 - y is ~0, so
       delta2 is ~0 and every downstream gradient is ~0 too. Multiplying dW2 by
       ten still "passes", because ten times nothing is nothing.

    So: fresh random weights, non-zero biases, and a sample where the residual is
    large. Now a wrong gradient shows up as a wrong number.
    """
    x = np.array([1.0, 1.0])
    y = np.array([0.0])
    hidden = 8
    W1 = rng.normal(0, 1.0, (hidden, 2))
    b1 = rng.normal(0, 0.5, hidden)          # non-zero: keeps a1 off the ReLU kink
    W2 = rng.normal(0, 1.0, (1, hidden))
    b2 = rng.normal(0, 0.5, 1)

    z1, a1, z2 = forward(x, W1, b1, W2, b2)
    assert abs(z2[0] - y[0]) > 0.1, "residual too small -- check would be vacuous"
    dW1_a, db1_a, dW2_a, db2_a = backward(x, y, z1, a1, z2, W1, W2)

    def loss_with(W1_=None, b1_=None, W2_=None, b2_=None):
        _, _, z2_ = forward(x,
                            W1 if W1_ is None else W1_,
                            b1 if b1_ is None else b1_,
                            W2 if W2_ is None else W2_,
                            b2 if b2_ is None else b2_)
        return mse(z2_, y)

    checks = [
        ("W1", dW1_a, numerical_gradient(
            lambda f: loss_with(W1_=f.reshape(W1.shape)), W1.flatten()).reshape(W1.shape)),
        ("b1", db1_a, numerical_gradient(lambda f: loss_with(b1_=f), b1.copy())),
        ("W2", dW2_a, numerical_gradient(
            lambda f: loss_with(W2_=f.reshape(W2.shape)), W2.flatten()).reshape(W2.shape)),
        ("b2", db2_a, numerical_gradient(lambda f: loss_with(b2_=f), b2.copy())),
    ]

    print("Gradient check on an untrained network (x = [1, 1], residual "
          f"{z2[0] - y[0]:+.3f}):")
    worst = 0.0
    for name, ana, num in checks:
        scale = max(np.max(np.abs(ana)), np.max(np.abs(num)), 1e-12)
        rel = np.max(np.abs(ana - num)) / scale        # RELATIVE error: a 10x bug
        worst = max(worst, rel)                        # cannot hide behind a small scale
        print(f"  {name}: max relative error {rel:.2e}")
    ok = worst < tol
    print(f"All gradients agree to < {tol:g} relative: {ok}  (worst {worst:.2e})")
    return ok


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # XOR dataset: 4 samples, 2 inputs, 1 output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    Y = np.array([[0], [1], [1], [0]], dtype=float)

    hidden = 8
    W1 = rng.normal(0, 1.0, (hidden, 2))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0, (1, hidden))
    b2 = np.zeros(1)

    lr = 0.05
    for step in range(3000):
        dW1_acc = np.zeros_like(W1)
        db1_acc = np.zeros_like(b1)
        dW2_acc = np.zeros_like(W2)
        db2_acc = np.zeros_like(b2)
        for x, y in zip(X, Y):
            z1, a1, z2 = forward(x, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward(x, y, z1, a1, z2, W1, W2)
            dW1_acc += dW1; db1_acc += db1
            dW2_acc += dW2; db2_acc += db2
        W1 -= lr * dW1_acc / 4
        b1 -= lr * db1_acc / 4
        W2 -= lr * dW2_acc / 4
        b2 -= lr * db2_acc / 4

    gradient_check(rng)

    print("\nFinal XOR predictions:")
    for x, y in zip(X, Y):
        _, _, z2 = forward(x, W1, b1, W2, b2)
        print(f"  input={x.astype(int)}  target={int(y[0])}  pred={z2[0]:.4f}")
