"""
Ch09 — Convolutions
Covers: 1-D conv as matrix multiply (Toeplitz view),
        2-D conv forward/backward from scratch, gradient check.
Here conv = cross-correlation, as in the book.
Forward: y = conv(x, w)
Backward: dx = full_conv(dy, flip(w)) (a true convolution with w),
          dw = valid_conv(x, dy)
"""
import numpy as np


# ── 1-D convolution ───────────────────────────────────────────────────────────

def conv1d_toeplitz(x, w):
    """
    1-D valid convolution via explicit Toeplitz matrix.
    x: (n,)  w: (k,)  →  y: (n-k+1,)
    Illustrates that conv IS matrix multiply: y = C_w x
    """
    n, k = len(x), len(w)
    out_len = n - k + 1
    C = np.zeros((out_len, n))
    for i in range(out_len):
        C[i, i:i+k] = w
    return C @ x, C          # return output and the Toeplitz matrix


def conv1d(x, w):
    """Efficient 1-D valid conv without building the full matrix."""
    k = len(w)
    return np.array([np.dot(x[i:i+k], w) for i in range(len(x) - k + 1)])


# ── 2-D convolution ───────────────────────────────────────────────────────────

def conv2d_valid(x, w):
    """
    2-D valid convolution (no padding).
    x: (H, W)   w: (kH, kW)   →  y: (H-kH+1, W-kW+1)
    """
    kH, kW = w.shape
    oH = x.shape[0] - kH + 1
    oW = x.shape[1] - kW + 1
    y  = np.zeros((oH, oW))
    for i in range(oH):
        for j in range(oW):
            y[i, j] = np.sum(x[i:i+kH, j:j+kW] * w)
    return y


def conv2d_backward(x, w, dy):
    """
    Gradients of valid 2-D conv w.r.t. input x and kernel w.
    dx = full_conv(dy, flip(w))   [backprop through conv = transposed conv]
    dw = valid_conv(x, dy)
    """
    kH, kW = w.shape
    # dx: pad dy and cross-correlate with flipped kernel
    pH, pW = kH - 1, kW - 1
    dy_pad = np.pad(dy, ((pH, pH), (pW, pW)))
    w_flip = w[::-1, ::-1]
    dx = conv2d_valid(dy_pad, w_flip)

    # dw: cross-correlate x with dy — each entry dw[i,j] = sum of x patches * dy
    #     equiv. to valid conv(x, dy) since dy is not flipped here
    dw = conv2d_valid(x, dy)
    return dx, dw


# ── Receptive field ───────────────────────────────────────────────────────────

def receptive_field(n_layers, kernel_size, stride=1):
    """
    Receptive field after L conv layers with kernel k and stride s.
    RF_L = 1 + L * (k - 1)   (stride=1 case, book formula)
    """
    rf = 1
    for _ in range(n_layers):
        rf = stride * (rf - 1) + kernel_size
    return rf


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(3)

    # 1-D: Toeplitz == direct conv
    x = rng.normal(size=8)
    w = np.array([1.0, -1.0, 0.5])
    y_mat, C = conv1d_toeplitz(x, w)
    y_dir    = conv1d(x, w)
    print("1-D conv == Toeplitz matrix-multiply:", np.allclose(y_mat, y_dir))
    print("Toeplitz matrix C_w:\n", C.round(2))

    # 2-D: gradient check
    H, W, kH, kW = 5, 5, 3, 3
    x2 = rng.normal(size=(H, W))
    w2 = rng.normal(size=(kH, kW))

    def loss2d(x_flat, w_flat):
        return 0.5 * np.sum(conv2d_valid(x_flat.reshape(H, W),
                                          w_flat.reshape(kH, kW))**2)

    y2 = conv2d_valid(x2, w2)
    dy = y2                                        # dL/dy = y for loss = 0.5||y||^2
    dx_an, dw_an = conv2d_backward(x2, w2, dy)

    h = 1e-5
    dw_num = np.zeros_like(w2)
    for i in range(kH):
        for j in range(kW):
            wp = w2.copy(); wp[i, j] += h
            wm = w2.copy(); wm[i, j] -= h
            dw_num[i, j] = (loss2d(x2, wp) - loss2d(x2, wm)) / (2 * h)

    print(f"\n2-D kernel gradient check max err: {np.max(np.abs(dw_an - dw_num)):.2e}")

    dx_num = np.zeros_like(x2)
    for i in range(H):
        for j in range(W):
            xp = x2.copy(); xp[i, j] += h
            xm = x2.copy(); xm[i, j] -= h
            dx_num[i, j] = (loss2d(xp, w2) - loss2d(xm, w2)) / (2 * h)

    print(f"2-D input gradient check max err:  {np.max(np.abs(dx_an - dx_num)):.2e}")

    # Receptive field growth
    for L in [1, 3, 5, 10]:
        print(f"  RF with L={L:2d} layers, k=3: {receptive_field(L, 3)}")
