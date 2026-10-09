"""
Ch10 — Recurrent Networks (RNN, LSTM, BPTT)
Covers: vanilla RNN forward/BPTT gradient, LSTM forward with all gates,
        gradient check confirming analytic BPTT equals finite differences.
"""
import numpy as np


# ── Vanilla RNN ───────────────────────────────────────────────────────────────

def rnn_step(x_t, h_prev, Wx, Wh, b):
    """h_t = tanh(Wx x_t + Wh h_{t-1} + b)."""
    return np.tanh(Wx @ x_t + Wh @ h_prev + b)


def rnn_forward(X, h0, Wx, Wh, b, Wy, by):
    """
    Forward pass over sequence X: (T, d_x).
    Returns hidden states H: (T, d_h) and outputs Y: (T, d_y).
    """
    T, _ = X.shape
    d_h  = h0.shape[0]
    H    = np.zeros((T, d_h))
    h    = h0.copy()
    for t in range(T):
        h     = rnn_step(X[t], h, Wx, Wh, b)
        H[t]  = h
    Y = H @ Wy.T + by                       # output projection
    return H, Y


def rnn_bptt(X, Y_target, H, h0, Wx, Wh, b, Wy, by):
    """
    Backpropagation through time (BPTT).
    Loss = (1/T) Σ_t ||y_t - Y_target_t||^2 / 2
    Returns gradients dWx, dWh, db, dWy, dby.
    """
    T, d_x = X.shape
    d_h    = h0.shape[0]
    H_full = np.vstack([h0[np.newaxis, :], H])  # (T+1, d_h) — H_full[0] = h0

    Y    = H @ Wy.T + by
    # dY is the gradient of  L = 0.5 * mean((Y - Y_target)**2).
    # d/dY [0.5*(Y-Y_target)**2 / N]  =  (Y - Y_target) / N,  N = Y.size.
    dY   = (Y - Y_target) / Y.size              # (T, d_y)

    dWy  = dY.T @ H                             # (d_y, d_h)
    dby  = dY.sum(axis=0)                        # (d_y,)

    dWx  = np.zeros_like(Wx)
    dWh  = np.zeros_like(Wh)
    db   = np.zeros_like(b)
    dh_next = np.zeros(d_h)

    for t in reversed(range(T)):
        dh = Wy.T @ dY[t] + dh_next             # gradient into h_t
        dtanh = (1 - H[t] ** 2) * dh            # through tanh
        db   += dtanh
        dWx  += np.outer(dtanh, X[t])
        dWh  += np.outer(dtanh, H_full[t])      # H_full[t] = h_{t-1}
        dh_next = Wh.T @ dtanh

    return dWx, dWh, db, dWy, dby


# ── LSTM ──────────────────────────────────────────────────────────────────────

def lstm_step(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo):
    """
    One LSTM step. Returns (h_t, c_t, cache).
    Gates: f=forget, i=input, g=cell, o=output.
    Book equations: h_t = o_t ⊙ tanh(c_t)
    """
    def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    v = np.concatenate([h_prev, x_t])        # stacked input
    f = sigmoid(Wf @ v + bf)
    i = sigmoid(Wi @ v + bi)
    g = np.tanh(Wg @ v + bg)
    o = sigmoid(Wo @ v + bo)
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c, (f, i, g, o, c, c_prev)


def lstm_forward(X, h0, c0, Wf, Wi, Wg, Wo, bf, bi, bg, bo):
    """Forward pass over sequence. Returns H: (T, d_h)."""
    T = len(X)
    d_h = h0.shape[0]
    H, C = np.zeros((T, d_h)), np.zeros((T, d_h))
    h, c = h0.copy(), c0.copy()
    for t in range(T):
        h, c, _ = lstm_step(X[t], h, c, Wf, Wi, Wg, Wo, bf, bi, bg, bo)
        H[t], C[t] = h, c
    return H, C


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(9)
    T, d_x, d_h, d_y = 6, 3, 5, 2

    Wx = rng.normal(size=(d_h, d_x)) * 0.1
    Wh = rng.normal(size=(d_h, d_h)) * 0.1
    b  = np.zeros(d_h)
    Wy = rng.normal(size=(d_y, d_h)) * 0.1
    by = np.zeros(d_y)
    h0 = np.zeros(d_h)

    X        = rng.normal(size=(T, d_x))
    Y_target = rng.normal(size=(T, d_y))

    H, Y = rnn_forward(X, h0, Wx, Wh, b, Wy, by)
    dWx, dWh, db, dWy, dby = rnn_bptt(X, Y_target, H, h0, Wx, Wh, b, Wy, by)

    # Gradient check on Wh
    def total_loss(Wh_flat):
        Wh_ = Wh_flat.reshape(d_h, d_h)
        _, Y_ = rnn_forward(X, h0, Wx, Wh_, b, Wy, by)
        return 0.5 * np.mean((Y_ - Y_target)**2)

    eps = 1e-5
    dWh_num = np.zeros_like(Wh)
    for i in range(d_h):
        for j in range(d_h):
            Wp = Wh.copy(); Wp[i, j] += eps
            Wm = Wh.copy(); Wm[i, j] -= eps
            dWh_num[i, j] = (total_loss(Wp.ravel()) - total_loss(Wm.ravel())) / (2*eps)

    _bptt_err = np.max(np.abs(dWh - dWh_num))
    print(f"BPTT Wh gradient check max err: {_bptt_err:.2e}  (< 1e-5: {_bptt_err < 1e-5})")

    # LSTM forward
    d_xh = d_h + d_x
    def W(d_in=d_xh, d_out=d_h): return rng.normal(size=(d_out, d_in)) * 0.1
    def bv(): return np.zeros(d_h)
    H_lstm, C_lstm = lstm_forward(X, h0, np.zeros(d_h),
                                  W(), W(), W(), W(), bv(), bv(), bv(), bv())
    print(f"LSTM output H shape: {H_lstm.shape}  — last hidden norm: {np.linalg.norm(H_lstm[-1]):.4f}")
    print("Cell state C[-1] (should differ from H[-1]):", C_lstm[-1].round(3))
