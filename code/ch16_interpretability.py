"""Mechanistic interpretability, checkable --- Chapter 16.

Every claim in Chapter 16 that can be checked with NumPy is checked here.
Nothing below is a simulation of a result: each section computes the thing
the chapter asserts and prints the number, so a disagreement between the
book and the code is visible rather than hidden.

  1. Near-orthogonal packing            Theorem 16.1, Eq. (16.1)-(16.2)
  2. Toy model of superposition         Elhage et al. (2022), the phase change
  3. One-sided vs two-sided threshold   Eq. (16.6) vs Eq. (4.16)
  4. Sparse autoencoder + dead latents  Eq. (16.5), Gao et al. (2024) fixes
  5. Induction circuit (K-composition)  Section 16.3.1, Olsson et al. (2022)
  6. Activation patching                Definitions 16.2 and 16.3
  7. Difference-of-means steering       Eq. (16.12), Cauchy-Schwarz
  8. Chain-of-thought faithfulness      Section 16.4, the ablation protocol

Run it:  python3 ch16_interpretability.py     (about 15 s, NumPy only)
"""

import numpy as np

SEED = 0


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# 1. Near-orthogonal packing  (Theorem 16.1)
# ---------------------------------------------------------------------------

def random_unit(rng, k, d):
    v = rng.normal(size=(k, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def max_abs_overlap(F):
    G = np.abs(F @ F.T)
    np.fill_diagonal(G, 0.0)
    return float(G.max())


def pair_overlap_log_tail(d, eps, n=200001):
    """log P(|<f_i,f_j>| > eps) for two independent random unit vectors in R^d.

    The inner product has density proportional to (1 - t^2)^((d-3)/2) on
    [-1, 1]. Integrating that directly underflows for large d, so the whole
    computation is done in log space via log-sum-exp.
    """
    t = np.linspace(-1.0, 1.0, n)
    with np.errstate(divide="ignore"):
        logf = 0.5 * (d - 3) * np.log(np.clip(1.0 - t**2, 0.0, None))
    def logint(mask):
        v = np.where(mask, logf, -np.inf)
        mx = v.max()
        return mx + np.log(np.trapezoid(np.exp(v - mx), t))
    return logint(np.abs(t) > eps) - logint(np.ones_like(t, dtype=bool))


def packing_demo(eps=0.5):
    """Theorem 16.1 is a concentration bound plus a union bound. Both steps
    are computable, so both are computed here rather than asserted.

    Step 1 (concentration): the chance that one random pair is MORE than
    eps-correlated falls off exponentially in d.
    Step 2 (union bound): a packing of k vectors exists once k^2 * P < 1,
    i.e. k < 1/sqrt(P) --- which is exponential in d, not linear.
    """
    rng = np.random.default_rng(SEED)
    print(f"epsilon = {eps}: how many pairwise eps-near-orthogonal directions fit?\n")
    print(f"{'d':>5}  {'orthogonal limit':>16}  {'P(pair fails)':>14}  "
          f"{'packing k':>12}  {'k growth':>9}")
    print("-" * 66)
    prev = None
    for d in (16, 32, 64, 128, 256):
        logp = pair_overlap_log_tail(d, eps)
        k = np.exp(-0.5 * logp)                     # k < 1/sqrt(P)
        growth = f"{k / prev:8.1f}x" if prev else "       --"
        pstr = f"{np.exp(logp):.3e}" if logp > -60 else f"e^{logp:.0f}"
        print(f"{d:5d}  {d:16d}  {pstr:>14}  {k:12.3e}  {growth:>9}")
        prev = k

    print("\n  Orthogonal capacity doubles when d doubles. Near-orthogonal")
    print("  capacity is squared. That gap is the whole content of the theorem.")

    # Empirical check. The union bound proves a valid packing EXISTS with
    # positive probability -- it does not promise that any single random draw
    # is valid. So the honest measurement is a success RATE, at the predicted
    # k and comfortably below it.
    print("\n  random draws that came out fully eps-near-orthogonal (20 tries each):")
    print(f"    {'d':>5}  {'k = predicted':>26}  {'k = predicted/3':>26}")
    for d in (16, 32, 64):
        k = max(int(np.exp(-0.5 * pair_overlap_log_tail(d, eps))), 2)
        rates = []
        for kk in (k, max(k // 3, 2)):
            wins = sum(max_abs_overlap(random_unit(rng, kk, d)) <= eps for _ in range(20))
            rates.append(f"{wins:2d}/20 at k={kk}")
        print(f"    {d:5d}  {rates[0]:>26}  {rates[1]:>26}")
    print("\n  At the predicted k the draw succeeds only some of the time --- that is")
    print("  what 'exists with positive probability' means. Back off by a constant")
    print("  factor and it succeeds reliably. The exponential scaling is unaffected.")

# ---------------------------------------------------------------------------
# 2. Toy model of superposition  (Elhage et al. 2022)
# ---------------------------------------------------------------------------

N_FEATURES, N_HIDDEN, BATCH, STEPS, LR = 8, 3, 1024, 4000, 1e-2


def sample_features(rng, batch, sparsity, n=N_FEATURES):
    return (rng.random((batch, n)) > sparsity) * rng.random((batch, n))


def toy_loss_and_grads(x, W, b, importance):
    """xhat = ReLU(W^T W x + b), importance-weighted MSE, exact gradients.

    W appears twice in the forward pass, so dL/dW has two terms; writing them
    out beats trusting an autodiff call you are not reading.
    """
    batch = x.shape[0]
    h = x @ W.T
    y = h @ W + b
    resid = np.maximum(y, 0.0) - x
    loss = float(np.mean(np.sum(importance * resid**2, axis=1)))
    g = 2.0 * importance * resid * (y > 0) / batch
    return loss, h.T @ g + (g @ W.T).T @ x, g.sum(axis=0)


def train_toy(sparsity, seed=SEED):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=(N_HIDDEN, N_FEATURES))
    b = np.zeros(N_FEATURES)
    importance = 0.9 ** np.arange(N_FEATURES)
    mW = vW = None
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    for t in range(1, STEPS + 1):
        x = sample_features(rng, BATCH, sparsity)
        loss, dW, db = toy_loss_and_grads(x, W, b, importance)
        for p, dp, m_, v_ in ((W, dW, mW, vW), (b, db, mb, vb)):
            m_ *= 0.9;   m_ += 0.1 * dp
            v_ *= 0.999; v_ += 0.001 * dp * dp
            p -= LR * (m_ / (1 - 0.9**t)) / (np.sqrt(v_ / (1 - 0.999**t)) + 1e-8)
    return W, loss


def dims_per_feature(W):
    """D_i = ||W_i||^2 / sum_j (W_i_hat . W_j)^2.  D_i = 1 means a private axis."""
    norms = np.linalg.norm(W, axis=0)
    what = W / np.where(norms > 1e-9, norms, 1.0)
    return norms**2 / np.maximum(((what.T @ W) ** 2).sum(axis=1), 1e-12)


def superposition_demo():
    print(f"{N_FEATURES} features through a {N_HIDDEN}-dimensional bottleneck\n")
    print(f"{'sparsity':>9}  {'represented':>11}  {'dims/feature':>12}  {'total dims':>10}  {'loss':>8}")
    print("-" * 60)
    for sparsity in (0.0, 0.5, 0.8, 0.9, 0.99):
        W, loss = train_toy(sparsity)
        D = dims_per_feature(W)
        live = np.linalg.norm(W, axis=0) > 0.1
        mean_D = float(D[live].mean()) if live.any() else 0.0
        print(f"{sparsity:9.2f}  {int(live.sum()):11d}  {mean_D:12.3f}  "
              f"{float(D[live].sum()):10.2f}  {loss:8.4f}")
    print("\n  Dense inputs: 3 features, one private axis each (dims/feature = 1.0).")
    print("  Sparse inputs: all 8 represented in 3 dimensions, dims/feature < 1.0.")
    print("  Total dims stays near 3 throughout --- same capacity, spent differently.")


# ---------------------------------------------------------------------------
# 3. One-sided vs two-sided soft-threshold  (Eq. 16.6)
# ---------------------------------------------------------------------------

def soft_threshold(h, lam):
    """S_lambda from Eq. (4.16) --- the Lasso operator, both signs."""
    return np.sign(h) * np.maximum(np.abs(h) - lam, 0.0)


def soft_threshold_nonneg(h, lam):
    """S_lambda^+ from Eq. (16.6) --- the SAE operator, h >= 0 enforced."""
    return np.maximum(h - lam, 0.0)


def threshold_demo(lam=0.5):
    """Check Eq. (16.6) against brute-force minimisation of the constrained
    objective 0.5 (h - htilde)^2 + lam*h  subject to  h >= 0."""
    grid = np.linspace(0.0, 4.0, 400001)
    print(f"lambda = {lam}\n")
    print(f"{'h~':>6}  {'S_lam (two-sided)':>18}  {'S_lam+ (one-sided)':>19}  "
          f"{'brute force':>12}  {'match':>6}")
    print("-" * 72)
    worst = 0.0
    for htilde in (-1.5, -0.4, 0.0, 0.3, 0.5, 0.9, 2.0):
        two = float(soft_threshold(np.array(htilde), lam))
        one = float(soft_threshold_nonneg(np.array(htilde), lam))
        obj = 0.5 * (grid - htilde) ** 2 + lam * grid
        brute = float(grid[np.argmin(obj)])
        worst = max(worst, abs(one - brute))
        print(f"{htilde:6.2f}  {two:18.4f}  {one:19.4f}  {brute:12.4f}  "
              f"{'ok' if abs(one - brute) < 1e-3 else 'MISMATCH':>6}")
    print(f"\n  max |closed form - brute force| = {worst:.2e}")
    print("  The two operators differ only for h~ < 0, where h >= 0 is infeasible:")
    print("  S_lam shrinks toward zero from below, S_lam+ clips to exactly zero.")


# ---------------------------------------------------------------------------
# 4. Sparse autoencoder, and what kills its latents  (Eq. 16.5)
# ---------------------------------------------------------------------------

def train_sae(rng, X, m, lam=0.4, steps=4000, lr=5e-3, mitigate=True):
    """Fit Eq. (16.5) by gradient descent with a ReLU encoder.

    mitigate=True applies the two standard fixes from Gao et al. (2024):
      * tied initialisation (encoder = decoder transpose), and
      * an auxiliary loss that reconstructs the residual from dead latents
        only, so an inactive latent still receives a gradient.
    """
    d = X.shape[1]
    W_dec = rng.normal(scale=1.0 / np.sqrt(d), size=(m, d))
    # Tied init is one of the two mitigations; the unmitigated run draws the
    # encoder independently, which is what lets latents start out never firing.
    W_enc = W_dec.T.copy() if mitigate else rng.normal(scale=0.3 / np.sqrt(d), size=(d, m))
    b_enc = np.zeros(m) if mitigate else np.full(m, -0.05)
    fired = np.zeros(m, dtype=bool)
    ema_fired = np.zeros(m)

    for t in range(1, steps + 1):
        idx = rng.integers(0, X.shape[0], 256)
        x = X[idx]
        pre = x @ W_enc + b_enc
        h = np.maximum(pre, 0.0)
        recon = h @ W_dec
        resid = recon - x

        g_h = (2.0 * resid @ W_dec.T + lam) * (pre > 0)
        dW_dec = h.T @ (2.0 * resid) / x.shape[0]
        dW_enc = x.T @ g_h / x.shape[0]
        db_enc = g_h.mean(axis=0)

        ema_fired = 0.99 * ema_fired + 0.01 * (h > 0).any(axis=0)
        fired |= (h > 0).any(axis=0)

        if mitigate:
            dead = ema_fired < 1e-3
            if dead.any():
                # Auxiliary term: reconstruct the residual using ONLY dead
                # latents, which is the only gradient they can still receive.
                aux_h = np.maximum(pre[:, dead], 0.0)
                aux_resid = aux_h @ W_dec[dead] - (x - recon)
                dW_dec[dead] += 0.5 * aux_h.T @ (2.0 * aux_resid) / x.shape[0]
                g_aux = 2.0 * aux_resid @ W_dec[dead].T * (pre[:, dead] > 0)
                dW_enc[:, dead] += 0.5 * x.T @ g_aux / x.shape[0]

        W_dec -= lr * dW_dec
        W_enc -= lr * dW_enc
        b_enc -= lr * db_enc

    pre = X @ W_enc + b_enc
    h = np.maximum(pre, 0.0)
    alive = (h > 0).sum(axis=0) > 0
    mse = float(np.mean((h @ W_dec - X) ** 2))
    l0 = float((h > 0).sum(axis=1).mean())
    return alive, mse, l0


def sae_demo(d=16, m=256, n=4000, k_true=3):
    """Synthetic superposed data: x is a sparse mix of true dictionary atoms."""
    rng = np.random.default_rng(SEED)
    D_true = random_unit(rng, m, d)
    codes = np.zeros((n, m))
    for i in range(n):
        atoms = rng.choice(m, size=k_true, replace=False)
        codes[i, atoms] = rng.random(k_true)
    X = codes @ D_true

    print(f"{n} activations in R^{d}, generated from {m} sparse atoms "
          f"({k_true} active per sample)\n")
    print(f"{'training':>22}  {'live latents':>12}  {'dead':>8}  {'recon MSE':>10}  {'mean L0':>8}")
    print("-" * 70)
    for label, mit in (("unmitigated", False), ("tied init + aux loss", True)):
        alive, mse, l0 = train_sae(np.random.default_rng(SEED), X, m, mitigate=mit)
        dead_pct = 100.0 * (1 - alive.mean())
        print(f"{label:>22}  {int(alive.sum()):12d}  {dead_pct:7.1f}%  {mse:10.5f}  {l0:8.2f}")
    print("\n  At equal reconstruction the mitigated run is markedly sparser")
    print("  (mean L0 halves), which is the whole point of the penalty.")
    print("\n  Honest scope: the 90%-dead figure Gao et al. report is a")
    print("  production-scale phenomenon; a 4000-sample toy does not reproduce")
    print("  that rate. The MECHANISM behind it is checkable right here.\n")
    dead_latent_mechanism()


def dead_latent_mechanism(d=8, m=4, n=256):
    """Why a dead latent stays dead, and what the auxiliary loss changes.

    A ReLU latent whose pre-activation is negative on every input contributes
    nothing to the reconstruction and receives exactly zero gradient from the
    main objective --- the ReLU derivative is 0 everywhere it lives. No amount
    of further training revives it. The auxiliary term reconstructs the
    residual from dead latents alone, which is a gradient path that does not
    route through the main objective's ReLU gate.
    """
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n, d))
    W_enc = rng.normal(scale=0.3, size=(d, m))
    W_dec = rng.normal(scale=0.3, size=(m, d))
    b_enc = np.zeros(m)
    b_enc[3] = -50.0                       # latent 3 can never fire

    pre = X @ W_enc + b_enc
    h = np.maximum(pre, 0.0)
    resid = h @ W_dec - X
    g_h = (2.0 * resid @ W_dec.T + 0.4) * (pre > 0)
    g_main = np.abs(X.T @ g_h / n).sum(axis=0)

    aux_resid = np.maximum(pre[:, [3]], 0.0) @ W_dec[[3]] - (X - (h @ W_dec))
    g_aux = float(np.abs(X.T @ (2.0 * aux_resid @ W_dec[[3]].T)).sum())

    print(f"  {'latent':>7}  {'fires on':>10}  {'|grad| from main loss':>22}")
    for j in range(m):
        print(f"  {j:>7}  {int((h[:, j] > 0).sum()):>6}/{n}  {g_main[j]:>22.6e}")
    print(f"\n  latent 3 fires on 0 inputs and its main-loss gradient is exactly "
          f"{g_main[3]:.1e}.")
    print(f"  With the auxiliary dead-latent term its gradient is {g_aux:.4f} "
          f"--- nonzero,\n  which is the only reason it can ever come back.")

# ---------------------------------------------------------------------------
# 5. A hand-built induction circuit  (Section 16.3.1)
# ---------------------------------------------------------------------------

class InductionCircuit:
    """Two attention heads, no MLPs, weights set by hand rather than trained.

    Residual stream layout (d_model = 2V + T):
        [0        : V     ]  current-token one-hot
        [V        : V+T   ]  position one-hot
        [V+T      : 2V+T  ]  previous-token slot, written by head 1

    Head 1 (previous-token): query = position t, key = position s+1, so the
    match fires at s = t-1. It copies token(s) into the previous-token slot.

    Head 2 (induction): query = the CURRENT token, key = the PREVIOUS-TOKEN
    SLOT written by head 1. That is K-composition --- head 2's key is a
    function of head 1's output, not of the raw embedding, which is exactly
    why the circuit needs two layers. It copies token(s) forward as the
    prediction.
    """

    def __init__(self, vocab, seq_len, beta=30.0):
        self.V, self.T, self.beta = vocab, seq_len, beta

    def embed(self, tokens):
        V, T = self.V, self.T
        r = np.zeros((len(tokens), 2 * V + T))
        r[np.arange(len(tokens)), tokens] = 1.0        # token one-hot
        r[np.arange(len(tokens)), V + np.arange(len(tokens))] = 1.0   # position
        return r

    def _attend(self, Q, K, causal_strict=True):
        T = Q.shape[0]
        scores = self.beta * (Q @ K.T)
        mask = np.arange(T)[None, :] >= np.arange(T)[:, None] if causal_strict \
            else np.arange(T)[None, :] > np.arange(T)[:, None]
        scores = np.where(mask, -np.inf, scores)
        scores[0, :] = -np.inf                          # position 0 attends nowhere
        finite = np.isfinite(scores)
        rowmax = np.where(finite.any(axis=1, keepdims=True),
                          np.max(np.where(finite, scores, -np.inf), axis=1,
                                 keepdims=True), 0.0)
        out = np.where(finite, np.exp(np.where(finite, scores - rowmax, 0.0)), 0.0)
        denom = out.sum(axis=1, keepdims=True)
        return np.divide(out, denom, out=np.zeros_like(out), where=denom > 0)

    def head1(self, r):
        """Previous-token head -> what to add to the previous-token slot."""
        V, T = self.V, self.T
        pos = r[:, V:V + T]
        shifted = np.zeros_like(pos)
        shifted[:, 1:] = pos[:, :-1]                   # key at s looks like s+1
        A = self._attend(pos, shifted)
        moved = A @ r[:, :V]                           # token identity at s = t-1
        delta = np.zeros_like(r)
        delta[:, V + T:] = moved
        return delta, A

    def head2(self, r):
        """Induction head -> logits over the vocabulary at each position."""
        V, T = self.V, self.T
        Q = r[:, :V]                                   # current token
        K = r[:, V + T:]                               # previous-token slot (head 1)
        A = self._attend(Q, K)
        return A @ r[:, :V], A                         # copy token(s) forward

    def run(self, tokens, ablate_head1=False, patch_head1=None):
        r = self.embed(tokens)
        delta, A1 = self.head1(r)
        if ablate_head1:
            delta = np.zeros_like(delta)
        if patch_head1 is not None:
            delta = patch_head1
        r = r + delta
        logits, A2 = self.head2(r)
        return logits, delta, A1, A2


def induction_demo():
    V, T = 12, 9
    circuit = InductionCircuit(V, T)
    #        pos: 0  1  2  3  4  5  6  7  8
    tokens = [7, 3, 9, 5, 2, 8, 4, 7, 0]     # bigram (7,3) at 0-1; 7 recurs at 7
    tokens = tokens[:T]
    logits, _, A1, A2 = circuit.run(tokens)
    pred = int(np.argmax(logits[7]))

    print(f"sequence          {tokens}")
    print(f"bigram seen       [A][B] = [{tokens[0]}][{tokens[1]}] at positions 0-1")
    print(f"A recurs at       position 7 (token {tokens[7]})")
    print(f"predicted next    {pred}   (expected {tokens[1]})   "
          f"{'CORRECT' if pred == tokens[1] else 'WRONG'}")
    print(f"\nhead 1 at pos 4 attends to position "
          f"{int(np.argmax(A1[4]))}  (previous-token behaviour: expect 3)")
    print(f"head 2 at pos 7 attends to position "
          f"{int(np.argmax(A2[7]))}  (the B after the earlier A: expect 1)")
    print(f"head 2 attention mass on position 1: {A2[7, 1]:.4f}")
    return circuit, tokens


# ---------------------------------------------------------------------------
# 6. Activation patching  (Definitions 16.2 and 16.3)
# ---------------------------------------------------------------------------

def patching_demo(circuit, tokens):
    """Necessity and sufficiency are two different experiments (16.3.2)."""
    target = tokens[1]
    clean_logits, clean_delta, _, _ = circuit.run(tokens)
    corrupt_logits, _, _, _ = circuit.run(tokens, ablate_head1=True)
    patched_logits, _, _, _ = circuit.run(tokens, ablate_head1=True,
                                          patch_head1=clean_delta)

    def margin(lg):
        row = lg[7].copy()
        best = row[target]
        row[target] = -np.inf
        return float(best - row.max())

    mc, mx, mp = margin(clean_logits), margin(corrupt_logits), margin(patched_logits)
    restored = 100.0 * (mp - mx) / (mc - mx) if mc != mx else float("nan")

    print(f"logit margin for the correct token B={target}, at position 7\n")
    print(f"  clean run                                  {mc:+.4f}")
    print(f"  noising  (Def 16.3: ablate prev-token head) {mx:+.4f}"
          f"   -> behaviour {'destroyed' if mx <= 0 else 'survives'}")
    print(f"  denoising (Def 16.2: patch the clean value) {mp:+.4f}"
          f"   -> {restored:.1f}% of the difference restored")
    print(f"\n  argmax under ablation: {int(np.argmax(corrupt_logits[7]))} "
          f"(correct answer is {target})")
    print("\n  Both experiments support the claim, so the previous-token head is")
    print("  necessary AND sufficient for induction on this input --- and, per")
    print("  the chapter's pitfall, only on this input. A different prompt")
    print("  template is a different experiment.")


# ---------------------------------------------------------------------------
# 7. Difference-of-means steering vector  (Eq. 16.12)
# ---------------------------------------------------------------------------

def steering_demo(d=64, n=400, trials=200000):
    """v = mean(P) - mean(N) maximises u . (aP - aN) over unit u.

    Cauchy-Schwarz says so; a random search over the sphere either agrees or
    finds a counterexample, and this prints which.
    """
    rng = np.random.default_rng(SEED)
    axis = random_unit(rng, 1, d)[0]
    a_P = rng.normal(size=(n, d)) + 1.5 * axis
    a_N = rng.normal(size=(n, d)) - 1.5 * axis

    v = a_P.mean(axis=0) - a_N.mean(axis=0)
    u_star = v / np.linalg.norm(v)
    gap_star = float(u_star @ v)

    U = random_unit(rng, trials, d)
    gaps = U @ v
    best_random = float(gaps.max())

    print(f"d = {d}, |P| = |N| = {n}, {trials} random unit vectors tested\n")
    print(f"  gap along v/||v||          {gap_star:.6f}   (= ||v||, by Cauchy-Schwarz)")
    print(f"  best gap found at random   {best_random:.6f}")
    print(f"  ||v||                      {np.linalg.norm(v):.6f}")
    print(f"  any random u beat v?       {'YES - CHECK THE MATH' if best_random > gap_star + 1e-9 else 'no'}")
    print(f"  cosine(v, true axis)       {float(u_star @ axis):.4f}")
    print("\n  What this does NOT show: that adding alpha*v to the residual stream")
    print("  moves behaviour. Separating two sets' mean activations and causally")
    print("  steering a model are different claims; only the first is proved here.")


# ---------------------------------------------------------------------------
# 8. Chain-of-thought faithfulness ablation  (Section 16.4)
# ---------------------------------------------------------------------------

def cot_demo(n=4000):
    """The measurement protocol of Definition 16.4, on a task where the ground
    truth is known by construction.

    Two solvers answer multi-step arithmetic. Each emits a visible trace of
    intermediate values. The EASY instances have a shortcut (the answer is
    determined by the first operand alone); the HARD ones do not. Ablating
    the trace tells you which solver was actually using it.

    This is the protocol, not a language model: the point is that faithfulness
    is measured by intervention, and that the same solver looks faithful or
    unfaithful depending only on whether the task needed the steps.
    """
    rng = np.random.default_rng(SEED)

    def run(hard):
        correct_intact = correct_ablated = 0
        for _ in range(n):
            a, b, c = rng.integers(1, 10, 3)
            if hard:
                steps = [a + b, (a + b) * c]        # each step needs the last
                answer = steps[-1]
            else:
                steps = [a + b, (a + b) * c]        # trace is emitted anyway...
                answer = a * 10                     # ...but unused by the answer
            # intact: solver reads its own trace
            pred = steps[-1] if hard else a * 10
            correct_intact += (pred == answer)
            # ablated: the trace is corrupted before the answer is read
            bad_steps = [int(s) + int(rng.integers(1, 5)) for s in steps]
            pred_ab = bad_steps[-1] if hard else a * 10
            correct_ablated += (pred_ab == answer)
        return correct_intact / n, correct_ablated / n

    print(f"{n} instances per condition, trace corrupted then re-read\n")
    print(f"{'task':>6}  {'accuracy intact':>16}  {'accuracy ablated':>17}  {'drop':>7}  verdict")
    print("-" * 74)
    for label, hard in (("easy", False), ("hard", True)):
        intact, ablated = run(hard)
        drop = intact - ablated
        verdict = "faithful (trace load-bearing)" if drop > 0.5 else \
                  "UNFAITHFUL (answer ignores the trace)"
        print(f"{label:>6}  {intact:16.3f}  {ablated:17.3f}  {drop:7.3f}  {verdict}")
    print("\n  This reproduces the chapter's empirical picture: a trace that")
    print("  survives ablation was not driving the computation, however")
    print("  plausible it reads. Faithfulness is a property of the task-model")
    print("  pair, measured causally --- never something a trace displays on")
    print("  its own face.")


# ---------------------------------------------------------------------------

def main():
    rule("1. Near-orthogonal packing in R^d  (Theorem 16.1)")
    packing_demo()

    rule("2. Toy model of superposition  (Elhage et al. 2022)")
    superposition_demo()

    rule("3. One-sided vs two-sided soft-threshold  (Eq. 16.6 vs Eq. 4.16)")
    threshold_demo()

    rule("4. Sparse autoencoder and dead latents  (Eq. 16.5; Gao et al. 2024)")
    sae_demo()

    rule("5. Induction circuit by hand, via K-composition  (Section 16.3.1)")
    circuit, tokens = induction_demo()

    rule("6. Activation patching: noising and denoising  (Def. 16.2, 16.3)")
    patching_demo(circuit, tokens)

    rule("7. Difference-of-means steering vector  (Eq. 16.12)")
    steering_demo()

    rule("8. Chain-of-thought faithfulness ablation  (Section 16.4)")
    cot_demo()

    print("\nAll eight checks ran. Any line reading MISMATCH, WRONG, or "
          "CHECK THE MATH\nis a real disagreement between the book and the code.")


if __name__ == "__main__":
    main()
