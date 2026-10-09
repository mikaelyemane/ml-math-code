"""Ch. 13 — Policy Gradient.

Reproduces the seeded baseline example of Section 13.4 (the two-armed
bandit at state "med": Var 581.3 / 92.5, a 6.3x reduction), then runs
REINFORCE on a 3-state chain MDP with a running-mean baseline, and shows
the PPO clipped surrogate as a one-step contrast against vanilla policy
gradient.
"""
import numpy as np


N_STATES = 3
N_ACTIONS = 2   # 0=left, 1=right
GOAL_STATE = 2


def softmax_policy(theta, state):
    logits = theta[state]
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def step(state, action):
    if action == 1:
        next_state = min(state + 1, N_STATES - 1)
    else:
        next_state = max(state - 1, 0)
    reward = 1.0 if next_state == GOAL_STATE else 0.0
    done = next_state == GOAL_STATE
    return next_state, reward, done


def sample_trajectory(theta, rng, max_steps=20):
    traj = []
    state = 0
    for _ in range(max_steps):
        probs = softmax_policy(theta, state)
        action = rng.choice(N_ACTIONS, p=probs)
        next_state, reward, done = step(state, action)
        traj.append((state, action, reward))
        state = next_state
        if done:
            break
    return traj


def returns(traj, gamma=0.99):
    G = 0.0
    out = []
    for _, _, r in reversed(traj):
        G = r + gamma * G
        out.append(G)
    return list(reversed(out))


def collect_returns(theta, rng, batch_size):
    trajs = [sample_trajectory(theta, rng) for _ in range(batch_size)]
    Gs = [returns(t) for t in trajs]
    return trajs, Gs


def policy_gradient_step(theta, trajs, Gs, lr, baseline=0.0):
    """One REINFORCE update with optional scalar baseline subtracted from G."""
    grad = np.zeros_like(theta)
    for traj, G_traj in zip(trajs, Gs):
        for (s, a, _), G in zip(traj, G_traj):
            probs = softmax_policy(theta, s)
            grad_log = -probs.copy()
            grad_log[a] += 1.0
            grad[s] += (G - baseline) * grad_log
    theta += lr * grad / len(trajs)
    return theta


def grad_variance(theta, trajs, Gs, baseline=0.0):
    """Return the variance of the per-trajectory policy-gradient estimate."""
    per_traj = []
    for traj, G_traj in zip(trajs, Gs):
        g = np.zeros_like(theta)
        for (s, a, _), G in zip(traj, G_traj):
            probs = softmax_policy(theta, s)
            grad_log = -probs.copy()
            grad_log[a] += 1.0
            g[s] += (G - baseline) * grad_log
        per_traj.append(g.ravel())
    arr = np.stack(per_traj)
    return arr.var(axis=0).sum()


def reach_prob(theta, rng, n_eval=500):
    successes = sum(
        any(r > 0 for _, _, r in sample_trajectory(theta, rng))
        for _ in range(n_eval)
    )
    return successes / n_eval


def baseline_variance_example(n=1000, seed=0):
    """Section 13.4 example, call for call: single-sample REINFORCE at theta=0
    on the two-armed bandit with Q(med, study)=46.84, Q(med, rest)=43.10 and
    reward noise sd 19.15. Returns (r_bar, Var[g] at b=0, Var[g] at b=r_bar)."""
    rng = np.random.default_rng(seed)
    action_is_study = rng.random(n) < 0.5
    rewards = np.where(action_is_study,
                       rng.normal(46.84, 19.15, n),
                       rng.normal(43.10, 19.15, n))
    score = np.where(action_is_study, 0.5, -0.5)   # grad log pi at theta=0
    r_bar = rewards.mean()
    g_no_baseline = score * rewards
    g_baseline = score * (rewards - r_bar)
    return r_bar, g_no_baseline.var(), g_baseline.var()


def ppo_clip_loss(ratio, advantage, eps=0.2):
    """Clipped surrogate objective from Schulman et al. 2017 (Eq. 7)."""
    clipped = np.clip(ratio, 1 - eps, 1 + eps)
    return -np.minimum(ratio * advantage, clipped * advantage).mean()


def vanilla_pg_loss(ratio, advantage):
    return -(ratio * advantage).mean()


# ── DPO: Direct Preference Optimization ─────────────────────────────────────
# Reference: Rafailov et al., NeurIPS 2023.
#
# Setup: a toy preference task with a discrete prompt-and-completion space.
# Each "preference pair" is (x, y_w, y_l) where y_w is preferred to y_l.
# We learn pi_theta(y | x) parameterized as softmax over completions,
# starting from a fixed pi_ref. Loss:
#
#   L_DPO(theta) = -E [ log sigmoid( beta * log pi_theta(y_w|x)/pi_ref(y_w|x)
#                                  - beta * log pi_theta(y_l|x)/pi_ref(y_l|x) ) ]
#
# Empirically: as training progresses, the log-ratio margin keeps growing.
# On a pair labelled only one way (empirical p = 1) the logistic loss has no
# finite optimum, so beta does not cap the margin (DPO's overfitting pathology, Ch.13 §DPO).


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def dpo_step(theta, ref_logp, pairs, beta=0.5, lr=0.1):
    """One gradient step on the DPO loss.
    theta:    (N_prompts, N_completions) log-prob parameters (softmax-normalised).
    ref_logp: (N_prompts, N_completions) reference log-probs (fixed).
    pairs:    list of (x, y_w, y_l) tuples.
    Returns:  updated theta, current mean loss, current mean margin.
    """
    # Current policy log-probs from theta (softmax over the completion axis)
    z = theta - theta.max(axis=1, keepdims=True)
    pol_logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    loss = 0.0
    margins = 0.0
    grad = np.zeros_like(theta)

    for (x, y_w, y_l) in pairs:
        log_ratio_w = pol_logp[x, y_w] - ref_logp[x, y_w]
        log_ratio_l = pol_logp[x, y_l] - ref_logp[x, y_l]
        h = beta * (log_ratio_w - log_ratio_l)
        margins += h
        loss += -np.log(sigmoid(h) + 1e-12)

        # dL/dh = -(1 - sigmoid(h)) = -sigmoid(-h)
        dL_dh = -sigmoid(-h)
        # dh/dpol_logp[x, y_w] = beta;  dh/dpol_logp[x, y_l] = -beta
        # dpol_logp[x, y]/dtheta[x, k] = delta(y, k) - p(k|x)
        p_x = np.exp(pol_logp[x])  # softmax probs for prompt x
        # grad through y_w
        for k in range(theta.shape[1]):
            indicator_w = 1.0 if k == y_w else 0.0
            indicator_l = 1.0 if k == y_l else 0.0
            grad[x, k] += dL_dh * beta * (indicator_w - p_x[k])
            grad[x, k] += dL_dh * (-beta) * (indicator_l - p_x[k])

    n = len(pairs)
    grad /= n
    theta_new = theta - lr * grad
    return theta_new, loss / n, margins / n


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    theta = np.zeros((N_STATES, N_ACTIONS))

    n_episodes, batch_size, lr = 500, 10, 0.05
    running_baseline = 0.0
    bl_alpha = 0.1  # EMA factor for the baseline

    # ---- Section 13.4 seeded baseline example ----
    r_bar, v0, v1 = baseline_variance_example()
    print("Section 13.4 bandit example (default_rng(0), 1000 rollouts):")
    print(f"  r_bar = {r_bar:.2f}  Var[g] b=0 = {v0:.1f}  "
          f"Var[g] b=r_bar = {v1:.1f}  ratio = {v0 / v1:.1f}x")
    print()

    # ---- variance comparison on the chain MDP, single batch at init ----
    trajs, Gs = collect_returns(theta, rng, batch_size=64)
    flat_G = np.array([G for traj_G in Gs for G in traj_G])
    var_no_baseline = grad_variance(theta, trajs, Gs, baseline=0.0)
    var_with_baseline = grad_variance(theta, trajs, Gs, baseline=flat_G.mean())
    print(f"grad var, no baseline   = {var_no_baseline:.4f}")
    print(f"grad var, with baseline = {var_with_baseline:.4f}")
    print(f"variance reduction      = {var_no_baseline / max(var_with_baseline, 1e-12):.2f}x")
    print()

    # ---- training with running-mean baseline ----
    for ep in range(1, n_episodes + 1):
        trajs, Gs = collect_returns(theta, rng, batch_size)
        flat_G = np.array([G for traj_G in Gs for G in traj_G])
        running_baseline = (1 - bl_alpha) * running_baseline + bl_alpha * flat_G.mean()
        theta = policy_gradient_step(theta, trajs, Gs, lr, baseline=running_baseline)
        if ep % 100 == 0:
            p = reach_prob(theta, rng)
            print(f"Episode {ep:4d}  P(reach goal) = {p:.3f}  baseline = {running_baseline:.3f}")
    print()

    # ---- PPO clip vs vanilla PG: one-step demonstration ----
    ratio = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.5])
    adv = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    print(f"PPO clip loss (positive adv)  = {ppo_clip_loss(ratio, adv):.4f}")
    print(f"Vanilla PG loss (positive adv) = {vanilla_pg_loss(ratio, adv):.4f}")
    adv = -adv
    print(f"PPO clip loss (negative adv)  = {ppo_clip_loss(ratio, adv):.4f}")
    print(f"Vanilla PG loss (negative adv) = {vanilla_pg_loss(ratio, adv):.4f}")

    # ---- DPO training loop on a toy preference task ------------------------
    # Toy task: 4 prompts x 3 completions. For each prompt, completion 0 is
    # preferred over completion 2 (so the model should up-weight 0 and
    # down-weight 2). Reference policy: uniform (log 1/3 everywhere).
    rng_dpo = np.random.default_rng(13)
    N_PROMPTS, N_COMPL = 4, 3
    ref_logp = np.full((N_PROMPTS, N_COMPL), np.log(1.0 / N_COMPL))
    theta_dpo = np.zeros((N_PROMPTS, N_COMPL))  # init = ref policy
    # Preference pairs: for each prompt x, (x, y_w=0, y_l=2)
    pairs = [(x, 0, 2) for x in range(N_PROMPTS)]

    print("\nDPO training (toy preference task, 4 prompts x 3 completions):")
    print(f"  init log-ratio margin = 0.00 (policy = reference)")
    for step_t in range(1, 301):
        theta_dpo, loss, margin = dpo_step(theta_dpo, ref_logp, pairs,
                                           beta=0.5, lr=0.2)
        if step_t in (1, 50, 100, 200, 300):
            print(f"  step {step_t:3d}  loss = {loss:.4f}  "
                  f"beta * (log pi_w/ref - log pi_l/ref) = {margin:.3f}")
    # Final softmax probs for prompt 0
    z = theta_dpo[0] - theta_dpo[0].max()
    final_p = np.exp(z) / np.exp(z).sum()
    print(f"  final policy for prompt 0: {final_p.round(3)}  "
          f"(was {np.array([1/3]*3).round(3)} at init)")
    print(f"  margin keeps growing: no finite optimum on single-label "
          f"pairs, so beta does not anchor the policy (Ch.13 §DPO).")
