# Solution — "Find the bug" (Chapter 5)

**Stop here if you have not yet run the script and read the step-1 ratio.**

---

## The bug

`adam_step`, line 3 of the update:

```python
m_hat = m / (1.0 - BETA2**t)      # WRONG
m_hat = m / (1.0 - BETA1**t)      # right
```

One character: `2` → `1`. The first-moment estimate is being de-biased with the
**second** moment's decay rate.

## Why it produces exactly a 100× step

Adam's whole claim to scale-invariance is that the update

$$\Delta w_j = -\eta \cdot \frac{\hat{m}_j}{\sqrt{\hat{v}_j} + \epsilon}$$

is bounded by roughly $\eta$, because $\hat m$ and $\sqrt{\hat v}$ are estimates
of the same quantity — the gradient — and their ratio is $O(1)$.

That only holds if each is de-biased with its own decay rate. Both EMAs start at
zero, so at step $t$ they are biased low by exactly the factor that the
correction divides out:

$$m_t = (1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i} g_i,
\qquad \mathbb{E}[m_t] \approx (1-\beta_1^t)\,\mathbb{E}[g],$$

$$v_t = (1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i} g_i^2,
\qquad \mathbb{E}[v_t] \approx (1-\beta_2^t)\,\mathbb{E}[g^2].$$

Take $t = 1$ and a single coordinate with gradient $g$:

| quantity | value |
|---|---|
| $m_1$ | $(1-\beta_1)\,g = 0.1\,g$ |
| $v_1$ | $(1-\beta_2)\,g^2 = 0.001\,g^2$ |
| $\hat m_1$ correct | $m_1 / (1-\beta_1) = 0.1g / 0.1 = g$ |
| $\hat m_1$ buggy | $m_1 / (1-\beta_2) = 0.1g / 0.001 = 100\,g$ |
| $\hat v_1$ | $v_1 / (1-\beta_2) = 0.001g^2 / 0.001 = g^2$ |
| $\sqrt{\hat v_1}$ | $\lvert g \rvert$ |

So the correct update is $-\eta \cdot g/\lvert g\rvert = \pm\eta$, and the buggy
one is $-\eta \cdot 100g/\lvert g\rvert = \pm 100\eta$.

**The ratio is $\dfrac{1-\beta_1}{1-\beta_2} = \dfrac{0.1}{0.001} = 100$** —
which is precisely the `|update| / lr = 100.0` the script prints. That round
number is the fingerprint: nothing in the data or the loss could produce it.
It falls out of the two hyperparameters alone.

Confirm it against the run:

```
g[j]           = -2.287411e+01
m[j]           = -2.287411e+00     = 0.1 * g[j]          ✓
m_hat[j]       = -2.287411e+03     = m[j] / 0.001        ← should be m[j] / 0.1
sqrt(v_hat[j]) = +2.287411e+01     = |g[j]|              ✓
update         = -1.000000e+00     = -lr * 100
```

With $\eta = 0.01$ the first step moves every one of the 100 coordinates by
$1.0$, while the optimum sits about $0.05$ away. The iterate overshoots by
roughly twenty times the distance to the solution, in every coordinate at once,
and the loss goes from $24.2$ to $1.12\times10^4$.

## Why it recovers — and why that is the dangerous part

The distortion factor at step $t$ is

$$\frac{1-\beta_1^t}{1-\beta_2^t}.$$

| $t$ | factor |
|---|---|
| 1 | 100 |
| 10 | 65 |
| 100 | 10.5 |
| 1000 | 1.6 |
| 5000 | 1.01 |

It decays to 1. Given enough steps on a convex problem, the buggy optimiser
still lands on the noise floor — in this run it reaches `3.829e-05`, the same
value the correct code reaches. A convergence test at step 400 passes. A unit
test asserting `final_loss < 1e-4` passes. A short smoke run with a small
learning rate never spikes far enough to look alarming.

What it destroys is the **first few hundred steps**, and that is where the damage
is permanent in real training:

- Warmup exists precisely because early steps are where a transformer is most
  fragile. A 100× step during warmup can push the model into a region it never
  fully recovers from, and the loss curve afterwards just looks "a bit worse"
  with no obvious cause.
- With a learning-rate schedule, the wasted early steps are the ones with the
  highest learning rate — the most expensive ones to throw away.
- On a non-convex objective there is no theorem bringing you back. The optimiser
  does not return to where it would have been; it converges somewhere else.

## Fix and verify

```python
m_hat = m / (1.0 - BETA1**t)
```

Re-run:

```
step    1   1.6037e+01      <- below the 2.4154e+01 starting loss
step   10   2.2734e+00
step  100   2.5890e-04
step  400   3.8289e-05
largest increase : 1.034x
```

Smooth descent, never worse than the starting point, floor reached.

## The residual 3% ripple is not a bug

The fixed run still shows increases of up to ~3% near the floor. That is Adam's
own non-convergence behaviour, not a defect: $\hat v$ is an EMA, so after a
stretch of small gradients the denominator is too small when a larger gradient
arrives, and the step overshoots slightly. See the *"Adam Does Not Always
Converge"* pitfall later in the chapter, and AMSGrad
(Reddi et al., 2018) for the standard remedy.

## How to catch this class of bug

The generalisable lesson is not "check your bias correction". It is that
**Adam's update magnitude is a testable invariant**. Assert it:

```python
assert np.max(np.abs(update)) <= 5 * lr, (
    f"Adam step {np.max(np.abs(update)):.3e} exceeds 5*lr={5*lr:.3e} -- "
    "the m_hat / sqrt(v_hat) ratio is not O(1)"
)
```

Any error in the correction algebra, the moment recursions, or the epsilon
placement breaks that bound, usually by a round factor built out of
$\beta_1$ and $\beta_2$. One assertion, three bug classes, zero cost at
training time relative to a matmul.
