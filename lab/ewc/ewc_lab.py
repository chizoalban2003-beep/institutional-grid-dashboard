"""EWC elasticity lab — numpy prototype of the Phase-2 plasticity economy.

Simulates the math sub-grid (gru.weight_ih[:, 0:18], 576x18) under Phase-2
clinical gradient pressure with the EWC consolidation term restricted to
the math columns:

    L_EWC = (lam/2) * sum_{i,j in sub-grid} F[i,j] * (theta[i,j] - theta*[i,j])^2

The per-parameter dynamics are exactly solvable. With a quadratic clinical
pull toward theta_clin, the gradient is

    g = (theta - theta_clin) + lam * F * (theta - theta*)

and the fixed point is the per-parameter blend

    theta_eq(i) = (theta_clin(i) + lam*F(i)*theta*(i)) / (1 + lam*F(i))

so the "exchange rate" lam*F(i) interpolates each parameter between full
clinical rewrite (lam*F -> 0) and hard lock (lam*F -> inf) — the literal
"it costs money to overwrite the math columns" story, and the hard lock
is just lam*F -> inf (gradient masked to zero on the math block).

Exam: masked R2 of the current math mapping vs the Phase-1 mapping on
math-distributed inputs (the Phase-1 certificate protocol).

Usage: python3 lab/ewc/ewc_lab.py
"""

import numpy as np

MATH_OUT = 576        # 3*HIDDEN of the GRU input projection
MATH_COLS = 18        # the 6 math kinds x [value, mask, delta]
N_TEST = 4096         # exam windows
STEPS = 2000


def make_phase1_weights(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0 / np.sqrt(MATH_COLS), (MATH_OUT, MATH_COLS))


def make_fisher(seed: int = 1) -> np.ndarray:
    """Diagonal Fisher mock: log-uniform over 3 decades — some parameters
    highly sensitive (near 1.0), others effectively free (near 1e-3)."""
    rng = np.random.default_rng(seed)
    return 10.0 ** rng.uniform(-3.0, 0.0, (MATH_OUT, MATH_COLS))


def make_clinical_target(theta_star: np.ndarray, seed: int = 2,
                         frac: float = 0.8, drift: float = 0.6) -> np.ndarray:
    """What Phase-2 clinical training "wants" the math columns to become:
    a large rewrite of most parameters (the forgetting pressure)."""
    rng = np.random.default_rng(seed)
    theta_clin = theta_star.copy()
    mask = rng.random(theta_star.shape) < frac
    theta_clin[mask] += rng.normal(0.0, drift, int(mask.sum()))
    return theta_clin


def ewc_gradient(theta, theta_clin, theta_star, fisher, lam):
    return (theta - theta_clin) + lam * fisher * (theta - theta_star)


def train(theta_star, theta_clin, fisher, lam, seed: int = 3):
    """Gradient descent on L = 1/2||theta-theta_clin||^2 + L_EWC.

    Step is curvature-scaled (lr = 1/(1+lam*F_max)) so the sim stays
    stable at every lam; the dynamics still converge to the analytic
    fixed point (verified below).
    """
    rng = np.random.default_rng(seed)
    theta = theta_star.copy()
    lr = 1.0 / (1.0 + lam * fisher.max())
    for _ in range(STEPS):
        g = ewc_gradient(theta, theta_clin, theta_star, fisher, lam)
        theta -= lr * g
    return theta


def fixed_point(theta_star, theta_clin, fisher, lam):
    """Analytic equilibrium: (theta_clin + lam*F*theta*) / (1 + lam*F)."""
    lamf = lam * fisher
    return (theta_clin + lamf * theta_star) / (1.0 + lamf)


def hard_lock(theta_star, theta_clin, seed: int = 3):
    """Current implementation: gradients zeroed on the math block.

    (The loop is vestigial — the gradient is identically zero, so the
    returned weights are exactly theta*; kept for structural parity.)
    """
    theta = theta_star.copy()
    for _ in range(STEPS):
        g = (theta - theta_clin).copy()
        g[...] = 0.0                        # masked: nothing can move
        theta -= 0.1 * g
    return theta


def exam_r2(theta, theta_star, seed: int = 4) -> float:
    """Masked R2 of current mapping vs Phase-1 mapping on math inputs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (N_TEST, MATH_COLS))
    y_star = X @ theta_star.T
    y = X @ theta.T
    num = ((y - y_star) ** 2).sum()
    den = ((y_star - y_star.mean()) ** 2).sum()
    return float(1.0 - num / max(den, 1e-12))


def clinical_fit(theta, theta_star, theta_clin) -> float:
    """Remaining-distance fraction: 1 = still Phase-1, 0 = fully rewritten.
    (progress toward the clinical target = 1 - clinical_fit)."""
    base = float(((theta_star - theta_clin) ** 2).sum())
    cur = float(((theta - theta_clin) ** 2).sum())
    return cur / max(base, 1e-12)


def param_economics(theta, theta_star, fisher, lam, k: int = 3):
    """Per-parameter economy: drift of high-F vs low-F deciles."""
    lamf = lam * fisher
    drift = np.abs(theta - theta_star)
    hi = lamf >= np.quantile(lamf, 1 - k / 10.0)
    lo = lamf <= np.quantile(lamf, k / 10.0)
    return float(drift[hi].mean()), float(drift[lo].mean())


def main():
    theta_star = make_phase1_weights()
    fisher = make_fisher()
    theta_clin = make_clinical_target(theta_star)

    # --- self-checks: dynamics match the analytic fixed point -----------
    for lam in (0.0, 0.01, 1.0, 10.0):
        th = train(theta_star, theta_clin, fisher, lam)
        fp = fixed_point(theta_star, theta_clin, fisher, lam)
        assert np.allclose(th, fp, rtol=1e-4, atol=1e-6), lam
    # the EWC gradient is the analytic derivative
    eps = 1e-6
    th0 = theta_star.copy()
    dth = np.zeros_like(th0); dth[0, 0] = eps
    analytic = ewc_gradient(th0, theta_clin, theta_star, fisher, 2.0)[0, 0]
    numeric = ((0.5 * np.sum((th0 + dth - theta_clin) ** 2)
                + 2.0 * 0.5 * (fisher[0, 0] * (th0[0, 0] + eps - theta_star[0, 0]) ** 2))
               - (0.5 * np.sum((th0 - theta_clin) ** 2)
                  + 2.0 * 0.5 * (fisher[0, 0] * (th0[0, 0] - theta_star[0, 0]) ** 2))) / eps
    assert abs(analytic - numeric) < 1e-5
    print("self-checks PASS (train == analytic fixed point; gradient exact)")

    # --- the lambda sweep (the economy's exchange rate) -----------------
    lams = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e6]
    lock = hard_lock(theta_star, theta_clin)
    lock_r2 = exam_r2(lock, theta_star)

    print(f"\n{'lam':>9} {'exam R2':>9} {'clin-prg':>9} {'locked%':>7} "
          f"{'drift hi-F':>10} {'drift lo-F':>10}  regime")
    for lam in lams:
        th = fixed_point(theta_star, theta_clin, fisher, lam)
        r2 = exam_r2(th, theta_star)
        prog = 1.0 - clinical_fit(th, theta_star, theta_clin)
        lamf = lam * fisher
        locked = float((lamf > 1.0).mean()) * 100.0
        dhi, dlo = param_economics(th, theta_star, fisher, lam)
        if lam == 0.0:
            regime = "FORGOTTEN"
        elif r2 >= 0.80:
            regime = "ELASTIC (exam held)"
        elif locked > 90:
            regime = "hard-lock-like"
        else:
            regime = "partial forget"
        print(f"{lam:>9g} {r2:>9.4f} {prog:>9.4f} {locked:>6.1f}% "
              f"{dhi:>10.4f} {dlo:>10.4f}  {regime}")

    print(f"\n{'HARD LOCK':>9} {lock_r2:>9.4f}   (gradient masked to zero "
          f"on math columns = the current v3 implementation)")

    # --- verdict threshold ----------------------------------------------
    for lam in lams:
        th = fixed_point(theta_star, theta_clin, fisher, lam)
        r2 = exam_r2(th, theta_star)
        if r2 >= 0.80:
            prog = 1.0 - clinical_fit(th, theta_star, theta_clin)
            print(f"\ncheapest lam holding the exam floor (R2 >= 0.80): "
                  f"lam = {lam:g}, exam R2 {r2:.4f}, clinical progress "
                  f"{prog:.2%} of the full rewrite")
            break


if __name__ == "__main__":
    main()
