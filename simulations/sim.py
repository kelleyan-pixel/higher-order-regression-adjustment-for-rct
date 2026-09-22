import numpy as np

def draw_Z(rng, shape, kinds):
    B, m, p = shape
    Z = np.empty(shape)
    for j, k in enumerate(kinds):
        if k == "gauss":
            Z[..., j] = rng.standard_normal((B, m))
        elif k == "cexp":
            Z[..., j] = rng.exponential(1.0, (B, m)) - 1.0
        elif k == "rade":
            Z[..., j] = rng.integers(0, 2, (B, m)) * 2.0 - 1.0
    return Z

def arm_stats(Z, Y):
    zb = Z.mean(1); yb = Y.mean(1)
    Zc = Z - zb[:, None, :]; Yc = Y - yb[:, None]
    S = np.einsum('bij,bik->bjk', Zc, Zc)
    s = np.einsum('bij,bi->bj', Zc, Yc)
    return zb, yb, S, s

def estimators(Z1, Y1, Z0, Y0):
    """Exact algebraic forms (checked against lstsq in check_ols):
       REG  = Ybar1-Ybar0 - Delta' beta_pool   (pooled within-arm slope)
       IREG = Ybar1-Ybar0 - Delta' (b1+b0)/2   (balanced design)"""
    z1, y1, S1, s1 = arm_stats(Z1, Y1)
    z0, y0, S0, s0 = arm_stats(Z0, Y0)
    b1 = np.linalg.solve(S1, s1[..., None])[..., 0]
    b0 = np.linalg.solve(S0, s0[..., None])[..., 0]
    bp = np.linalg.solve(S1 + S0, (s1 + s0)[..., None])[..., 0]
    D = z1 - z0
    tR = y1 - y0 - np.einsum('bj,bj->b', D, bp)
    tI = y1 - y0 - np.einsum('bj,bj->b', D, (b1 + b0) / 2)
    return tR, tI, z1, z0

def run(dgp, n, reps, chunk, seed=0):
    """Returns n^2 * (MSE_I - MSE_R) estimates: raw and control-variate."""
    rng = np.random.default_rng(seed)
    m = n // 2
    p = dgp['p']; gam = np.asarray(dgp['gamma'], float)
    beta0 = np.full(p, 0.9); tau = 1.0
    raw, cv = [], []
    done = 0
    while done < reps:
        B = min(chunk, reps - done)
        draw = dgp.get('draw', lambda r, shape: draw_Z(r, shape, dgp['kinds']))   # custom draw for dependent coordinates
        Z1 = draw(rng, (B, m, p)); Z0 = draw(rng, (B, m, p))
        e1 = dgp['g1'](Z1) + dgp['s1'](Z1) * rng.standard_normal((B, m))
        e0 = dgp['g0'](Z0) + dgp['s0'](Z0) * rng.standard_normal((B, m))
        Y1 = tau + Z1 @ (beta0 + gam) + e1
        Y0 = Z0 @ beta0 + e0
        tR, tI, z1, z0 = estimators(Z1, Y1, Z0, Y0)
        eR, eI = tR - tau, tI - tau
        # control variate: d1 = 1/4 Delta'(A1-A0) gamma,  L = gamma'Zbar + eps1bar - eps0bar
        A1 = np.einsum('bij,bik->bjk', Z1, Z1) / m - np.eye(p)
        A0 = np.einsum('bij,bik->bjk', Z0, Z0) / m - np.eye(p)
        D = z1 - z0
        d1 = 0.25 * np.einsum('bj,bjk,k->b', D, A1 - A0, gam)
        L = (z1 + z0) / 2 @ gam + e1.mean(1) - e0.mean(1)
        r = n**2 * (eI**2 - eR**2)
        raw.append(r)
        cv.append(r - 2 * n**2 * d1 * L)
        done += B
    return np.concatenate(raw), np.concatenate(cv)

def check_ols(dgp, n, seed=1):
    """Compare fast formulas with brute-force lstsq on a few replications."""
    rng = np.random.default_rng(seed)
    m = n // 2; p = dgp['p']; gam = np.asarray(dgp['gamma'], float)
    beta0 = np.full(p, 0.9)
    Z1 = draw_Z(rng, (3, m, p), dgp['kinds']); Z0 = draw_Z(rng, (3, m, p), dgp['kinds'])
    Y1 = 1 + Z1 @ (beta0 + gam) + dgp['g1'](Z1) + dgp['s1'](Z1) * rng.standard_normal((3, m))
    Y0 = Z0 @ beta0 + dgp['g0'](Z0) + dgp['s0'](Z0) * rng.standard_normal((3, m))
    tR, tI, _, _ = estimators(Z1, Y1, Z0, Y0)
    out = []
    for b in range(3):
        X = np.vstack([Z1[b], Z0[b]]); Y = np.concatenate([Y1[b], Y0[b]])
        W = np.r_[np.ones(m), np.zeros(m)]
        Dr = np.column_stack([np.ones(n), X, W])
        cr = np.linalg.lstsq(Dr, Y, rcond=None)[0]
        Di = np.column_stack([np.ones(n), X, W, W[:, None] * X])
        ci = np.linalg.lstsq(Di, Y, rcond=None)[0]
        tIi = ci[p + 1] + X.mean(0) @ ci[p + 2:]
        out.append((tR[b] - cr[p + 1], tI[b] - tIi))
    return out
