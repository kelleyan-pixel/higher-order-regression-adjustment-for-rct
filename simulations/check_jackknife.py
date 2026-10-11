"""Check the batched jackknife of correction_python.jackknife against brute-force refitting.

For each design, every leave-one-out estimate is recomputed by refitting the regression on the
remaining n - 1 units (IREG: the uncentred interacted regression, tau + Xbar_(-i)' delta, i.e. the fit
recentred at the leave-one-out mean), leave-one-out fits whose design is rank deficient are omitted,
and the resulting jackknife standard error is compared with the deletion-formula value used in the
simulations.  The rare-value design is included because it produces rank-deficient leave-one-out fits
(an arm with a single rare unit); datasets whose full interacted design is already rank deficient
give no IREG estimate in the simulations and are skipped here.

Usage: python3 simulations/check_jackknife.py
Output: results/jackknife_check.json
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import correction_python as cp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def brute(X, W, Y, kind):
    n, p = X.shape
    vals = []
    for i in range(n):
        keep = np.arange(n) != i
        Xs, Ws, Ys = X[keep], W[keep], Y[keep]
        one = np.ones((n - 1, 1))
        if kind == "REG":
            D = np.column_stack([one, Xs, Ws])
        else:
            D = np.column_stack([one, Xs, Ws, Ws[:, None] * Xs])
        c, _, rank, _ = np.linalg.lstsq(D, Ys, rcond=None)
        if rank < D.shape[1]:
            continue                                  # undefined leave-one-out estimate: omitted
        vals.append(c[p + 1] if kind == "REG" else c[p + 1] + Xs.mean(0) @ c[p + 2:])
    vals = np.array(vals)
    m = len(vals)
    if m == 0:
        return np.nan, n
    return float(np.sqrt((m - 1) / m * np.sum((vals - vals.mean()) ** 2))), n - m


def main():
    rng = np.random.default_rng(2026)
    out = {"datasets": 0, "omitted_loo_fits": 0, "max_rel_gap": 0.0}
    for fam in ["gauss_iid", "misspec", "null_hte", "heavy_het", "rare_extreme", "rare_extreme", "rare_extreme"]:
        p, n, B = 5, 200, 6
        X, W, Y, _ = cp.dgp(rng, fam, B, n, p)
        one = np.ones((B, n, 1))
        DR = np.concatenate([one, X, W[..., None]], -1)
        Xc = X - X.mean(1, keepdims=True)
        DI = np.concatenate([one, Xc, W[..., None], W[..., None] * Xc], -1)
        jR, dR = cp.jackknife(DR, Y, p + 1)
        jI, dI = cp.jackknife(DI, Y, p + 1, delta_cols=list(range(p + 2, 2 * p + 2)), Xc=Xc)
        for b in range(B):
            for kind, j, d in (("REG", jR[b], dR[b]), ("IREG", jI[b], dI[b])):
                ref, omitted = brute(X[b], W[b], Y[b], kind)
                if omitted == n:                     # full design rank deficient: no estimate in the simulations
                    continue
                assert omitted == d, (fam, b, kind, omitted, d)
                out["max_rel_gap"] = max(out["max_rel_gap"], abs(j / ref - 1))
                out["omitted_loo_fits"] += int(omitted)
                out["datasets"] += 1
    # scalar lognormal engine (p = 1)
    for _ in range(3):
        n, m, s, c, g = 120, 60, 1.5, 1.0, 0.4
        X = np.exp(s * rng.standard_normal(n))
        W = np.zeros(n); W[rng.permutation(n)[:m]] = 1
        Y = 0.9 * X + W * (1 + g * X) + (1 + c * X) * rng.standard_normal(n)
        Xc = X - X.mean()
        one = np.ones(n)
        jR, _ = cp.jackknife(np.stack([one, X, W], -1)[None], Y[None], 2)
        jI, _ = cp.jackknife(np.stack([one, Xc, W, W * Xc], -1)[None], Y[None], 2, delta_cols=[3], Xc=Xc[None, :, None])
        for kind, j in (("REG", jR[0]), ("IREG", jI[0])):
            ref, _ = brute(X[:, None], W, Y, kind)
            out["max_rel_gap"] = max(out["max_rel_gap"], abs(j / ref - 1))
            out["datasets"] += 1
    print(f"jackknife: {out['datasets']} estimator-dataset pairs, {out['omitted_loo_fits']} omitted leave-one-out fits, "
          f"max relative gap to refitting {out['max_rel_gap']:.1e}")
    assert out["max_rel_gap"] < 1e-10 and out["omitted_loo_fits"] > 0
    json.dump(out, open(os.path.join(ROOT, "results", "jackknife_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
