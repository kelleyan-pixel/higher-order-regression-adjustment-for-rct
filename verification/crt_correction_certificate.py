"""Numerical certificate for Theorem 10 (D_CRT = D_Bern + 4G).

The symbolic certificate establishes the Bernoulli constant D_Bern.  Theorem 10
converts it to the completely-randomized design.  This script checks that
conversion directly: it estimates

    D^design = n^2 { MSE(tau_IREG) - MSE(tau_REG) }

under (i) an exactly balanced CRT and (ii) Bernoulli(1/2) assignment, on the
same DGPs, and reports  D_CRT - D_Bern  against the predicted 4G = 4||gamma||^2.

Both estimates use the plain (uncontrolled) statistic, because the control
variate of run_validation.py is derived for the two-independent-sample CRT
representation; the difference of the two designs is large enough (4G) that the
plain estimator resolves it comfortably.

Usage:  python crt_correction_certificate.py
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulations"))
from formula import Dp_formula            # noqa: E402
from dgps import DGPS                     # noqa: E402
from sim import draw_Z                    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def estimators_masked(Z, Y, W):
    """REG and IREG with arbitrary (possibly unequal) arm sizes.

    tau_REG  = Ybar1 - Ybar0 - Delta' (S1+S0)^{-1}(s1+s0)
    tau_IREG = Ybar1 - Ybar0 - (Xbar1-Xbar)' b1 + (Xbar0-Xbar)' b0
    """
    B, n, p = Z.shape
    out = []
    for w in (1.0, 0.0):
        M = (W == w).astype(float)[..., None]                  # (B,n,1)
        nw = M.sum(1)                                          # (B,1)
        zb = (Z * M).sum(1) / nw
        yb = (Y * M[..., 0]).sum(1) / nw[:, 0]
        Zc = (Z - zb[:, None, :]) * M
        Yc = (Y - yb[:, None]) * M[..., 0]
        S = np.einsum("bij,bik->bjk", Zc, Zc)
        s = np.einsum("bij,bi->bj", Zc, Yc)
        out.append((nw[:, 0], zb, yb, S, s))
    (n1, z1, y1, S1, s1), (n0, z0, y0, S0, s0) = out
    b1 = np.linalg.solve(S1, s1[..., None])[..., 0]
    b0 = np.linalg.solve(S0, s0[..., None])[..., 0]
    bp = np.linalg.solve(S1 + S0, (s1 + s0)[..., None])[..., 0]
    zbar = (n1[:, None] * z1 + n0[:, None] * z0) / (n1 + n0)[:, None]
    tR = y1 - y0 - np.einsum("bj,bj->b", z1 - z0, bp)
    tI = (y1 - np.einsum("bj,bj->b", z1 - zbar, b1)) - (y0 - np.einsum("bj,bj->b", z0 - zbar, b0))
    return tR, tI


def run(dgp, n, reps, design, chunk, seed=0):
    rng = np.random.default_rng(seed)
    p = dgp["p"]
    gam = np.asarray(dgp["gamma"], float)
    beta0 = np.full(p, 0.9)
    tau = 1.0
    acc = []
    done = 0
    while done < reps:
        B = min(chunk, reps - done)
        Z = draw_Z(rng, (B, n, p), dgp["kinds"])
        if design == "crt":
            m = n // 2
            W = np.take_along_axis(np.r_[np.ones(m), np.zeros(n - m)][None].repeat(B, 0),
                                   np.argsort(rng.random((B, n)), 1), 1)
        else:
            W = (rng.random((B, n)) < 0.5).astype(float)
        eps = np.where(W == 1,
                       dgp["g1"](Z) + dgp["s1"](Z) * rng.standard_normal((B, n)),
                       dgp["g0"](Z) + dgp["s0"](Z) * rng.standard_normal((B, n)))
        Y = Z @ beta0 + W * (Z @ gam) + tau * W + eps
        keep = (W.sum(1) > p + 2) & (n - W.sum(1) > p + 2)
        tR, tI = estimators_masked(Z[keep], Y[keep], W[keep])
        acc.append(n ** 2 * ((tI - tau) ** 2 - (tR - tau) ** 2))
        done += B
    a = np.concatenate(acc)
    return float(a.mean()), float(a.std() / np.sqrt(a.size))


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name in ["gauss3_homo", "cexp2_homo"]:
        d = DGPS[name]
        s = d["sym"]
        f = Dp_formula(d["p"], d["kinds"], d["gamma"], s["g1"], s["g0"], s["s1sq"], s["s0sq"], s["zs"])
        for n, reps in [(200, 200000), (800, 100000)]:
            crt = run(d, n, reps, "crt", max(250, 400000 // n), seed=11)
            ber = run(d, n, reps, "bern", max(250, 400000 // n), seed=23)
            diff = crt[0] - ber[0]
            se = np.hypot(crt[1], ber[1])
            res[f"{name}_n{n}"] = dict(D_crt=crt, D_bern=ber, diff=diff, se=float(se),
                                       predicted=4 * f["G"], z=float((diff - 4 * f["G"]) / se),
                                       Dp_formula=f["Dp"])
            print(f"{name} n={n}: D_CRT={crt[0]:8.2f} ({crt[1]:.2f})  D_Bern={ber[0]:8.2f} ({ber[1]:.2f})  "
                  f"diff={diff:6.2f} ({se:.2f})  4G={4*f['G']:.2f}  z={res[f'{name}_n{n}']['z']:+.2f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "crt_correction.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
