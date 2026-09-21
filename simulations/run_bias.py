"""Numerical check of the bias decomposition (paper Remark 3).

The remark states that, to leading order under a balanced CRT,

    E[tau_REG  - tau] = -(4 tr(E_W) + gamma'S)/n + O(n^{-3/2}),
    E[tau_IREG - tau] = -(4 tr(E_W))/n          + O(n^{-3/2}),

so that the squared-bias difference accounts for the -Q - 8 B_P terms of D_p.
This script estimates both biases by Monte Carlo on the analytic DGPs of
dgps.py, where tr(E_W) and gamma'S are known exactly.

Usage: python run_bias.py [reps]
"""
import json, os, sys
import numpy as np
from formula import Dp_formula
from dgps import DGPS
from sim import draw_Z, estimators

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEED0 = 4_000_000


def bias(dgp, n, reps, chunk, seed):
    rng = np.random.default_rng(seed)
    m, p = n // 2, dgp["p"]
    gam = np.asarray(dgp["gamma"], float)
    beta0, tau = np.full(p, 0.9), 1.0
    accR, accI = [], []
    done = 0
    while done < reps:
        B = min(chunk, reps - done)
        Z1 = draw_Z(rng, (B, m, p), dgp["kinds"]); Z0 = draw_Z(rng, (B, m, p), dgp["kinds"])
        Y1 = tau + Z1 @ (beta0 + gam) + dgp["g1"](Z1) + dgp["s1"](Z1) * rng.standard_normal((B, m))
        Y0 = Z0 @ beta0 + dgp["g0"](Z0) + dgp["s0"](Z0) * rng.standard_normal((B, m))
        tR, tI, _, _ = estimators(Z1, Y1, Z0, Y0)
        accR.append(tR - tau); accI.append(tI - tau)
        done += B
    eR, eI = np.concatenate(accR), np.concatenate(accI)
    return ((eR.mean(), eR.std() / np.sqrt(eR.size)),
            (eI.mean(), eI.std() / np.sqrt(eI.size)))


def main(reps=2000000):
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for i, name in enumerate(["cexp2_full", "gauss2_misspec_het", "mixed2_het"]):
        d = DGPS[name]; s = d["sym"]
        f = Dp_formula(d["p"], d["kinds"], d["gamma"], s["g1"], s["g0"], s["s1sq"], s["s0sq"], s["zs"])
        for j, n in enumerate([200, 800]):
            (bR, seR), (bI, seI) = bias(d, n, reps, max(500, 800000 // n), SEED0 + 100 * i + j)
            pR, pI = -(4 * f["trEW"] + f["gS"]) / n, -4 * f["trEW"] / n
            res[f"{name}_n{n}"] = dict(bias_REG=bR, se_REG=seR, pred_REG=pR,
                                       z_REG=(bR - pR) / seR,
                                       bias_IREG=bI, se_IREG=seI, pred_IREG=pI,
                                       z_IREG=(bI - pI) / seI, reps=reps)
            print(f"{name} n={n}: REG {bR:+.5f} ({seR:.5f}) vs {pR:+.5f} [z={(bR-pR)/seR:+.1f}] | "
                  f"IREG {bI:+.5f} ({seI:.5f}) vs {pI:+.5f} [z={(bI-pI)/seI:+.1f}]", flush=True)
    json.dump(res, open(os.path.join(OUT, "bias.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000000)
