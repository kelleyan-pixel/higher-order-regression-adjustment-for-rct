"""Ablation check (paper Table A.3).

Given the control-variate estimates produced by run_validation.py, test the
Monte-Carlo estimate of D_p against deliberately wrong versions of the
formula.  A consistency check that cannot reject any of these is not evidence
that the constant is correct; the point of this table is that it can.

Usage: python run_ablation.py [n]        (default: largest n available)
"""
import json, os, sys
import numpy as np
from formula import Dp_formula
from dgps import DGPS

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

ALTS = {
    "$D_p$ (Theorem 2)": lambda c, p: c["Dp"],
    "$D_{\\mathrm{Bern}}=D_p-4G$": lambda c, p: c["Dp"] - 4 * c["G"],
    "Gaussian form $-(p+1)G$": lambda c, p: -(p + 1) * c["G"],
    "drop $-3R$": lambda c, p: c["Dp"] + 3 * c["R"],
    "drop $-Q$": lambda c, p: c["Dp"] + c["Q"],
    "drop residual block": lambda c, p: c["Dp"] - 8 * c["trE2"] - 8 * c["SZe2"],
    "drop $E_W$ block": lambda c, p: c["Dp"] + 16 * c["AP"] + 8 * c["BP"] - 8 * c["CP"],
}


def main(n=None):
    res = json.load(open(os.path.join(RES, "validation.json")))
    rows = {}
    for name, d in DGPS.items():
        s = d["sym"]
        comp = Dp_formula(d["p"], d["kinds"], d["gamma"], s["g1"], s["g0"],
                          s["s1sq"], s["s0sq"], s["zs"])
        est = res[name]["est"]
        nn = str(n) if n is not None else max(est, key=lambda k: int(k))
        e, se = est[nn]["cv"], est[nn]["se_cv"]
        rows[name] = {lab: float((e - f(comp, d["p"])) / se) for lab, f in ALTS.items()}
        rows[name]["_n"], rows[name]["_est"], rows[name]["_se"] = int(nn), e, se
    json.dump(rows, open(os.path.join(RES, "ablation.json"), "w"), indent=1)
    labs = list(ALTS)
    print(" & ".join(["DGP"] + labs) + " \\\\")
    for name, r in rows.items():
        print(" & ".join([name.replace("_", "\\_")] + [f"{r[l]:+.1f}" for l in labs]) + " \\\\")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
