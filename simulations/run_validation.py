"""Control-variate verification of Theorem 2 (paper Table 7).

For each DGP in dgps.py the target D_p is computed exactly (formula.py) and
compared with the control-variate Monte-Carlo estimator described in
Appendix B of the paper:

    Dhat_p = mean[ n^2 (e_IREG^2 - e_REG^2) - 2 n^2 d1 L ] + 2 n^2 E[d1 L],

    d1 = (1/4) Delta' (A1 - A0) gamma,   L = gamma' Zbar + eps1bar - eps0bar,

whose per-replication variance is O(1) rather than O(n).  E[d1 L] is exact at
every finite n (formula.py, key 'cv_mean').

Usage: python run_validation.py [reps_small] [reps_large]
"""
import json, os, sys, time
import numpy as np
from formula import Dp_formula
from dgps import DGPS
from sim import run

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main(reps_small=400000, reps_large=120000, only=None):
    os.makedirs(OUT, exist_ok=True)
    plan = {200: reps_small, 800: reps_small, 3200: reps_large}
    # explicit, process-independent seeds (Python's hash() is salted per process)
    SEEDS = {(name, n): 7_000_000 + 1000 * i + j
             for i, name in enumerate(DGPS) for j, n in enumerate([200, 800, 3200])}
    path = os.path.join(OUT, 'validation.json')
    res = json.load(open(path)) if (only and os.path.exists(path)) else {}
    for name, d in DGPS.items():
        if only and name not in only:
            continue
        s = d['sym']
        f = Dp_formula(d['p'], d['kinds'], d['gamma'], s['g1'], s['g0'], s['s1sq'], s['s0sq'], s['zs'])
        assert max(abs(o) for o in f['orth']) < 1e-12, "g_w must be orthogonal to (1, Z)"
        res[name] = dict(formula=f['Dp'], V=f['V'], G=f['G'], est={})
        for n, reps in plan.items():
            t = time.time()
            raw, cv = run(d, n, reps, max(500, 800000 // n), seed=SEEDS[(name, n)])
            res[name]['est'][n] = dict(cv=float(cv.mean() + f['cv_mean']),
                                       se_cv=float(cv.std() / np.sqrt(reps)),
                                       raw=float(raw.mean()),
                                       se_raw=float(raw.std() / np.sqrt(reps)))
            print(name, n, res[name]['est'][n], f"{time.time()-t:.0f}s", flush=True)
            json.dump(res, open(os.path.join(OUT, 'validation.json'), 'w'), indent=1)


if __name__ == '__main__':
    args = sys.argv[1:]
    only = [a for a in args if not a.isdigit()] or None
    nums = [int(a) for a in args if a.isdigit()]
    main(*(nums if nums else []), only=only)
