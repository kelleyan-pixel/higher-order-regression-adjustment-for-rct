"""
Exact evaluation of the paper's Theorem 2 constant D_p for DGPs whose whitened
covariates Z have independent coordinates with known moments, and whose
oracle residuals are eps(w) = g_w(Z) + s_w(Z) * eta, eta indep N(0,1) (or any
mean-0 var-1 noise), with g_w a polynomial orthogonal to (1, Z) and s_w^2 a
polynomial.  Everything is computed symbolically => no Monte Carlo error in
the target.
"""
import sympy as sp
import numpy as np

def moment_table(kind):
    if kind == "gauss":
        # E Z^k for N(0,1)
        return lambda k: 0 if k % 2 else int(sp.factorial2(k - 1)) if k > 0 else 1
    if kind == "cexp":
        # central moments of Exp(1): subfactorials !k
        sub = [1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496]
        return lambda k: sub[k]
    if kind == "rade":
        return lambda k: 0 if k % 2 else 1
    raise ValueError(kind)

def expect(expr, zs, mom):
    """E[poly(Z)] with independent coords, mom[j](k) = E Z_j^k."""
    poly = sp.Poly(sp.expand(expr), *zs)
    tot = sp.Integer(0)
    for monom, coeff in poly.terms():
        t = coeff
        for j, k in enumerate(monom):
            t *= mom[j](k)
        tot += t
    return sp.nsimplify(tot)

def Dp_formula(p, kinds, gamma, g1, g0, s1sq, s0sq, zs):
    """
    gamma: list of numbers (whitened beta1-beta0)
    g1, g0: sympy polys in zs (misspecification parts of oracle residual)
    s1sq, s0sq: sympy polys (conditional variance of noise part)
    Returns dict of all ingredients and D_p.
    """
    mom = [moment_table(k) for k in kinds]
    Z = sp.Matrix(zs)
    gam = sp.Matrix([sp.nsimplify(g) for g in gamma])
    nZ2 = sum(z**2 for z in zs)
    gZ = (gam.T * Z)[0]
    E = lambda e: expect(e, zs, mom)
    G = (gam.T * gam)[0]
    K = sp.Matrix(p, p, lambda a, b: E(zs[a] * zs[b] * nZ2))
    S = sp.Matrix([E(zs[a] * nZ2) for a in range(p)])
    Sg = sp.Matrix(p, p, lambda a, b: E(zs[a] * zs[b] * gZ))
    E1 = sp.Matrix(p, p, lambda a, b: E(zs[a] * zs[b] * g1))
    E0 = sp.Matrix(p, p, lambda a, b: E(zs[a] * zs[b] * g0))
    Emat = (E1 + E0) / 2                 # E[ZZ' eps], W ~ Bern(1/2)
    EW = (E1 - E0) / 4                   # E[ZZ'(W-1/2) eps]
    Zeps2 = sp.Matrix([(E(zs[a] * (g1**2 + s1sq)) + E(zs[a] * (g0**2 + s0sq))) / 2
                       for a in range(p)])
    Eeps2 = (E(g1**2 + s1sq) + E(g0**2 + s0sq)) / 2
    F = (gam.T * K * gam)[0]
    Q = ((gam.T * S)[0]) ** 2
    R = sum(Sg[a, b] ** 2 for a in range(p) for b in range(p))
    AP = (Sg * EW).trace()
    BP = (gam.T * S)[0] * EW.trace()
    CP = (gam.T * EW * S)[0]
    trE2 = (Emat * Emat).trace()
    SZe2 = (S.T * Zeps2)[0]
    Dp = (-(2 * p + 3) * G + F - Q - 3 * R - 16 * AP - 8 * BP + 8 * CP
          + 8 * trE2 + 8 * SZe2)
    V = G + 4 * Eeps2
    # exact mean of the control variate C = 2 n^2 d1 L (see sim code)
    cv_mean = 2 * E((nZ2 - 1) * gZ * (gZ + g1 - g0))
    # sanity: orthogonality of g_w to (1, Z)
    orth = [E(g1), E(g0)] + [E(z * g1) for z in zs] + [E(z * g0) for z in zs]
    out = dict(G=G, F=F, Q=Q, R=R, AP=AP, BP=BP, CP=CP, trE2=trE2, SZe2=SZe2,
               Dp=Dp, V=V, cv_mean=cv_mean, trEW=EW.trace(), gS=(gam.T * S)[0],
               orth=orth)
    return {k: (float(v) if k != "orth" else [float(o) for o in v]) for k, v in out.items()}
