"""Independent NumPy implementation of OLS with HC1/HC2/HC3 standard errors.

Used only for cross-checking the R/estimatr simulation engine
(crosscheck_r.py) and for the statsmodels cross-check.  Conventions follow
estimatr: HC1 = n/(n-k) meat(e^2), HC2 = meat(e^2/(1-h)), HC3 = meat(e^2/(1-h)^2),
with rows whose 1-h is numerically zero contributing nothing (estimatr >= 2.0).
"""
import numpy as np

LEV_TOL = 1e-10
RCOND_TOL = 1e-10


def fit(D, Y, col):
    """Batched OLS with HC1/HC2/HC3 standard errors of coefficient `col`."""
    B, n, k = D.shape
    XtX = np.einsum("bij,bik->bjk", D, D)
    ok = (1.0 / np.linalg.cond(XtX)) > RCOND_TOL
    with np.errstate(all="ignore"):
        A = np.linalg.pinv(XtX)
        coef = np.einsum("bjk,bik,bi->bj", A, D, Y)
        res = Y - np.einsum("bij,bj->bi", D, coef)
        h = np.einsum("bij,bjk,bik->bi", D, A, D)
        c = np.einsum("bk,bik->bi", A[:, col, :], D)        # tau_hat = c'Y
        omh = 1.0 - h
        lev = omh <= LEV_TOL                                 # estimatr: zero contribution
        den = np.where(lev, 1.0, omh)
        w2 = np.where(lev, 0.0, res ** 2 / den)
        w3 = np.where(lev, 0.0, res ** 2 / den ** 2)
        se = {"HC1": np.sqrt(np.einsum("bi,bi->b", c ** 2, res ** 2) * n / (n - k)),
              "HC2": np.sqrt(np.einsum("bi,bi->b", c ** 2, w2)),
              "HC3": np.sqrt(np.einsum("bi,bi->b", c ** 2, w3))}
    return coef, se, lev.any(1), ok
