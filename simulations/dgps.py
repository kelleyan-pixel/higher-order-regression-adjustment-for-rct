import numpy as np
import sympy as sp

z1, z2, z3 = sp.symbols('z1 z2 z3')
one = lambda Z: np.ones(Z.shape[:-1])
zero = lambda Z: np.zeros(Z.shape[:-1])

DGPS = {}

# 1. Gaussian, homoskedastic, well specified: paper predicts Dp = -(p+1)G
DGPS['gauss3_homo'] = dict(
    p=3, kinds=['gauss'] * 3, gamma=[1.0, 0.5, 0.0],
    g1=zero, g0=zero, s1=one, s0=one,
    sym=dict(zs=[z1, z2, z3], g1=0, g0=0, s1sq=1, s0sq=1))

# 2. Skewed covariates (centered Exp(1)), homoskedastic, well specified
DGPS['cexp2_homo'] = dict(
    p=2, kinds=['cexp'] * 2, gamma=[1.0, 0.5],
    g1=zero, g0=zero, s1=one, s0=one,
    sym=dict(zs=[z1, z2], g1=0, g0=0, s1sq=1, s0sq=1))

# 3. Skewed covariates, heteroskedastic, NO treatment heterogeneity
DGPS['cexp2_het_null'] = dict(
    p=2, kinds=['cexp'] * 2, gamma=[0.0, 0.0],
    g1=zero, g0=zero,
    s1=lambda Z: 1 + 0.5 * (Z[..., 0] + 1), s0=lambda Z: 1 + 0.5 * (Z[..., 0] + 1),
    sym=dict(zs=[z1, z2], g1=0, g0=0,
             s1sq=(1 + sp.Rational(1, 2) * (z1 + 1))**2, s0sq=(1 + sp.Rational(1, 2) * (z1 + 1))**2))

# 4. Everything on: skew, HTE, arm-specific misspecification (so E and E_W != 0),
#    arm-specific heteroskedasticity
q = lambda Z: Z[..., 0]**2 - 1 - 2 * Z[..., 0]          # orthogonal to (1, Z) for cexp
qs = z1**2 - 1 - 2 * z1
DGPS['cexp2_full'] = dict(
    p=2, kinds=['cexp'] * 2, gamma=[0.8, -0.4],
    g1=lambda Z: 0.5 * q(Z) + 0.3 * Z[..., 0] * Z[..., 1],
    g0=lambda Z: -0.3 * q(Z),
    s1=lambda Z: 1 + 0.4 * (Z[..., 1] + 1), s0=lambda Z: 1 + 0.3 * (Z[..., 0] + 1),
    sym=dict(zs=[z1, z2],
             g1=sp.Rational(1, 2) * qs + sp.Rational(3, 10) * z1 * z2,
             g0=-sp.Rational(3, 10) * qs,
             s1sq=(1 + sp.Rational(2, 5) * (z2 + 1))**2, s0sq=(1 + sp.Rational(3, 10) * (z1 + 1))**2))

# 5. Gaussian covariates, arm-specific quadratic misspecification + heteroskedasticity
DGPS['gauss2_misspec_het'] = dict(
    p=2, kinds=['gauss'] * 2, gamma=[1.0, 0.0],
    g1=lambda Z: 0.4 * (Z[..., 0]**2 - 1) + 0.3 * Z[..., 0] * Z[..., 1],
    g0=lambda Z: -0.2 * (Z[..., 0]**2 - 1),
    s1=lambda Z: np.sqrt(1 + 0.5 * Z[..., 0]**2), s0=one,
    sym=dict(zs=[z1, z2],
             g1=sp.Rational(2, 5) * (z1**2 - 1) + sp.Rational(3, 10) * z1 * z2,
             g0=-sp.Rational(1, 5) * (z1**2 - 1),
             s1sq=1 + sp.Rational(1, 2) * z1**2, s0sq=1))

# 6. Mixed: one skewed + one Gaussian coordinate, HTE along the skewed one,
#    heteroskedasticity along the skewed one (REG-favoured by 8 S'E[Z eps^2])
DGPS['mixed2_het'] = dict(
    p=2, kinds=['cexp', 'gauss'], gamma=[0.5, 0.5],
    g1=zero, g0=zero,
    s1=lambda Z: 1 + 0.5 * (Z[..., 0] + 1), s0=lambda Z: 1 + 0.5 * (Z[..., 0] + 1),
    sym=dict(zs=[z1, z2], g1=0, g0=0,
             s1sq=(1 + sp.Rational(1, 2) * (z1 + 1))**2, s0sq=(1 + sp.Rational(1, 2) * (z1 + 1))**2))
