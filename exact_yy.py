#!/usr/bin/env python3
"""Exact thermodynamics of the Lieb-Liniger gas H = -lam sum d^2/dx_i^2 + g sum_{i<j} delta(x_i - x_j)
in the thermodynamic limit at temperature T (Yang-Yang, J. Math. Phys. 10, 1115 (1969)), in the units of
worm_pimc.cpp (pot=3, lam = hbar^2/2m):
    eps(k) = lam k^2 - mu - (T/2pi) int dq 2c/(c^2 + (k-q)^2) ln(1 + exp(-eps(q)/T)),   c = g/(2 lam)
    p = (T/2pi) int dk ln(1 + exp(-eps(k)/T))
Derived by finite differences of p(mu, T, g): n = dp/dmu, <dN^2>/L = T dn/dmu, E/L = mu n - p + T dp/dT,
<g sum delta>/L = -g dp/dg; plus the dimensionless coupling gamma = g/(2 lam n).
For a ring of length L these hold up to corrections exponentially small in L/xi_T.
usage: python3 exact_yy.py beta mu g [lam=0.5] [L=...]"""
import math, sys
import numpy as np


def pressure(mu, T, g, lam, K=None, nk=1501):
    c = g / (2 * lam)
    if K is None:
        K = math.sqrt(max(mu, 0) / lam + 40 * T / lam) + 2.0
    k, dk = np.linspace(-K, K, nk, retstep=True)
    w = np.full(nk, dk); w[0] = w[-1] = dk / 2           # trapezoid
    ker = (2 * c / (c * c + (k[:, None] - k[None, :]) ** 2)) * w[None, :] / (2 * math.pi)
    eps = lam * k * k - mu
    for it in range(2000):
        f = np.logaddexp(0, -eps / T) * T                  # T ln(1 + e^{-eps/T})
        new = lam * k * k - mu - ker @ f
        if np.max(np.abs(new - eps)) < 1e-14 * (1 + abs(mu)):
            eps = new; break
        eps = 0.5 * eps + 0.5 * new if c < 0.5 else new
    f = np.logaddexp(0, -eps / T) * T
    return (w * f).sum() / (2 * math.pi)


def thermo(beta, mu, g, lam=0.5):
    T = 1.0 / beta
    p = lambda m=mu, t=T, gg=g: pressure(m, t, gg, lam)
    hm, ht = 1e-3 * max(1.0, abs(mu)), 1e-3 * T
    p0, pp, pm = p(), p(mu + hm), p(mu - hm)
    n = (pp - pm) / (2 * hm)
    dn = (pp - 2 * p0 + pm) / hm ** 2
    s = (p(t=T + ht) - p(t=T - ht)) / (2 * ht)
    out = dict(p=p0, n=n, dN2_L=T * dn, E_L=mu * n - p0 + T * s)
    if g > 0:
        hg = 1e-3 * g
        out["Vint_L"] = -g * (p(gg=g + hg) - p(gg=g - hg)) / (2 * hg)
        out["gamma"] = g / (2 * lam * n)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    beta, mu, g = map(float, a[:3])
    kw = dict(x.split("=") for x in a[3:])
    lam = float(kw.get("lam", 0.5))
    r = thermo(beta, mu, g, lam)
    print(f"beta={beta} mu={mu} g={g} lam={lam}")
    print("  ".join(f"{k}= {v:.7f}" for k, v in r.items()))
    if "L" in kw:
        L = float(kw["L"])
        print(f"L={L}: N= {r['n'] * L:.6f}  dN2= {r['dN2_L'] * L:.6f}  E= {r['E_L'] * L:.6f}"
              + (f"  Vbar= {r['Vint_L'] * L:.6f}" if "Vint_L" in r else ""))
