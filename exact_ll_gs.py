#!/usr/bin/env python3
"""Ground state of the Lieb-Liniger gas (Lieb & Liniger, Phys. Rev. 130, 1605 (1963)): energy e(gamma) and
Luttinger parameter K(gamma), gamma = m g/(hbar^2 n) = g/(2 lam n).
Lieb equation on [-1, 1] (alpha = c/q):  2 pi rho(x) = 1 + int_{-1}^{1} 2 alpha/(alpha^2 + (x-y)^2) rho(y) dy,
gamma = alpha / int rho,   e = (gamma/alpha)^3 int x^2 rho   (E/N = (hbar^2 n^2/2m) e = lam n^2 e).
K = v_F / v_s = pi / sqrt(3 e - 2 gamma e' + gamma^2 e''/2)   (K = 1 in the Tonks limit).
usage: python3 exact_ll_gs.py gamma [gamma ...]"""
import math, sys
import numpy as np
from scipy.optimize import brentq

NG = 400
xg, wg = np.polynomial.legendre.leggauss(NG)


def lieb(alpha):
    A = np.eye(NG) - (2 * alpha / (alpha ** 2 + (xg[:, None] - xg[None, :]) ** 2)) * wg[None, :] / (2 * math.pi)
    rho = np.linalg.solve(A, np.full(NG, 1 / (2 * math.pi)))
    nrm = (wg * rho).sum()
    g = alpha / nrm
    return g, (g / alpha) ** 3 * (wg * xg ** 2 * rho).sum()


def e_of_gamma(gamma):
    a = brentq(lambda al: lieb(al)[0] - gamma, 1e-6, 1e6 * max(gamma, 1))
    return lieb(a)[1]


def K_of_gamma(gamma):
    h = 1e-3 * gamma
    e0, ep, em = e_of_gamma(gamma), e_of_gamma(gamma + h), e_of_gamma(gamma - h)
    d1, d2 = (ep - em) / (2 * h), (ep - 2 * e0 + em) / h ** 2
    return math.pi / math.sqrt(3 * e0 - 2 * gamma * d1 + 0.5 * gamma ** 2 * d2), e0


if __name__ == "__main__":
    for gm in map(float, sys.argv[1:] or ["0.5", "1", "2.5", "10", "100"]):
        K, e = K_of_gamma(gm)
        print(f"gamma= {gm:g}  e= {e:.8f}  K= {K:.6f}")
