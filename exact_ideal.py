#!/usr/bin/env python3
"""
Exact grand-canonical reference for the IDEAL Bose gas in a periodic box (side L, d dims),
H = -lam * nabla^2.  The primitive approximation is exact for free particles (up to the
minimum-image truncation of each link, error ~ exp(-L^2/(16 lam tau))), so this is a strict
test of the pbc code at any P.

Cycle expansion: ln Xi = sum_j e^{beta mu j}/j * z(j)^d, with the 1D single-particle trace
at imaginary time j*beta split by winding number w:
    z(j,w) = L (4 pi lam j beta)^{-1/2} exp(-(w L)^2 / (4 lam j beta)),  z(j) = sum_w z(j,w).
Cycles of each (j, w) are independent Poisson variables, hence
    <N>   = sum_j e^{beta mu j} z(j)^d
    <D^2> = sum_j e^{beta mu j}/j * d * L^2 * (sum_w w^2 z(j,w)) * z(j)^{d-1}
    <E>   = -d ln Xi / d beta |_mu + mu <N>
(D = total winding displacement; rho_s/rho = <D^2>/(2 lam d beta <N>)).
Exchange cycles: the mean number of particles in cycles of length j is
    <n_j> = e^{beta mu j} z(j)^d        (sum_j <n_j> = <N>).
"""
import argparse, math


def z1(j, beta, lam, L, wmax=30):
    t = 4.0 * lam * j * beta
    pre = L / math.sqrt(math.pi * t)
    zs = [pre * math.exp(-(w * L) ** 2 / t) for w in range(-wmax, wmax + 1)]
    return sum(zs), sum(w * w * z for w, z in zip(range(-wmax, wmax + 1), zs))


def lnxi(beta, mu, d, lam, L, jmax):
    s, N, D2 = 0.0, 0.0, 0.0
    for j in range(1, jmax + 1):
        f = math.exp(beta * mu * j)
        if f < 1e-18:
            break
        z, zw2 = z1(j, beta, lam, L)
        s += f / j * z ** d
        N += f * z ** d
        D2 += f / j * d * L * L * zw2 * z ** (d - 1)
    return s, N, D2


def cycles(beta, mu, d, lam, L, J=12, jmax=100000):
    """<n_j> for j = 1..J-1, last entry: sum over j >= J"""
    n = [0.0] * J
    for j in range(1, jmax + 1):
        f = math.exp(beta * mu * j)
        if f < 1e-18:
            break
        n[min(j, J) - 1] += f * z1(j, beta, lam, L)[0] ** d
    return n


def observables(beta, mu, d, lam, L, jmax=100000):
    s, N, D2 = lnxi(beta, mu, d, lam, L, jmax)
    h = 1e-5 * beta
    E = -(lnxi(beta + h, mu, d, lam, L, jmax)[0] - lnxi(beta - h, mu, d, lam, L, jmax)[0]) / (2 * h) + mu * N
    return N, E, D2, D2 / (2 * lam * d * beta * N)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=-0.5)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--L", type=float, default=4.0)
    ap.add_argument("--J", type=int, default=12, help="cycle-length bins (last: j >= J)")
    a = ap.parse_args()
    if a.mu >= 0:
        raise SystemExit("need mu < 0")
    N, E, D2, rs = observables(a.beta, a.mu, a.d, a.lam, a.L)
    print("ideal  <N> = %.5f   <E> = %.5f   <D2> = %.5f   rhos = %.5f" % (N, E, D2, rs))
    n = cycles(a.beta, a.mu, a.d, a.lam, a.L, a.J)
    print("cycles " + " ".join("n%d%s= %.6f" % (j + 1, "+" if j == a.J - 1 else "", x) for j, x in enumerate(n)))
