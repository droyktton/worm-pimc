#!/usr/bin/env python3
"""
Exact reference for bosons in an isotropic harmonic trap with an optional
harmonic pair interaction  (H = sum p^2/2 + w^2/2 sum x^2 + kappa/2 sum_{i<j}(x_i-x_j)^2,
hbar = m = 1), in the grand-canonical ensemble.

Two versions:
  * 'PA'        : the *primitive-approximation* discretisation with P slices that the
                  worm code samples (strict test, no discretisation error);
  * 'continuum' : P -> infinity limit (exact quantum result).

Method: Z_N = (1/N!) sum_perm tr(perm e^{-bH}).  For a quadratic H the trace factorises
over normal modes; a permutation acts on the N coordinates through an orthogonal matrix
whose eigenvalues are roots of unity (one set per cycle).  The centre-of-mass mode has
frequency w, the N-1 relative modes sqrt(w^2 + N kappa).
For each eigenvalue lambda of the permutation matrix and mode frequency w_k:
    continuum:  e^{-b w_k/2} / (1 - lambda e^{-b w_k})
    PA (P slices): lam0(w_k)^P / (1 - lambda rho(w_k)^P)
where the primitive-approximation kernel of a 1D oscillator has eigenvalues lam0*rho^n
(Mehler formula):  rho + 1/rho = 2 + tau^2 w^2.
<E>: thermodynamic estimator, -d ln Xi/d beta at fixed P and mu, plus mu<N>.
<V> (mean over slices of the potential energy) = -(1/beta) d ln Z / d s at s=1 where
the potential is scaled by s (w^2 -> s w^2, kappa -> s kappa).
"""
import argparse, math, cmath


def partitions(n, maxp=None):
    if maxp is None:
        maxp = n
    if n == 0:
        yield []
        return
    for k in range(min(n, maxp), 0, -1):
        for rest in partitions(n - k, k):
            yield [k] + rest


def pa_params(w, tau):
    ch = 1.0 + 0.5 * tau * tau * w * w
    rho = ch - math.sqrt(ch * ch - 1.0)
    B = 1.0 / tau
    alpha2 = B * (1.0 - rho * rho) / (2.0 * rho)
    c = (2.0 * math.pi * tau) ** -0.5
    lam0 = c * math.sqrt(math.pi) * math.sqrt(1.0 - rho * rho) / math.sqrt(alpha2)
    return lam0, rho


def mode_factor(lam, w, beta, P):
    if P is None:                       # continuum
        x = math.exp(-beta * w)
        return math.exp(-0.5 * beta * w) / (1.0 - lam * x)
    tau = beta / P
    lam0, rho = pa_params(w, tau)
    return lam0 ** P / (1.0 - lam * rho ** P)


def ZN(N, d, beta, omega, kappa, P, s=1.0):
    if N == 0:
        return 1.0
    wc = omega * math.sqrt(s)
    wr = math.sqrt(s * (omega ** 2 + N * kappa))
    total = 0.0
    for part in partitions(N):
        # symmetry factor 1/prod(c^{m_c} m_c!)
        sym = 1.0
        for c in set(part):
            m = part.count(c)
            sym *= (c ** m) * math.factorial(m)
        t = 1.0 + 0j
        ncyc = len(part)
        # eigenvalue 1: one COM mode, ncyc-1 relative modes
        t *= mode_factor(1.0, wc, beta, P)
        for _ in range(ncyc - 1):
            t *= mode_factor(1.0, wr, beta, P)
        for c in part:
            for k in range(1, c):
                t *= mode_factor(cmath.exp(2j * math.pi * k / c), wr, beta, P)
        total += (t.real ** d if abs(t.imag) < 1e-9 * abs(t.real) else float('nan')) / sym
    return total


def grand(beta, mu, d, omega, kappa, P, Nmax, s=1.0):
    Z = 0.0
    ZNs = []
    for N in range(Nmax + 1):
        z = math.exp(beta * mu * N) * ZN(N, d, beta, omega, kappa, P, s)
        ZNs.append(z)
        Z += z
    return Z, ZNs


def observables(beta, mu, d, omega, kappa, P, Nmax=12):
    Z, ZNs = grand(beta, mu, d, omega, kappa, P, Nmax)
    Nav = sum(N * z for N, z in enumerate(ZNs)) / Z
    h = 1e-4
    lp = math.log(grand(beta, mu, d, omega, kappa, P, Nmax, 1 + h)[0])
    lm = math.log(grand(beta, mu, d, omega, kappa, P, Nmax, 1 - h)[0])
    Vbar = -(lp - lm) / (2 * h) / beta
    tail = ZNs[-1] / Z
    # thermodynamic energy at fixed P and mu: <E> = -d ln Xi/d beta + mu <N>
    hb = 1e-5 * beta
    lbp = math.log(grand(beta + hb, mu, d, omega, kappa, P, Nmax)[0])
    lbm = math.log(grand(beta - hb, mu, d, omega, kappa, P, Nmax)[0])
    E = -(lbp - lbm) / (2 * hb) + mu * Nav
    return Nav, Vbar, tail, E


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=1)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--P", type=int, default=24)
    ap.add_argument("--Nmax", type=int, default=12)
    a = ap.parse_args()
    for label, P in (("PA  P=%d" % a.P, a.P), ("continuum", None)):
        N, V, tl, E = observables(a.beta, a.mu, a.d, a.omega, a.kappa, P, a.Nmax)
        print("%-12s <N> = %.5f   <Vbar> = %.5f   (weight of N=Nmax: %.1e)   <E> = %.5f" % (label, N, V, tl, E))
