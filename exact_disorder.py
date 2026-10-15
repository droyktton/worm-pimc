#!/usr/bin/env python3
"""
Exact grand-canonical reference for the IDEAL Bose gas in a 1D periodic box of side L with the
disorder potential of the worm codes (Vd, qc, dseed; same coefficients, splitmix64), for the
SAME primitive discretisation as the simulation (P slices, tau = beta/P): no Trotter error.

One-slice transfer matrix on a uniform grid of M points (trapezoid rule, spectrally accurate for
a periodic smooth kernel), with a twist theta to count windings:
    T_theta(x,x') = h e^{-tau V(x)/2} sum_n G0(x'-x+nL; tau) e^{i theta (x'-x+nL)/L} e^{-tau V(x')/2}
With Lambda_i(theta) = t_i(theta)^P (t_i eigenvalues of the Hermitian T_theta) and f = e^{beta mu}:
    ln Xi(theta) = -sum_i ln(1 - f Lambda_i)
    <N> = sum_i n_i, n_i = f Lambda_i / (1 - f Lambda_i),   <dN^2> = sum_i n_i (1 + n_i)
    <D^2> = -L^2 d^2 ln Xi / d theta^2 |_0                  (D = total winding displacement)
    <n_j> = sum_i (f Lambda_i)^j  (particles in exchange cycles of length j; last bin j >= J)
    Vbar = -(1/beta) d ln Xi / d alpha |_{alpha=1} for V -> alpha V
    E    = -d ln Xi / d beta |_mu + mu <N>                  (thermodynamic estimator, fixed P)
mu must lie below the lowest level of the discretised problem, -ln(t_0)/tau (printed as e0).
"""
import argparse, math
import numpy as np

M64 = (1 << 64) - 1


def coefficients(Vd, qc, L, dseed):
    """c_k, k = 1..M, exactly as disorder() in worm_pimc.cpp / worm_pimc_gpu.cu"""
    M = int(math.floor(qc * L / (2 * math.pi)))
    c, s = [], dseed & M64
    for _ in range(M):
        s = (s + 0x9E3779B97F4A7C15) & M64
        z = s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        z ^= z >> 31
        ph = 2 * math.pi * (z >> 11) * 2.0 ** -53
        c.append(Vd * math.sqrt(2.0 / M) * complex(math.cos(ph), math.sin(ph)))
    return np.array(c)


def vdis(x, c, L):
    k = np.arange(1, len(c) + 1)
    return np.real(np.exp(2j * np.pi * np.outer(x, k) / L) @ c)


class Box:
    def __init__(self, L, lam, Vd, qc, dseed, M=256, nimg=3):
        self.L, self.lam, self.M, self.nimg = L, lam, M, nimg
        self.h = L / M
        self.x = -0.5 * L + self.h * np.arange(M)
        self.V = vdis(self.x, coefficients(Vd, qc, L, dseed), L) if Vd != 0 else np.zeros(M)
        dx = self.x[None, :] - self.x[:, None]
        self.dx = dx - L * np.rint(dx / L)

    def levels(self, tau, P, theta=0.0, alpha=1.0):
        """Lambda_i = t_i^P"""
        L, t4 = self.L, 4 * self.lam * tau
        K = np.zeros((self.M, self.M), complex)
        for n in range(-self.nimg, self.nimg + 1):
            y = self.dx + n * L
            K += np.exp(-y * y / t4) * np.exp(1j * theta * y / L)
        w = np.exp(-0.5 * tau * alpha * self.V)
        K *= self.h / math.sqrt(math.pi * t4) * w[:, None] * w[None, :]
        t = np.linalg.eigvalsh(K)
        return t[t > 0] ** P

    def e0(self, beta, P):
        tau = beta / P
        return -math.log(self.levels(tau, P).max()) / beta

    def lnxi(self, beta, mu, P, theta=0.0, alpha=1.0):
        fl = math.exp(beta * mu) * self.levels(beta / P, P, theta, alpha)
        if fl.max() >= 1:
            raise ValueError("mu above the lowest level")
        return -np.sum(np.log1p(-fl))

    def observables(self, beta, mu, P, J=12):
        fl = math.exp(beta * mu) * self.levels(beta / P, P)
        if fl.max() >= 1:
            raise ValueError("mu above the lowest level")
        n = fl / (1 - fl)
        N, dN2 = n.sum(), (n * (1 + n)).sum()
        th, hb, ha = 0.02, 1e-4 * beta, 1e-4
        f = lambda **kw: self.lnxi(**{**dict(beta=beta, mu=mu, P=P), **kw})
        l0 = f()
        D2 = -self.L ** 2 * (f(theta=th) + f(theta=-th) - 2 * l0) / th ** 2
        E = -(f(beta=beta + hb) - f(beta=beta - hb)) / (2 * hb) + mu * N
        Vbar = -(f(alpha=1 + ha) - f(alpha=1 - ha)) / (2 * ha) / beta
        cyc = [np.sum(fl ** j) for j in range(1, J)] + [np.sum(fl ** J / (1 - fl))]
        W2 = D2 / self.L ** 2
        return dict(N=N, Vbar=Vbar, E=E, D2=D2, rhos=D2 / (2 * self.lam * beta * N), dN2=dN2, W2=W2,
                    K=math.pi * math.sqrt(W2 * dN2),
                    **{f"n{j + 1}" + ("+" if j == J - 1 else ""): x for j, x in enumerate(cyc)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--Vd", type=float, default=1.0)
    ap.add_argument("--qc", type=float, default=6.0)
    ap.add_argument("--dseed", type=int, default=1)
    ap.add_argument("--mu", type=float, help="chemical potential (default: e0 + dmu)")
    ap.add_argument("--dmu", type=float, default=-0.3, help="mu - e0 when --mu is not given")
    ap.add_argument("--M", type=int, default=256, help="grid points (convergence: compare with 2M)")
    ap.add_argument("--J", type=int, default=12, help="cycle-length bins (last: j >= J)")
    a = ap.parse_args()
    box = Box(a.L, a.lam, a.Vd, a.qc, a.dseed, a.M)
    e0 = box.e0(a.beta, a.P)
    mu = a.mu if a.mu is not None else e0 + a.dmu
    o = box.observables(a.beta, mu, a.P, a.J)
    print("e0= %.10g  mu= %.10g" % (e0, mu))
    print(" ".join("%s= %.8g" % kv for kv in o.items()))
