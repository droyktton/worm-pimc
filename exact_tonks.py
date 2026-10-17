#!/usr/bin/env python3
"""Exact grand-canonical thermodynamics of hard-core (Tonks-Girardeau) bosons on a 1D ring of length L,
optionally in the disorder potential of worm_pimc.cpp (Vd, qc, dseed), by the Bose-Fermi mapping.

N hard-core bosons with twist phi map to free fermions with twist phi (N odd) or phi + pi (N even), so
    Z_N(phi) = e_N(x_1(phi_N), x_2(phi_N), ...),   x_n = exp(-beta eps_n),   phi_N = phi + pi [N even]
(e_N: elementary symmetric polynomial). The winding-number distribution follows from
    Z(phi) = sum_W Z_W exp(i W phi)  =>  P(W) = (1/2pi) int dphi exp(-i W phi) Z(phi) / Z(0).
Single-particle levels: H = -lam d^2/dx^2 + V(x) in the plane-wave basis k = (2 pi n + phi)/L.
Prints P(N), <N>, <dN^2>, E, P(W), <W^2> and K_th (theta-function form, as validation/lutt_combine.py).
With nsq=M also <|rho_q|^2> and <rho_q> (rho_q = sum_i exp(i q x_i), q = 2 pi m/L, m = 1..M): density
correlations of hard-core bosons equal those of the mapped fermions, so in each boundary-condition sector
    sum_N z^N Z_N <O>_N = Xi(z) <O>_GC(z),   Xi(z) = prod_a (1 + z x_a)
with Wick's theorem at complex fugacity, f_a = z x_a/(1 + z x_a), M_ab = <a|e^{iqx}|b>:
    <rho_q> = sum_a f_a M_aa,   <|rho_q|^2> = sum_a f_a M_aa * sum_a f_a conj(M_aa) + sum_ab |M_ab|^2 f_a (1 - f_b),
and the N-parity projection is (F(r) +- F(-r))/2 with r = exp(beta mu).

usage: python3 exact_tonks.py L beta mu [lam=0.5] [Vd=0 qc=2 dseed=1] [nmax=200] [nsq=0]"""
import math, sys
import numpy as np


def disorder(M, Vd, seed):
    """c_k = Vd sqrt(2/M) exp(i phi_k), same splitmix64 stream as worm_pimc.cpp"""
    c, mask = [], (1 << 64) - 1
    for _ in range(M):
        seed = (seed + 0x9e3779b97f4a7c15) & mask
        z = seed
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & mask
        z ^= z >> 31
        ph = 2 * math.pi * (z >> 11) * 2.0 ** -53
        c.append(Vd * math.sqrt(2.0 / M) * complex(math.cos(ph), math.sin(ph)))
    return c


def levels(L, lam, phi, c, nmax, vecs=False):
    n = np.arange(-nmax, nmax + 1)
    k = (2 * np.pi * n + phi) / L
    if not c and not vecs:
        return np.sort(lam * k * k)
    H = np.diag(lam * k * k).astype(complex)
    # V(x) = Re sum_k c_k e^{2 pi i k x/L} = sum_k (c_k e^{iqx} + conj(c_k) e^{-iqx}) / 2 couples n -> n +- k
    for j, ck in enumerate(c, start=1):
        H += np.diag(np.full(2 * nmax + 1 - j, ck / 2), -j) + np.diag(np.full(2 * nmax + 1 - j, np.conj(ck) / 2), j)
    return np.linalg.eigh(H) if vecs else np.linalg.eigvalsh(H)


def structure(L, beta, lam, mu, c, nmax, nsq):
    """<N>, <|rho_q|^2>, <rho_q> (m = 1..nsq) of hard-core bosons, grand canonical, by the parity projection"""
    r = math.exp(beta * mu)
    num = {"N": 0.0, "r2": np.zeros(nsq), "r1": np.zeros(nsq, complex)}; den = 0.0
    secs = []
    for par, ph in ((1, 0.0), (0, math.pi)):          # odd N: periodic fermions, even N: antiperiodic
        e, U = levels(L, lam, ph, c, nmax, vecs=True)
        x = np.exp(-beta * (e - mu))                   # r x_a with r = e^{beta mu} absorbed
        Mq = [U.conj().T @ shift(U, m) for m in range(1, nsq + 1)]
        secs.append((par, x, Mq))
    # log Xi(r) of each sector, to weight the two sectors on a common scale
    lx = [np.log1p(x).sum() for _, x, _ in secs]; l0 = max(lx)
    for (par, x, Mq), lp in zip(secs, lx):
        sgn = 1 if par == 0 else -1                    # even: (F(r) + F(-r))/2, odd: (F(r) - F(-r))/2
        rat = np.prod((1 - x) / (1 + x))               # Xi(-r) / Xi(r)
        w = math.exp(lp - l0)
        for z, fac in ((1.0, 0.5 * w), (-1.0, 0.5 * w * sgn * rat)):
            f = z * x / (1 + z * x)
            den += fac
            num["N"] += fac * f.sum()
            for i, M in enumerate(Mq):
                d = np.diag(M); t1 = (f * d).sum()
                num["r1"][i] += fac * t1
                num["r2"][i] += fac * (t1 * (f * d.conj()).sum()
                                       + np.einsum("ab,a,b->", np.abs(M) ** 2, f, 1 - f)).real
    return num["N"] / den, num["r2"] / den, num["r1"] / den


def shift(U, m):
    """e^{i q_m x} acting on plane-wave coefficients: component n -> n + m"""
    V = np.zeros_like(U); V[m:] = U[:-m]
    return V


def esym(x, Nmax):
    """log of elementary symmetric polynomials e_0..e_Nmax of x (scaled DP)"""
    e = np.zeros(Nmax + 1); e[0] = 1.0; ls = 0.0
    for xi in x:
        e[1:] = e[1:] + xi * e[:-1]
        s = e.max(); e /= s; ls += math.log(s)
    with np.errstate(divide="ignore"):
        return np.log(e) + ls


def logZN(L, beta, lam, phi, c, nmax, Nmax):
    """log Z_N(phi) for N = 0..Nmax (hard-core bosons, twist phi)"""
    out = np.empty(Nmax + 1)
    for par, ph in ((1, phi), (0, phi + math.pi)):
        eps = levels(L, lam, ph, c, nmax)
        lz = esym(np.exp(-beta * (eps - eps[0])), Nmax) - beta * eps[0] * np.arange(Nmax + 1)
        out[par::2] = lz[par::2]
    return out


def main(a):
    L, beta, mu = map(float, a[:3])
    kw = dict(x.split("=") for x in a[3:])
    lam = float(kw.get("lam", 0.5)); Vd = float(kw.get("Vd", 0)); qc = float(kw.get("qc", 2))
    dseed = int(kw.get("dseed", 1)); nmax = int(kw.get("nmax", 200))
    c = disorder(int(math.floor(qc * L / (2 * math.pi))), Vd, dseed) if Vd else []
    Nmax = int(kw.get("Nmax", 4 * nmax // 5))
    Ns = np.arange(Nmax + 1)

    def dist(b, ph=0.0):
        lw = logZN(L, b, lam, ph, c, nmax, Nmax) + b * mu * Ns
        return lw
    lw = dist(beta); m = lw.max(); w = np.exp(lw - m); Z = w.sum(); p = w / Z
    N = (p * Ns).sum(); dN2 = (p * (Ns - N) ** 2).sum()
    h = 1e-5 * beta                       # E = -d ln Z / d beta + mu N
    lzp = np.log(np.exp(dist(beta + h) - m).sum()); lzm = np.log(np.exp(dist(beta - h) - m).sum())
    E = -(lzp - lzm) / (2 * h) + mu * N
    # winding: Fourier transform of Z(phi)/Z(0) on a phi grid
    nphi = 64
    zphi = np.array([np.exp(dist(beta, 2 * np.pi * j / nphi) - m).sum() for j in range(nphi)]) / Z
    PW = np.real(np.fft.fft(zphi)) / nphi           # P(W) for W = 0, 1, ..., -1
    Wv = np.fft.fftfreq(nphi, 1.0 / nphi)
    W2 = (PW * Wv ** 2).sum()
    print(f"L={L} beta={beta} mu={mu} lam={lam} Vd={Vd} qc={qc} dseed={dseed} modes={len(c)} nmax={nmax}")
    print(f"N= {N:.7f}  dN2= {dN2:.7f}  E= {E:.7f}  W2= {W2:.7e}")
    print("PN " + " ".join(f"{n}:{q:.6g}" for n, q in zip(Ns, p) if q > 1e-8))
    print("PW " + " ".join(f"{int(x)}:{q:.6g}" for x, q in sorted(zip(Wv, PW)) if abs(q) > 1e-9))
    # theta-function K from the exact P(N), P(W)
    sys.path.insert(0, "validation")
    import lutt_combine as lc
    a_, nu = lc.ml_a({int(n): q * 1e9 for n, q in zip(Ns, p) if q > 1e-12})
    b_ = lc.ml_b({int(x): max(q, 0) * 1e9 for x, q in zip(Wv, PW) if q > 1e-12})
    nsq = int(kw.get("nsq", 0))
    if nsq:
        Nq, r2, r1 = structure(L, beta, lam, mu, c, nmax, nsq)
        print(f"structure factor (check: N= {Nq:.7f}); S = <|rho_q|^2>/<N>, Sc = (<|rho_q|^2> - |<rho_q>|^2)/<N>")
        for m in range(nsq):
            print(f"SQ m= {m + 1} q= {2 * math.pi * (m + 1) / L:.6f} S= {r2[m] / N:.7f} Sc= {(r2[m] - abs(r1[m]) ** 2) / N:.7f} rho2= {r2[m]:.7f}")
    print(f"K_G= {math.pi * math.sqrt(W2 * dN2):.6f}  K_th= {math.pi / (2 * math.sqrt(a_ * b_)):.6f}  (a={a_:.4f} nu={nu:.4f} b={b_:.4f})")


if __name__ == "__main__":
    main(sys.argv[1:])
