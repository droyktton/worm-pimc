"""Combine the seeds of validation/luttinger.sh: mean and standard error over chains of rho=N/L, E/N,
<dN^2>, <W^2>, Zfrac and the Luttinger parameter in two forms:
  K_G  = pi sqrt(<W^2><dN^2>)                       (Gaussian zero modes; valid only if <W^2>, <dN^2> >~ 1)
  K_th = pi / (2 sqrt(a b))                         (discrete Gaussian = theta-function zero modes)
where P(N) ~ exp(-a (N - nu)^2) over integers N and P(W) ~ exp(-b W^2) over integers W, with
a = pi u beta / (2 K L) and b = pi L / (2 K u beta). a, nu, b are obtained from the moments <N>, <dN^2>,
<W^2> or, when the outputs carry HISTN/HISTW lines, by maximum likelihood on the pooled histograms.
Errors: standard error over seeds (rho, E/N, ...) and bootstrap over seeds (K_G, K_th).
usage: python3 validation/lutt_combine.py [validation/out/lutt]"""
import glob, math, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq, minimize

d = sys.argv[1] if len(sys.argv) > 1 else "validation/out/lutt"
NR = np.arange(-30, 31)


def dg_moments(a, nu):
    """mean and variance of the discrete Gaussian exp(-a (n - nu)^2) over integers n"""
    n = np.floor(nu) + NR
    lw = -a * (n - nu) ** 2
    w = np.exp(lw - lw.max()); w /= w.sum()
    m = (w * n).sum()
    return m, (w * (n - m) ** 2).sum()


def a_from_moments(m, v):
    """(a, nu) with the given mean and variance; nan if v is below what the fractional part of m allows"""
    def solve_nu(a):                        # mean(a, nu) = m, monotonic in nu
        return brentq(lambda nu: dg_moments(a, nu)[0] - m, m - 0.6, m + 0.6)
    f = lambda la: dg_moments(math.exp(la), solve_nu(math.exp(la)))[1] - v
    lo, hi = math.log(1e-3), math.log(60.0)
    if f(lo) * f(hi) > 0:
        return float("nan"), float("nan")
    la = brentq(f, lo, hi)
    return math.exp(la), solve_nu(math.exp(la))


def b_from_moment(w2):
    g = lambda lb: (lambda w: (w * NR ** 2).sum() / w.sum())(np.exp(-math.exp(lb) * NR ** 2)) - w2
    try:
        return math.exp(brentq(g, math.log(1e-4), math.log(60.0)))
    except ValueError:
        return float("nan")


def ml_a(h):
    """maximum likelihood (a, nu) of a discrete Gaussian on the histogram {N: count}"""
    n = np.array(list(h)); c = np.array([h[k] for k in h], float)
    def nll(p):
        a, nu = math.exp(p[0]), p[1]
        g = np.floor(nu) + NR
        lz = np.logaddexp.reduce(-a * (g - nu) ** 2)
        return -(c * (-a * (n - nu) ** 2 - lz)).sum()
    m = (n * c).sum() / c.sum(); v = (c * (n - m) ** 2).sum() / c.sum()
    r = minimize(nll, [math.log(0.5 / max(v, 1e-3)), m], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-9))
    return math.exp(r.x[0]), r.x[1]


def ml_b(h):
    w = np.array(list(h)); c = np.array([h[k] for k in h], float)
    nll = lambda lb: -(c * (-math.exp(lb) * w ** 2 - np.logaddexp.reduce(-math.exp(lb) * NR ** 2))).sum()
    r = minimize(lambda p: nll(p[0]), [0.0], method="Nelder-Mead")
    return math.exp(r.x[0])


def hist(t, key):
    m = re.search(rf"^{key} (.*)$", t, re.M)
    return {int(k): int(v) for k, v in (x.split(":") for x in m.group(1).split())} if m else None


runs = defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "*.txt"))):
    t = open(f).read()
    if "RESULT3" not in t:
        continue
    g = lambda k: float(re.search(rf"\b{k}= +(-?[0-9.]+(?:e[-+][0-9]+)?)", t).group(1))
    L = float(re.search(r"Lbox=([0-9.]+)", t).group(1))
    runs[re.sub(r"_s\d+\.txt$", "", os.path.basename(f))].append(
        dict(N=g("N"), rho=g("N") / L, e=g("E") / g("N"), dN2=g("dN2"), W2=g("W2"), Z=g("Zfrac"),
             hN=hist(t, "HISTN"), hW=hist(t, "HISTW")))


def kvals(r):
    """K_G, K_th, a, nu, b from a list of seeds"""
    mean = lambda q: np.mean([x[q] for x in r])
    KG = math.pi * math.sqrt(mean("W2") * mean("dN2"))
    if all(x["hN"] and x["hW"] for x in r):
        hN, hW = defaultdict(int), defaultdict(int)
        for x in r:
            for k, v in x["hN"].items(): hN[k] += v
            for k, v in x["hW"].items(): hW[k] += v
        (a, nu), b = ml_a(hN), ml_b(hW)
    else:
        (a, nu), b = a_from_moments(mean("N"), mean("dN2")), b_from_moment(mean("W2"))
    return KG, math.pi / (2 * math.sqrt(a * b)), a, nu, b


rng = np.random.default_rng(1)
print(f"{'run':24s} {'n':>2s} {'rho':>15s} {'E/N':>15s} {'dN2':>14s} {'W2':>14s} {'K_G':>12s} {'K_th':>12s}"
      f" {'a':>6s} {'frac(nu)':>8s} {'b':>6s} {'Zfrac':>5s} src")
m = lambda a: (np.mean(a), np.std(a, ddof=1) / math.sqrt(len(a)) if len(a) > 1 else float("nan"))
for k in sorted(runs, key=lambda s: [float(x) if x.replace('.', '').isdigit() else x for x in re.findall(r"[0-9.]+|\D+", s)]):
    r = runs[k]
    v = {q: m([x[q] for x in r]) for q in ("rho", "e", "dN2", "W2", "Z")}
    KG, Kt, a, nu, b = kvals(r)
    bs = np.array([kvals([r[i] for i in rng.integers(0, len(r), len(r))])[:2] for _ in range(200)]) if len(r) > 2 else None
    e = np.nanstd(bs, axis=0) if bs is not None else (float("nan"),) * 2
    f = lambda q, p=4: f"{v[q][0]:.{p}f}({v[q][1]:.{p}f})"
    src = "hist" if all(x["hN"] and x["hW"] for x in r) else "mom"
    print(f"{k:24s} {len(r):2d} {f('rho'):>15s} {f('e'):>15s} {f('dN2', 3):>14s} {f('W2', 3):>14s} {KG:6.3f}({e[0]:.3f})"
          f" {Kt:6.3f}({e[1]:.3f}) {a:6.3f} {nu - math.floor(nu):8.3f} {b:6.3f} {v['Z'][0]:5.2f} {src}")
