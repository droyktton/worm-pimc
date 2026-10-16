"""Compare validation/contact.sh (pot=3, 1D contact) with the exact results: per case and P the mean and
standard error over seeds of N, <dN^2>, E, Vbar (= g <sum delta>), <W^2>, K_th (pooled histograms), the
deviation from the exact value in sigma, and the tau -> 0 fit a + b tau^2.
  tonks: exact_tonks.py (hard-core bosons on the same finite ring, incl. P(W) and K_th)
  g2, g05: exact_yy.py (Yang-Yang, thermodynamic limit; finite-L corrections are exponentially small)
usage (from the repository root, on a compute node): python3 validation/contact_combine.py"""
import glob, io, math, os, re, sys
from collections import defaultdict
from contextlib import redirect_stdout
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "validation")
import lutt_combine as lc
import exact_tonks, exact_yy

OUT = "validation/out/contact"
CASES = {"tonks": dict(L=10, beta=2, mu=1, g=None), "g2": dict(L=20, beta=1, mu=1, g=2.0),
         "g05": dict(L=20, beta=1, mu=1, g=0.5)}


def exact(case):
    c = CASES[case]
    if c["g"] is None:
        f = io.StringIO()
        with redirect_stdout(f):
            exact_tonks.main([str(c["L"]), str(c["beta"]), str(c["mu"]), "nmax=80"])
        t = f.getvalue()
        g = lambda k: float(re.search(rf"\b{k}= +(-?[0-9.e+-]+)", t).group(1))
        return dict(N=g("N"), dN2=g("dN2"), E=g("E"), Vbar=0.0, W2=g("W2"), K_th=g("K_th"))
    r = exact_yy.thermo(c["beta"], c["mu"], c["g"])
    L = c["L"]
    return dict(N=r["n"] * L, dN2=r["dN2_L"] * L, E=r["E_L"] * L, Vbar=r["Vint_L"] * L)


def parse(f):
    t = open(f).read()
    if "RESULT3" not in t:
        return None
    g = lambda k: float(re.search(rf"\b{k}= +(-?[0-9.]+(?:e[-+][0-9]+)?)", t).group(1))
    return dict(N=g("N"), dN2=g("dN2"), E=g("E"), Vbar=g("Vbar"), W2=g("W2"),
                hN=lc.hist(t, "HISTN"), hW=lc.hist(t, "HISTW"))


def kth(r):
    hN, hW = defaultdict(int), defaultdict(int)
    for x in r:
        for k, v in x["hN"].items(): hN[k] += v
        for k, v in x["hW"].items(): hW[k] += v
    if len(hW) < 2:
        return float("nan")
    (a, nu), b = lc.ml_a(hN), lc.ml_b(hW)
    return math.pi / (2 * math.sqrt(a * b))


def main():
    runs = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(f"{OUT}/*.txt"):
        m = re.match(r"(\w+?)_P(\d+)_s\d+\.txt", os.path.basename(f))
        r = parse(f)
        if m and r:
            runs[m.group(1)][int(m.group(2))].append(r)
    rng = np.random.default_rng(2)
    for case in [c for c in CASES if c in runs]:
        ex = exact(case)
        c = CASES[case]
        print(f"\n### {case}: L={c['L']} beta={c['beta']} mu={c['mu']} g={'inf' if c['g'] is None else c['g']}\n")
        keys = [k for k in ("N", "dN2", "E", "Vbar", "W2", "K_th") if k in ex and not (k == "Vbar" and c["g"] is None)]
        print("| P | τ | seeds | " + " | ".join(keys) + " |")
        print("|---" * (len(keys) + 3) + "|")
        print("| exact | 0 | | " + " | ".join(f"{ex[k]:.5g}" for k in keys) + " |")
        fit = defaultdict(list)
        for P in sorted(runs[case]):
            r = runs[case][P]; tau = c["beta"] / P; cells = []
            for k in keys:
                if k == "K_th":
                    v = kth(r)
                    bs = [kth([r[i] for i in rng.integers(0, len(r), len(r))]) for _ in range(100)]
                    e = float(np.nanstd(bs))
                else:
                    a = [x[k] for x in r]; v = np.mean(a); e = np.std(a, ddof=1) / math.sqrt(len(a))
                fit[k].append((tau, v, e))
                cells.append(f"{v:.5g}({e:.2g}) {(v - ex[k]) / e:+.1f}σ")
            print(f"| {P} | {tau:.4g} | {len(r)} | " + " | ".join(cells) + " |")
        print()
        for k in keys:
            pts = [p for p in fit[k] if p[2] > 0]
            if len(pts) >= 3:
                a, ea, x2, nd = lc_fit(pts)
                print(f"- {k}: τ→0 (a + b τ², {len(pts)} P) = {a:.5g}({ea:.2g}), exact {ex[k]:.5g}: {(a - ex[k]) / ea:+.1f}σ, fit χ²/dof = {x2:.1f}/{nd}")


def lc_fit(pts):
    t2 = np.array([p[0] for p in pts]) ** 2; y = np.array([p[1] for p in pts]); w = 1 / np.array([p[2] for p in pts])
    A = np.vstack([np.ones_like(t2), t2]).T
    cov = np.linalg.inv((A * w[:, None] ** 2).T @ A)
    cf = cov @ (A * w[:, None] ** 2).T @ y
    res = (y - A @ cf) * w
    return cf[0], math.sqrt(cov[0, 0]), float(res @ res), len(y) - 2


if __name__ == "__main__":
    main()
