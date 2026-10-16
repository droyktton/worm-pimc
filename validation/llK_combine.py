"""Compare validation/llK.sh (clean Lieb-Liniger gas, n = 1) with the Bethe-ansatz ground state (exact_ll_gs.py):
per gamma and tau the density n = N/L, gamma = g/(2 lam n), E/N against lam n^2 e(gamma), K_G and K_th
(pooled N/W histograms, bootstrap over seeds) against K(gamma).
usage (from the repository root, on a compute node): python3 validation/llK_combine.py"""
import glob, math, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "validation")
import contact_combine as cc
import exact_ll_gs

OUT, L = "validation/out/llK", 20.0
G = {"1": 1.0, "2.5": 2.5, "8": 8.0}


def main():
    runs = defaultdict(list)
    for f in glob.glob(f"{OUT}/*.txt"):
        m = re.match(r"g([\d.]+)_t([\d.]+)_s\d+\.txt", os.path.basename(f))
        r = cc.parse(f)
        if m and r:
            runs[(m.group(1), float(m.group(2)))].append(r)
    rng = np.random.default_rng(3)
    print("| γ (input) | τ | seeds | n | γ = g/n | E/N | E/N exact | K_G | K_th | K exact | K_th − K |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(runs, key=lambda k: (G[k[0]], -k[1])):
        r, g = runs[key], G[key[0]]
        se = lambda a: np.std(a, ddof=1) / math.sqrt(len(a))
        n = np.array([x["N"] for x in r]) / L
        eN = np.array([x["E"] / x["N"] for x in r])
        KG = np.array([math.pi * math.sqrt(x["W2"] * x["dN2"]) for x in r])
        K = cc.kth(r)
        eK = float(np.nanstd([cc.kth([r[i] for i in rng.integers(0, len(r), len(r))]) for _ in range(200)]))
        gam = g / n.mean()
        Kx, e = exact_ll_gs.K_of_gamma(gam)
        Ex = 0.5 * n.mean() ** 2 * e
        print(f"| {key[0]} | {key[1]} | {len(r)} | {n.mean():.4f}({se(n):.1g}) | {gam:.3f} | {eN.mean():.4f}({se(eN):.1g}) "
              f"| {Ex:.4f} | {KG.mean():.3f}({se(KG):.2g}) | {K:.3f}({eK:.2g}) | {Kx:.3f} | {(K - Kx) / eK:+.1f}σ |")


if __name__ == "__main__":
    main()
