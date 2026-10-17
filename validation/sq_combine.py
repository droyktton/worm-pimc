"""Combine validation/sq.sh: per case the static structure factor S(q) = <|rho_q|^2>/<N> (connected S_c with
disorder) as mean and standard error over seeds, against
  t10, t10d1, t40b*: exact_tonks.py (same finite ring and disorder realisation)
  all cases:         the harmonic Luttinger form S_L(q) = (K q / 2 pi n) coth(beta v q / 2), v = 2 lam pi n / K,
                     with K = 1 (Tonks) or the Bethe-ansatz K(gamma) (exact_ll_gs.py)
Writes validation/out/sq/<case>.dat (q, S, err, exact, S_L) for the figure.
usage (from the repository root, on a compute node): python3 validation/sq_combine.py"""
import glob, math, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
import exact_tonks, exact_ll_gs

OUT = "validation/out/sq"
CASES = {"t10": dict(L=10, beta=2, mu=1, g=None, P=40),
         "t10d1": dict(L=10, beta=2, mu=1, g=None, P=80, Vd=1.0),
         "t40b1": dict(L=40, beta=1, mu=4.9348, g=None, P=20),
         "t40b3": dict(L=40, beta=3, mu=4.9348, g=None, P=60),
         "t40b10": dict(L=40, beta=10, mu=4.9348, g=None, P=200),
         "ll2.5": dict(L=40, beta=10, mu=1.44231, g=2.5, P=400),
         "ll8": dict(L=40, beta=10, mu=2.83725, g=8.0, P=400)}


def parse(f):
    t = open(f).read()
    if "RESULT3" not in t:
        return None
    N = float(re.search(r"RESULT N= +([0-9.]+)", t).group(1))
    sq = [tuple(map(float, x)) for x in re.findall(r"SQ m= \d+ q= ([0-9.]+) S= ([0-9.e+-]+) \+- [0-9.e+-]+ Sc= ([0-9.e+-]+)", t)]
    return N, np.array(sq)


def main():
    runs = defaultdict(list)
    for f in glob.glob(f"{OUT}/*_s*.txt"):
        m = re.match(r"(.+)_s\d+\.txt", os.path.basename(f))
        r = parse(f)
        if m and r and m.group(1) in CASES:
            runs[m.group(1)].append(r)
    for case in [c for c in CASES if c in runs]:
        c, r = CASES[case], runs[case]
        L = c["L"]
        N = np.array([x[0] for x in r]); q = r[0][1][:, 0]
        col = 2 if "Vd" in c else 1
        S = np.array([x[1][:, col] for x in r]); Sm = S.mean(0); Se = S.std(0, ddof=1) / math.sqrt(len(r))
        n = N.mean() / L
        if c["g"] is None:
            K = 1.0
            Nq, r2, r1 = exact_tonks.structure(L, c["beta"], 0.5, c["mu"],
                                               exact_tonks.disorder(int(4 * L / (2 * math.pi)), c["Vd"], 1) if "Vd" in c else [],
                                               max(80, int(4 * L)), len(q))
            ex = (r2 - (np.abs(r1) ** 2 if "Vd" in c else 0)) / Nq
        else:
            K = exact_ll_gs.K_of_gamma(c["g"] / n)[0]; ex = None
        v = math.pi * n / K
        SL = K * q / (2 * math.pi * n) / np.tanh(c["beta"] * v * q / 2)
        print(f"\n### {case}: L={L} beta={c['beta']} mu={c['mu']} g={'inf' if c['g'] is None else c['g']}"
              + (f" Vd={c['Vd']}" if "Vd" in c else "") + f" P={c['P']}, {len(r)} seeds; n={n:.4f}, K={K:.4f}, v={v:.4f}\n")
        print("| m | q | S (MC) | exact | dev. | Luttinger (harmonic) |" if ex is not None else "| m | q | S (MC) | Luttinger (harmonic) | dev. |")
        print("|---|---|---|---|---|---|" if ex is not None else "|---|---|---|---|---|")
        for i in range(len(q)):
            if ex is not None:
                print(f"| {i + 1} | {q[i]:.4f} | {Sm[i]:.5f}({Se[i] * 1e5:.0f}) | {ex[i]:.5f} | {(Sm[i] - ex[i]) / Se[i]:+.1f}σ | {SL[i]:.5f} |")
            else:
                print(f"| {i + 1} | {q[i]:.4f} | {Sm[i]:.5f}({Se[i] * 1e5:.0f}) | {SL[i]:.5f} | {(Sm[i] - SL[i]) / Se[i]:+.1f}σ |")
        if ex is not None:
            x2 = (((Sm - ex) / Se) ** 2).sum()
            print(f"\nχ² against exact: {x2:.1f}/{len(q)}")
        np.savetxt(f"{OUT}/{case}.dat", np.c_[q, Sm, Se, ex if ex is not None else np.full(len(q), np.nan), SL],
                   header="q S err S_exact S_Luttinger_harmonic")


if __name__ == "__main__":
    main()
