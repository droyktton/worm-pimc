#!/usr/bin/env python3
"""Figures for docs/worm_pimc.md from demos/out/*.csv and docs/data/bench_lj.csv.
usage (from the repository root, on a compute node): python3 docs/make_figures.py"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated palette)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})
OUT = "docs/figures"


def read(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(l for l in f if not l.startswith("#"))]


def superfluid():
    p = "demos/out/superfluid.csv"
    if not os.path.exists(p):
        return
    d = sorted(read(p), key=lambda r: r["T"])
    T = [r["T"] for r in d]
    # dense exact curve at the same fixed <N> = 8 (mu(T) by bisection)
    import sys; sys.path.insert(0, ".")
    from exact_ideal import observables
    def exact_at(beta, n=8.0, L=4.0):
        lo, hi = -20.0, -1e-9
        for _ in range(60):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if observables(beta, m, 3, 0.5, L)[0] < n else (lo, m)
        N, E, D2, rs = observables(beta, 0.5 * (lo + hi), 3, 0.5, L)
        return {"E": E, "rhos": rs}
    Td = [0.3 + 0.05 * i for i in range(63)]
    ex = [exact_at(1 / t) for t in Td]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, key, lab in ((ax[0], "rhos", r"superfluid fraction $\rho_s/\rho$"), (ax[1], "E", r"energy $E$")):
        a.plot(Td, [e[key] for e in ex], color=S1, label="exact (cycle expansion)", zorder=2)
        a.errorbar(T, [r[key + "_mc"] for r in d], yerr=[r[key + "_err"] for r in d], fmt="o", color=S2,
                   mec=SURF, mew=1.5, elinewidth=1.2, capsize=0, label="worm PIMC, GPU", zorder=3)
        a.set_xlabel(r"temperature $T$  ($\hbar=m=1$)")
        a.set_ylabel(lab)
    ax[0].legend(loc="upper right")
    fig.suptitle(r"Ideal Bose gas, 3D periodic box $L=4$, fixed $\langle N\rangle=8$ (bulk $T_c\approx0.83$)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/superfluid.png", dpi=150)


def lj_eos():
    p = "demos/out/lj_eos.csv"
    if not os.path.exists(p):
        return
    d = sorted(read(p), key=lambda r: r["mu"])
    mu = [r["mu"] for r in d]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].errorbar(mu, [r["rho"] for r in d], yerr=[r["rho_err"] for r in d], fmt="o-", color=S1,
                   mec=SURF, mew=1.5, elinewidth=1.2)
    ax[0].set_ylabel(r"density $\rho\sigma^3$")
    ax[1].errorbar(mu, [r["E_per_N"] for r in d], yerr=[r["E_per_N_err"] for r in d], fmt="o-", color=S1,
                   mec=SURF, mew=1.5, elinewidth=1.2)
    ax[1].set_ylabel(r"energy per particle $E/N\varepsilon$")
    for a in ax:
        a.set_xlabel(r"chemical potential $\mu/\varepsilon$")
    fig.suptitle(r"Quantum Lennard-Jones fluid, $\lambda=0.05$, $T=1$, $P=32$, $L=4.5\sigma$ (GPU, W=32)",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/lj_eos.png", dpi=150)


def bench():
    d = read("docs/data/bench_lj.csv")
    N = [r["N"] for r in d]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, col, lab in (("gpu_w32_float", S1, "GPU W=32, float"), ("gpu_w32_double", S2, "GPU W=32, double"),
                          ("cpu_1core", S3, "CPU, 1 core")):
        y = [r[key] for r in d]
        ax.plot(N, y, "o-", color=col, mec=SURF, mew=1.5, label=lab)
        ax.annotate(lab, (N[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of particles N"); ax.set_ylabel("MC steps per second")
    ax.set_xlim(6, 1500)
    ax.legend(loc="lower left")
    ax.set_title("3D Lennard-Jones, RTX 5070 Ti vs Xeon Gold 6330", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/performance.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    superfluid(); lj_eos(); bench()
