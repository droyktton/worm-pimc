#!/usr/bin/env python3
"""Validation figures for docs/worm_pimc.md (Section 9) from validation/out/.
usage (from the repository root, on a compute node): python3 validation/figures.py
output: docs/figures/val_trotter.png, val_cycles.png, val_lj100.png, val_sq.png"""
import math, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "validation")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analyze as an
import exact_pa as ep

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # same palette as docs/make_figures.py
BAND = "#efeeea"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})
FIG = "docs/figures"
DOT = dict(mec=SURF, mew=1.5, elinewidth=1.2, capsize=0)
MU = {1: 0.5, 2: 1.5, 3: 2.0}           # mu of the Trotter scans (validation/trotter.sh <d> <mu> <C>)
OBS = {"N": r"$N$", "Vbar": r"$\bar V$", "E": r"$E$", "rhos": r"$\rho_s/\rho$"}


def sigma_band(ax):
    ax.axhspan(-2, 2, color=BAND, zorder=0, lw=0)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)


def trotter():
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), gridspec_kw=dict(height_ratios=[3, 2]))
    for i, d in enumerate((1, 2, 3)):
        path = f"{an.OUT}/trotter_d{d}.csv"
        if not os.path.exists(path):
            continue
        R = an.rows(path)
        v = sorted((r for r in R if r["obs"] == "N"), key=lambda r: int(r["P"]))
        ct = float(v[0]["exact_cont"])
        Pd = (4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 64)
        pa = [ep.observables(2.0, MU[d], d, 1.0, 0.3, P, 22)[0] for P in Pd]
        a = ax[0, i]
        a.plot([0] + [(2.0 / P) ** 2 for P in Pd], [ct] + pa, color=S1, label="exact, primitive approx.", zorder=2)
        a.plot([0], [ct], "*", color=INK, ms=11, mec=SURF, label=r"exact continuum ($\tau\to0$)", zorder=4)
        a.errorbar([float(r["tau"]) ** 2 for r in v], [float(r["mc"]) for r in v],
                   yerr=[float(r["err"]) for r in v], fmt="o", color=S2, label="worm PIMC, GPU", zorder=3, **DOT)
        a.set_title(f"d = {d}", loc="left", fontsize=10)
        a.set_xlabel(r"$\tau^2$")
        b = ax[1, i]
        sigma_band(b)
        for k, c, off in (("N", S1, 0.94), ("Vbar", S2, 1.0), ("E", S3, 1.06)):
            w = sorted((r for r in R if r["obs"] == k), key=lambda r: int(r["P"]))
            b.plot([float(r["tau"]) ** 2 * off for r in w],
                   [(float(r["mc"]) - float(r["exact_PA"])) / float(r["err"]) for r in w],
                   "o", color=c, mec=SURF, mew=1.5, label=OBS[k], zorder=3)
        b.set_xscale("log"); b.set_ylim(-3.5, 3.5)
        b.set_xlabel(r"$\tau^2$  ($P = 64 \ldots 4$)")
    ax[0, 0].set_ylabel(r"$\langle N\rangle$")
    ax[1, 0].set_ylabel(r"(MC $-$ exact PA) / $\sigma$")
    ax[0, 0].legend(loc="upper left")
    ax[1, 0].legend(loc="lower left", ncol=3, handletextpad=0.2)
    fig.suptitle(r"Trotter scan: harmonic trap with harmonic coupling, $\omega=1$, $\kappa=0.3$, $\beta=2$ "
                 r"(shaded: $\pm2\sigma$)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/val_trotter.png", dpi=150)


def cycles():
    path = f"{an.OUT}/cycles_ideal.csv"
    if not os.path.exists(path):
        return
    R = an.rows(path)
    cases = list(dict.fromkeys(r["case"] for r in R))
    fig, ax = plt.subplots(1, len(cases), figsize=(3.1 * len(cases), 3.6), sharey=True)
    for a, case in zip(ax, cases):
        js = [f"n{j}" for j in range(1, 12)]
        get = lambda prec, k: next(r for r in R if r["case"] == case and r["prec"] == prec and r["obs"] == k)
        x = range(1, 12)
        a.plot(x, [float(get("double", k)["exact"]) for k in js], color=INK2, lw=1.2, label="exact", zorder=2)
        for prec, c, dx in (("double", S1, -0.12), ("float", S2, 0.12)):
            a.errorbar([j + dx for j in x], [float(get(prec, k)["mc"]) for k in js],
                       yerr=[float(get(prec, k)["err"]) for k in js], fmt="o", color=c, ms=5,
                       label=f"GPU {prec}", zorder=3, **DOT)
        d, beta, mu, L = (t.split("=")[1] for t in case.split())
        a.set_title(f"d={d}, β={beta}, μ={mu}, L={L}", loc="left", fontsize=9.5)
        a.set_yscale("log"); a.set_xlabel("cycle length $j$"); a.set_xticks([1, 3, 5, 7, 9, 11])
    ax[0].set_ylabel(r"particles in cycles of length $j$, $\langle n_j\rangle$")
    ax[0].legend(loc="lower left")
    fig.suptitle("Ideal Bose gas in a periodic box: exchange-cycle occupations", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/val_cycles.png", dpi=150)


def lj100():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[3, 16]), sharey=True)
    for a, sysn, keys in ((ax[0], "A", ["N", "Vbar", "E"]),
                          (ax[1], "B", ["N", "Vbar", "E", "rhos"] + [f"n{j}" for j in range(1, 12)] + ["n12+"])):
        cols = an.lj100_columns(sysn)
        ref = cols.get("CPU (all seeds)")
        if not ref:
            continue
        sigma_band(a)
        for lab, c, dx in (("GPU double, all runs", S1, -0.15), ("GPU float, all runs", S2, 0.15)):
            g = cols.get(lab)
            if not g:
                continue
            z = [(g[k][0] - ref[k][0]) / math.hypot(g[k][1], ref[k][1]) for k in keys]
            a.plot([i + dx for i in range(len(keys))], z, "o", color=c, mec=SURF, mew=1.5,
                   label=lab.replace(", all runs", ""), zorder=3)
        a.set_xticks(range(len(keys)))
        a.set_xticklabels([OBS.get(k) or (r"$n_{\geq12}$" if k == "n12+" else f"$n_{{{k[1:]}}}$")
                           for k in keys], fontsize=9)
        a.set_title(f"system {sysn}" + (": liquid" if sysn == "A" else ": degenerate gas, long exchange cycles"), loc="left", fontsize=10)
        a.grid(axis="x", visible=False)
    ax[0].set_ylim(-4, 4)
    ax[0].set_ylabel(r"(GPU $-$ CPU) / $\sigma$")
    ax[1].legend(loc="lower right", ncol=2)
    fig.suptitle(r"GPU (W=32) versus CPU, 3D Lennard-Jones, $N\approx100$ (shaded: $\pm2\sigma$)",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/val_lj100.png", dpi=150)


def sq():
    """S(q) of the 1D contact gas at n = 1 (validation/sq_combine.py writes validation/out/sq/<case>.dat)"""
    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    qq = np.linspace(1e-3, 2.0, 300)
    for case, b, col in (("t40b1", 1, S2), ("t40b3", 3, S3), ("t40b10", 10, S1)):
        path = f"validation/out/sq/{case}.dat"
        if not os.path.exists(path):
            continue
        q, S, e, ex, SL = np.loadtxt(path, unpack=True)
        k = q < 2.05
        ax[0].plot(q[k], ex[k], color=col, lw=1.2)
        ax[0].errorbar(q[k], S[k], e[k], fmt="o", color=col, label=rf"$\beta={b}$", **DOT)
        v = math.pi
        ax[0].plot(qq, qq / (2 * math.pi) / np.tanh(b * v * qq / 2), color=col, lw=1, ls=":")
    ax[0].plot(qq, qq / (2 * math.pi), color=INK2, lw=1, ls="--", label=r"$T=0$: $q/2\pi n$")
    ax[0].set_title(r"Tonks gas, $L=40$ (lines: exact; dotted: harmonic Luttinger)", loc="left", fontsize=10)
    for case, g, col in (("ll2.5", 2.5, S2), ("ll8", 8, S1), ("t40b10", None, S3)):
        path = f"validation/out/sq/{case}.dat"
        if not os.path.exists(path):
            continue
        q, S, e, ex, SL = np.loadtxt(path, unpack=True)
        k = q < 2.05
        ax[1].errorbar(q[k], S[k], e[k], fmt="o", color=col, label=r"$\gamma=\infty$" if g is None else rf"$\gamma={g:g}$", **DOT)
        ax[1].plot(q[k], SL[k], color=col, lw=1, ls=":")
    ax[1].set_title(r"Lieb-Liniger, $n=1$, $\beta=10$ (dotted: harmonic Luttinger, Bethe-ansatz $K$)",
                    loc="left", fontsize=10)
    for a in ax:
        a.set_xlabel(r"$q$"); a.set_ylabel(r"$S(q)$"); a.set_xlim(0, 2.0); a.set_ylim(0, None); a.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/val_sq.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for w in sys.argv[1:] or ["trotter", "cycles", "lj100", "sq"]:
        globals()[w]()
