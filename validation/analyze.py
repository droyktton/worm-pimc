#!/usr/bin/env python3
"""Summaries of the validation runs in validation/out/ (markdown tables on stdout).
usage (from the repository root, on a compute node): python3 validation/analyze.py [trotter|cycles|lj100]"""
import csv, glob, math, os, re, sys
import numpy as np
from scipy.stats import chi2 as chi2dist

OUT = "validation/out"


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pchi2(x2, dof):
    return chi2dist.sf(x2, dof)


def fit_tau2(tau, y, e, order=2):
    """weighted least squares y = a + b tau^2 (+ c tau^4); returns a, err(a), chi2, dof"""
    t2 = np.asarray(tau) ** 2
    A = np.vstack([t2 ** k for k in range(order)]).T
    w = 1.0 / np.asarray(e)
    cov = np.linalg.inv((A * w[:, None] ** 2).T @ A)
    c = cov @ (A * w[:, None] ** 2).T @ np.asarray(y)
    r = (np.asarray(y) - A @ c) * w
    return c[0], math.sqrt(cov[0, 0]), float(r @ r), len(y) - order


def trotter():
    for path in sorted(glob.glob(f"{OUT}/trotter_d*.csv")):
        R = rows(path)
        d = R[0]["d"]
        print(f"\n### Trotter scan, d={d}  ({path})\n")
        Ps = sorted({int(r["P"]) for r in R})
        print("| P | τ | N (exact PA) | N (MC) | V̄ (exact PA) | V̄ (MC) | E (exact PA) | E (MC) |")
        print("|---|---|---|---|---|---|---|---|")
        for P in Ps:
            v = {r["obs"]: r for r in R if int(r["P"]) == P}
            cell = lambda k: (f"{float(v[k]['exact_PA']):.5f}", f"{float(v[k]['mc']):.5f}({round(float(v[k]['err']) * 1e5)})")
            print(f"| {P} | {float(v['N']['tau']):.4g} | " + " | ".join(" | ".join(cell(k)) for k in ("N", "Vbar", "E")) + " |")
        # MC versus PA at every P
        print()
        tot, dof, worst = 0.0, 0, (0, "")
        for r in R:
            if float(r["err"]) == 0:
                continue
            z = (float(r["mc"]) - float(r["exact_PA"])) / float(r["err"])
            tot += z * z; dof += 1
            if abs(z) > abs(worst[0]):
                worst = (z, f"{r['obs']} at P={r['P']}")
        zN = [(float(r["mc"]) - float(r["exact_PA"])) / float(r["err"]) for r in R if r["obs"] == "N"]
        x2N = sum(z * z for z in zN)
        print(f"- MC vs exact PA, N at every P (independent runs): chi2/dof = {x2N:.1f}/{len(zN)}, p = {pchi2(x2N, len(zN)):.2f}")
        print(f"- MC vs exact PA, all P and observables (incl. cycles; strongly correlated within a P, so this "
              f"chi2 is inflated): {tot:.1f}/{dof}, largest deviation {worst[0]:+.1f}σ ({worst[1]})")
        for k in ("N", "Vbar", "E"):
            v = sorted((r for r in R if r["obs"] == k), key=lambda r: int(r["P"]))
            tau = [float(r["tau"]) for r in v]
            y = [float(r["mc"]) for r in v]
            e = [float(r["err"]) for r in v]
            ct = float(v[0]["exact_cont"])
            for order, sel in ((2, [i for i, t in enumerate(tau) if t <= 0.17]), (3, list(range(len(tau))))):
                if len(sel) <= order:
                    continue
                a, ea, x2, nd = fit_tau2([tau[i] for i in sel], [y[i] for i in sel], [e[i] for i in sel], order)
                lab = "a + b τ², τ ≤ 1/6" if order == 2 else "a + b τ² + c τ⁴, all P"
                print(f"- {k}: fit {lab}: τ→0 = {a:.5f}({round(ea * 1e5)}), continuum exact = {ct:.5f} "
                      f"({(a - ct) / ea:+.1f}σ), fit chi2/dof = {x2:.1f}/{nd}")
            pa4 = float(v[0]["exact_PA"])
            print(f"  (Trotter error of the exact PA at P={v[0]['P']}: {pa4 - ct:+.5f}; at P={v[-1]['P']}: {float(v[-1]['exact_PA']) - ct:+.6f})")


def cycles():
    path = f"{OUT}/cycles_ideal.csv"
    if os.path.exists(path):
        R = rows(path)
        print(f"\n### Ideal Bose gas, exchange cycles  ({path})\n")
        for case in dict.fromkeys(r["case"] for r in R):
            for prec in ("double", "float"):
                v = [r for r in R if r["case"] == case and r["prec"] == prec]
                if not v:
                    continue
                z = [(float(r["mc"]) - float(r["exact"])) / float(r["err"]) for r in v if float(r["err"]) > 0]
                x2 = sum(t * t for t in z)
                print(f"- {case}, {prec}: chi2/dof = {x2:.1f}/{len(z)} (p = {pchi2(x2, len(z)):.2f}), "
                      f"max |dev| = {max(map(abs, z)):.1f}σ, Zfrac = {v[0]['Zfrac']}, overflow = {v[0]['overflow']}")
        print("\n| case | j | exact ⟨n_j⟩ | GPU double | GPU float |")
        print("|---|---|---|---|---|")
        for case in dict.fromkeys(r["case"] for r in R):
            for k in dict.fromkeys(r["obs"] for r in R if r["case"] == case):
                v = {r["prec"]: r for r in R if r["case"] == case and r["obs"] == k}
                f = lambda p: f"{float(v[p]['mc']):.5f}({round(float(v[p]['err']) * 1e5)})" if p in v else "—"
                print(f"| {case} | {k} | {float(v['double']['exact']):.5f} | {f('double')} | {f('float')} |")


def parse_run(path):
    s = open(path).read()
    out = {}
    for k in ("N", "Vbar", "E", "D2", "rhos"):
        m = re.search(rf"\b{k}= +([-0-9.]+) \+- ([0-9.]+)", s)
        if m:
            out[k] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"CYCLES(.*)", s)
    if m:
        for k, v, e in re.findall(r"(n\d+\+?)= ([-0-9.]+) \+- ([0-9.]+)", m.group(1)):
            out[k] = (float(v), float(e))
    for k in ("Zfrac", "overflow", "N0"):
        m = re.search(rf"{k}=\s*([0-9.]+)", s)
        if m:
            out[k] = float(m.group(1))
    return out


def lj100_columns(sysn):
    """{label: {obs: (mean, err)}} for the CPU seed groups and every GPU run of system A or B"""
    seeds = sorted(glob.glob(f"{OUT}/lj100_{sysn}_cpu/seed*.txt"))
    gpus = sorted(glob.glob(f"{OUT}/lj100_{sysn}_gpu_*.txt"))
    cols = {}
    if seeds:
        runs = [parse_run(p) for p in seeds]
        runs = [r for r in runs if "N" in r]
        # seeds with odd index started from N0=100, even from 64 (see lj100_cpu.sh)
        idx = [int(re.search(r"seed(\d+)", p).group(1)) for p in seeds]
        groups = {"CPU (all seeds)": runs,
                  "CPU, N0=100": [r for r, i in zip(runs, idx) if i % 2],
                  "CPU, N0=64": [r for r, i in zip(runs, idx) if not i % 2]}
        for lab, g in groups.items():
            if len(g) < 2:
                continue
            cols[lab] = {k: (np.mean([r[k][0] for r in g]), np.std([r[k][0] for r in g], ddof=1) / math.sqrt(len(g)))
                         for k in g[0] if isinstance(g[0][k], tuple)}
            cols[lab]["_n"] = len(g)
    for p in gpus:
        m = re.search(r"gpu_([a-z]+)_N0(\d+)(.*)\.txt", p)
        r = parse_run(p)
        if "N" in r:
            cols[f"GPU {m.group(1)}, N0={m.group(2)}" + (f" ({m.group(3).strip('_')})" if m.group(3) else "")] = r
    for prec in ("double", "float"):        # inverse-variance mean over all runs of one precision
        g = [c for lab, c in cols.items() if lab.startswith(f"GPU {prec}")]
        if len(g) > 1:
            cols[f"GPU {prec}, all runs"] = {k: (sum(c[k][0] / c[k][1] ** 2 for c in g) / sum(c[k][1] ** -2 for c in g),
                                                 sum(c[k][1] ** -2 for c in g) ** -0.5)
                                             for k in g[0] if isinstance(g[0][k], tuple) and all(k in c and c[k][1] > 0 for c in g)}
    return cols


def lj100():
    for sysn in ("A", "B"):
        cols = lj100_columns(sysn)
        if not cols:
            continue
        print(f"\n### System {sysn}: GPU (W=32) versus CPU\n")
        keys = ["N", "Vbar", "E", "rhos"] + [f"n{j}" for j in range(1, 12)] + ["n12+"]
        keys = [k for k in keys if any(k in c for c in cols.values())]
        print("| | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for k in keys:
            cells = []
            for c in cols.values():
                if k in c:
                    v, e = c[k]
                    nd = max(0, -int(math.floor(math.log10(e))) + 1) if e > 0 else 5
                    cells.append(f"{v:.{nd}f}({round(e * 10 ** nd)})")
                else:
                    cells.append("—")
            print(f"| {k} | " + " | ".join(cells) + " |")
        ref = cols.get("CPU (all seeds)")
        if ref:
            print()
            for lab, c in cols.items():
                if lab.startswith("CPU"):
                    continue
                line = f"- {lab} vs CPU:"
                for grp, ks in (("N, V̄, E, ρs", ["N", "Vbar", "E", "rhos"]), ("cycles", [k for k in keys if k.startswith("n")])):
                    zs = [(k, (c[k][0] - ref[k][0]) / math.hypot(c[k][1], ref[k][1])) for k in ks
                          if k in c and k in ref and math.hypot(c[k][1], ref[k][1]) > 0]
                    if not zs:
                        continue
                    x2 = sum(z * z for _, z in zs)
                    w = max(zs, key=lambda t: abs(t[1]))
                    line += f" [{grp}] chi2/dof = {x2:.1f}/{len(zs)}, largest {w[1]:+.1f}σ ({w[0]});"
                print(line + f" Zfrac = {c.get('Zfrac')}, overflow = {c.get('overflow')}")
            print(f"- CPU: {ref['_n']} seeds")


if __name__ == "__main__":
    what = sys.argv[1:] or ["trotter", "cycles", "lj100"]
    for w in what:
        globals()[w]()
