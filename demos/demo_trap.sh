#!/bin/bash
# Demo 1 (CPU): bosons in a harmonic trap with harmonic pair coupling, against the
# exact primitive-approximation result (exact_pa.py).  Four independent seeds per case
# run in parallel and are combined (mean and standard error over seeds).
# usage (from the repository root): bash demos/demo_trap.sh   or, on the cluster, sbatch slurm/demo_trap.sbatch
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B demos/out
for d in 1 2; do g++ -O3 -march=native -std=c++17 -DDIM=$d worm_pimc.cpp -o $B/c${d}d || exit 1; done
run() { # d beta mu kappa
  ex=$(python3 exact_pa.py --d $1 --beta $2 --mu $3 --omega 1 --kappa $4 --P 24 --Nmax 30 | head -1)
  for s in 1 2 3 4; do
    $B/c$1d P=24 beta=$2 mu=$3 omega=1 kappa=$4 C=1e-3 Lmax=8 Lbox=12 steps=1e8 therm=1e6 seed=$s | grep -E "^RESULT" | tr '\n' ' ' > $B/r_$1_$s &
  done
  wait
  python3 - "$*" "$ex" $B/r_$1_* <<'PY'
import re, sys, statistics as st, math
case, ex, files = sys.argv[1], sys.argv[2], sys.argv[3:]
runs = [open(f).read() for f in files]
x = dict(zip(("N", "Vbar", "E"), map(float, re.search(r"<N> = ([-0-9.]+) +<Vbar> = ([-0-9.]+).*<E> = ([-0-9.]+)", ex).groups())))
d, beta, mu, kappa = case.split()
print(f"d={d} beta={beta} mu={mu} kappa={kappa}   ({len(runs)} seeds x 1e8 steps)")
for k in ("N", "Vbar", "E"):
    v = [float(re.search(k + r"= +(-?[0-9.]+)", r).group(1)) for r in runs]
    m, e = st.mean(v), st.stdev(v) / math.sqrt(len(v))
    print(f"  {k:5s} exact = {x[k]:.5f}   MC = {m:.5f} +- {e:.5f}   ({(m - x[k]) / e:+.1f} sigma)")
PY
}
{ run 1 1 0 0.3; run 2 1 -0.3 0.3; } | tee demos/out/demo_trap.txt
rm -rf $B
