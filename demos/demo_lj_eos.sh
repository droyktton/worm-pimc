#!/bin/bash
# Demo 3 (GPU, W=32): quantum Lennard-Jones fluid (lam = hbar^2/(2 m sigma^2 eps) = 0.05,
# T = 1, P = 32) in a periodic box L = 4.5 sigma; grand-canonical equation of state:
# density <N>/V and energy per particle versus chemical potential, across the gas-liquid
# crossover.  Every point is run from two starts (N0 = 8 and N0 = 64 lattices): agreement
# within errors is the check that the replicas reached equilibrium (the script stops otherwise).
# usage (from the repository root): bash demos/demo_lj_eos.sh   or, on the cluster, sbatch slurm/demo_lj_eos.sbatch
# smaller GPUs: R=128 bash demos/demo_lj_eos.sh
ARCH=${ARCH:-"-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"}   # laptop/desktop: ARCH=-arch=native
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B demos/out
nvcc -O3 -std=c++17 $ARCH -DDIM=3 -DUSE_FLOAT worm_pimc_gpu.cu -o $B/g3f || exit 1
OUT=demos/out/lj_eos.csv
echo "mu,N_from8,N_from8_err,N_from64,N_from64_err,diff_sigma,rho,rho_err,E_per_N,E_per_N_err,Zfrac" > $OUT
L=4.5
for mu in -8 -6 -4 -3 -2 -1 0 1 2; do
  A="W=32 pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=$mu Lbox=$L rc=2.25 SM=160 C=1e-4 Lmax=8 R=${R:-256} steps=1e6 therm=4e6 seed=21"
  lo=$($B/g3f $A N0=8 2>&1); hi=$($B/g3f $A N0=64 2>&1)
  python3 - "$mu" "$L" "$lo" "$hi" >> $OUT <<'PY'
import re, sys, math
mu, L, runs = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3:]
def get(mc):
    g = lambda pat: [float(x) for x in re.search(pat, mc).groups()]
    (ov,) = g(r"overflow=([0-9]+)")
    if ov > 0:
        sys.exit(f"mu={mu}: {int(ov)} capacity overflows - increase SM")
    return g(r"N= ([-0-9.]+) \+- ([0-9.]+)") + g(r"E= ([-0-9.]+) \+- ([0-9.]+)") + g(r"Zfrac= ([0-9.]+)")
(N1, e1, E1, f1, Z1), (N2, e2, E2, f2, Z2) = map(get, runs)
diff = (N1 - N2) / math.hypot(e1, e2)
if abs(diff) > 3:
    sys.exit(f"mu={mu}: starts N0=8 and N0=64 disagree ({N1:.3f} vs {N2:.3f}, {diff:+.1f} sigma) - increase therm")
V = L ** 3
N, eN = 0.5 * (N1 + N2), 0.5 * math.hypot(e1, e2)
E, eE = 0.5 * (E1 + E2), 0.5 * math.hypot(f1, f2)
# E/N as a ratio of averages; error propagated from E and N
eEN = abs(E / N) * math.hypot(eE / E if E else 0.0, eN / N)
print(f"{mu},{N1},{e1},{N2},{e2},{diff:.2f},{N / V:.6g},{eN / V:.3g},{E / N:.6g},{eEN:.3g},{0.5 * (Z1 + Z2):.4f}")
PY
done
column -s, -t $OUT
rm -rf $B
