#!/bin/bash
# Demo 2 (GPU, W=1): ideal Bose gas in a periodic box (3D, L=4, lam=1/2) at fixed mean
# particle number <N> = 8 (mu(T) from the exact solution): <N>, E and superfluid fraction
# rho_s/rho (winding estimator) versus temperature, across the finite-size BEC crossover
# (bulk T_c = 4 pi lam (rho/zeta(3/2))^(2/3) = 0.83), against exact_ideal.py.
# usage (from the repository root): bash demos/demo_superfluid.sh   or, on the cluster, sbatch slurm/demo_superfluid.sbatch
# smaller GPUs: R=2048 bash demos/demo_superfluid.sh
ARCH=${ARCH:-"-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"}   # laptop/desktop: ARCH=-arch=native
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B demos/out
nvcc -O3 -std=c++17 $ARCH -DDIM=3 -DUSE_FLOAT worm_pimc_gpu.cu -o $B/g3f || exit 1
OUT=demos/out/superfluid.csv
echo "beta,T,mu,N_exact,N_mc,N_err,E_exact,E_mc,E_err,rhos_exact,rhos_mc,rhos_err,Zfrac,overflow" > $OUT
for beta in 0.3 0.5 0.75 1 1.25 1.5 2 3; do
  # chemical potential giving <N> = 8 exactly (bisection on the exact solution)
  mu=$(python3 - "$beta" <<'PY'
import sys; from exact_ideal import observables
b = float(sys.argv[1]); lo, hi = -20.0, -1e-9
for _ in range(80):
    m = 0.5 * (lo + hi)
    lo, hi = (m, hi) if observables(b, m, 3, 0.5, 4.0)[0] < 8.0 else (lo, m)
print(f"{0.5 * (lo + hi):.8f}")
PY
)
  ex=$(python3 exact_ideal.py --d 3 --beta $beta --mu $mu --L 4)
  # start from N0=8 straight worldlines; long thermalisation: at low T the condensate
  # occupation fluctuates strongly and slowly, and short runs bias <N> low
  mc=$($B/g3f W=1 pbc=1 omega=0 P=32 beta=$beta mu=$mu Lbox=4 C=1e-5 Lmax=8 SM=256 N0=8 R=${R:-8192} steps=2e5 therm=5e5 seed=11 2>&1)
  python3 - "$beta" "$mu" "$ex" "$mc" >> $OUT <<'PY'
import re, sys
beta, mu = float(sys.argv[1]), float(sys.argv[2]); ex, mc = sys.argv[3], sys.argv[4]
g = lambda pat, s: [float(x) for x in re.search(pat, s).groups()]
Nx, Ex, Dx, rx = g(r"<N> = ([-0-9.]+) +<E> = ([-0-9.]+) +<D2> = ([-0-9.]+) +rhos = ([-0-9.]+)", ex)
N, eN = g(r"N= ([-0-9.]+) \+- ([0-9.]+)", mc)
E, eE = g(r"E= ([-0-9.]+) \+- ([0-9.]+)", mc)
r, er = g(r"rhos= ([-0-9.]+) \+- ([0-9.]+)", mc)
(ov,) = g(r"overflow=([0-9]+)", mc)
(zf,) = g(r"Zfrac= ([0-9.]+)", mc)
if ov > 0:
    sys.exit(f"beta={beta}: {int(ov)} capacity overflows - increase SM")
print(f"{beta},{1/beta:.6g},{mu},{Nx},{N},{eN},{Ex},{E},{eE},{rx},{r},{er},{zf},{int(ov)}")
PY
done
column -s, -t $OUT
rm -rf $B
