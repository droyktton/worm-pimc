#!/bin/bash
# Exchange-cycle distribution of the ideal Bose gas in a periodic box against the exact
# cycle expansion (exact_ideal.py: <n_j> = e^{beta mu j} z(j)^d), GPU W=1 double and float.
# usage: sbatch -p gpu --gres=gpu:1 validation/cycles_ideal.sh
# output: validation/out/cycles_ideal.csv
ARCH=${ARCH:-"-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"}
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out
for d in 1 2 3; do
  nvcc -O3 -std=c++17 $ARCH -DDIM=$d worm_pimc_gpu.cu -o $B/g${d}d || exit 1
  nvcc -O3 -std=c++17 $ARCH -DDIM=$d -DUSE_FLOAT worm_pimc_gpu.cu -o $B/g${d}d_f || exit 1
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
OUT=validation/out/cycles_ideal.csv
echo "case,prec,obs,exact,mc,err,Zfrac,overflow" > $OUT
run() { # d beta mu L C N0 SM
  for prec in double float; do
    x=$B/g$1d; [ $prec = float ] && x=${x}_f
    mc=$($x W=1 pbc=1 omega=0 P=24 beta=$2 mu=$3 Lbox=$4 C=$5 Lmax=8 N0=$6 SM=$7 R=${R:-16384} steps=${STEPS:-4e5} therm=${THERM:-2e5} seed=31 2>&1)
    echo "$mc" | grep -E "^(RESULT|CYCLES)"
    python3 - "$1 $2 $3 $4" "$prec" "$mc" >> $OUT <<'PY'
import re, sys
import exact_ideal as ei
case, prec, mc = sys.argv[1], sys.argv[2], sys.argv[3]
d, beta, mu, L = case.split(); d = int(d); beta, mu, L = map(float, (beta, mu, L))
g = lambda pat: [float(x) for x in re.search(pat, mc).groups()]
N, E, D2, rs = ei.observables(beta, mu, d, 0.5, L)
ex = dict(N=N, E=E, rhos=rs)
val = {k: g(k + r"= +([-0-9.]+) \+- ([0-9.]+)") for k in ex}
cyc = re.search(r"CYCLES(.*)", mc).group(1)
nj = [float(x) for x in re.findall(r"n\d+\+?= ([-0-9.]+)", cyc)]
ej = [float(x) for x in re.findall(r"\+- ([0-9.]+)", cyc)]
for j, x in enumerate(ei.cycles(beta, mu, d, 0.5, L, len(nj))):
    k = f"n{j + 1}" + ("+" if j == len(nj) - 1 else "")
    ex[k] = x; val[k] = [nj[j], ej[j]]
(zf,) = g(r"Zfrac= ([0-9.]+)"); (ov,) = g(r"overflow=([0-9]+)")
for k in ex:
    print(f"d={d} beta={beta:g} mu={mu:g} L={L:g},{prec},{k},{ex[k]:.7g},{val[k][0]},{val[k][1]},{zf},{int(ov)}")
PY
  done
}
run 1 1 -0.2 4 1e-3 5 128
run 2 1 -0.3 4 1e-3 4 128
run 3 2 -0.1 3 1e-4 5 128
run 3 1.25 -0.1220 4 1e-5 8 256
column -s, -t $OUT
rm -rf $B
