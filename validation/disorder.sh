#!/bin/bash
# Ideal Bose gas in a 1D periodic box with a disorder potential (Vd, qc, dseed) against the exact
# transfer-matrix solution of the same primitive discretisation (exact_disorder.py): N, Vbar, E,
# <D^2>, <dN^2>, <W^2>, K and the exchange cycles <n_j>; CPU, GPU W=1 double and float.
# usage: sbatch -p gpu --gres=gpu:1 -c2 validation/disorder.sh
# output: validation/out/disorder.csv
ARCH=${ARCH:-"-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"}
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out
g++ -O2 -std=c++17 -DDIM=1 worm_pimc.cpp -o $B/c1d || exit 1
nvcc -O3 -std=c++17 $ARCH -DDIM=1 worm_pimc_gpu.cu -o $B/g1d || exit 1
nvcc -O3 -std=c++17 $ARCH -DDIM=1 -DUSE_FLOAT worm_pimc_gpu.cu -o $B/g1d_f || exit 1
nvidia-smi --query-gpu=name --format=csv,noheader
OUT=validation/out/disorder.csv
echo "case,code,obs,exact,mc,err" > $OUT
run() { # L beta P Vd qc dseed dmu C
  ex="--L $1 --beta $2 --P $3 --Vd $4 --qc $5 --dseed $6"
  mu=$(python3 exact_disorder.py $ex --dmu $7 | awk '/^e0=/{print $4}')
  a="pbc=1 omega=0 Lbox=$1 beta=$2 P=$3 Vd=$4 qc=$5 dseed=$6 mu=$mu C=$8 Lmax=8"
  for code in cpu double float; do
    case $code in
      cpu)    mc=$($B/c1d $a steps=${CSTEPS:-4e7} therm=4e6 seed=17 2>&1) ;;
      double) mc=$($B/g1d $a W=1 SM=128 R=${R:-16384} steps=${STEPS:-4e5} therm=2e5 seed=31 2>&1) ;;
      float)  mc=$($B/g1d_f $a W=1 SM=128 R=${R:-16384} steps=${STEPS:-4e5} therm=2e5 seed=31 2>&1) ;;
    esac
    echo "$mc" | grep -E "^(RESULT|CYCLES|DISORDER)"
    python3 - "$ex --mu $mu" "L=$1 beta=$2 P=$3 Vd=$4 qc=$5 dseed=$6" $code "$mc" >> $OUT <<'PY'
import re, sys
import exact_disorder as ed
args, case, code, mc = sys.argv[1].split(), sys.argv[2], sys.argv[3], sys.argv[4]
kw = {args[i][2:]: float(args[i + 1]) for i in range(0, len(args), 2)}
box = ed.Box(kw["L"], 0.5, kw["Vd"], kw["qc"], int(kw["dseed"]))
ex = box.observables(kw["beta"], kw["mu"], int(kw["P"]))
val = {k: (v, e) for k, v, e in re.findall(r"([A-Za-z0-9]+\+?)= +(-?[0-9.]+(?:e[-+][0-9]+)?) \+- ([0-9.]+(?:e[-+][0-9]+)?)", mc)}
for k in ex:
    print(f"{case},{code},{k},{ex[k]:.7g},{val[k][0]},{val[k][1]}")
PY
  done
}
run 8 2 16 0.3 6 1 -0.3 1e-2
run 8 2 16 3.0 6 2 -0.3 1e-2
run 8 4 32 0.3 4 3 -0.1 1e-3
column -s, -t $OUT
rm -rf $B
