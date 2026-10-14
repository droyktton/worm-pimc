#!/bin/bash
# Trotter (time-step) convergence and exchange-cycle test in a harmonic trap with harmonic
# pair coupling (omega=1, kappa=0.3, beta=2): GPU W=1 double at P = 4 ... 64, against
#   - the exact primitive-approximation result at the same P (exact_pa.py, strict test),
#   - the exact continuum result (P -> infinity), approached as tau^2.
# usage: sbatch -p gpu --gres=gpu:1 validation/trotter.sh <d> <mu> [C]
# C is the G-sector weight at P=16; it is scaled as (16/P)^2 to keep Zfrac roughly constant
# (with a fixed C, Zfrac fell to ~0.007 at P=64 and the replicas were under-thermalised).
# output: validation/out/trotter_d<d>.csv   (one line per P and observable)
d=$1; mu=$2; C=${3:-1e-3}
ARCH=${ARCH:-"-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"}
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out
nvcc -O3 -std=c++17 $ARCH -DDIM=$d worm_pimc_gpu.cu -o $B/g || exit 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
OUT=validation/out/trotter_d$d${TAG}.csv
echo "d,P,tau,obs,exact_PA,exact_cont,mc,err,Zfrac,overflow" > $OUT
for P in 4 6 8 12 16 24 32 48 64; do
  Lmax=$(( P - 2 < 8 ? P - 2 : 8 ))
  CP=$(python3 -c "print('%.4g' % ($C * (16 / $P) ** 2))")
  mc=$($B/g W=1 P=$P beta=2 mu=$mu omega=1 kappa=0.3 C=$CP Lmax=$Lmax Lbox=8 SM=32 R=${R:-32768} steps=${STEPS:-4e5} therm=${THERM:-2e5} seed=$((200 + P + ${SEEDOFF:-0})) 2>&1)
  echo "$mc" | grep -E "^(RESULT|CYCLES|accept)"
  python3 - "$d" "$P" "$mu" "$mc" >> $OUT <<'PY'
import re, sys
import exact_pa as ep
d, P, mu, mc = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
beta, J, Nmax = 2.0, 6, 22
g = lambda pat: [float(x) for x in re.search(pat, mc).groups()]
val = {k: g(k + r"= +([-0-9.]+) \+- ([0-9.]+)") for k in ("N", "Vbar", "E")}
cyc = re.search(r"CYCLES(.*)", mc).group(1)
nj = [float(x) for x in re.findall(r"n\d+\+?= ([-0-9.]+)", cyc)]
ej = [float(x) for x in re.findall(r"\+- ([0-9.]+)", cyc)]
# merge the program's bins j >= J into the last one (errors combined conservatively)
val.update({f"n{j + 1}": [nj[j], ej[j]] for j in range(J - 1)})
val[f"n{J}+"] = [sum(nj[J - 1:]), sum(ej[J - 1:])]
(zf,) = g(r"Zfrac= ([0-9.]+)"); (ov,) = g(r"overflow=([0-9]+)")
ex = {}
for lab, PP in (("pa", P), ("ct", None)):
    N, V, tl, E = ep.observables(beta, mu, d, 1.0, 0.3, PP, Nmax)
    n = ep.cycles(beta, mu, d, 1.0, 0.3, PP, Nmax, J)
    ex[lab] = dict(N=N, Vbar=V, E=E, **{f"n{j + 1}" + ("+" if j == J - 1 else ""): x for j, x in enumerate(n)})
for k, (m, e) in val.items():
    print(f"{d},{P},{beta / P:.6g},{k},{ex['pa'][k]:.7g},{ex['ct'][k]:.7g},{m},{e},{zf},{int(ov)}")
PY
done
column -s, -t $OUT
rm -rf $B
