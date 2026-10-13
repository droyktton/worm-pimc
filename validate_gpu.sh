#!/bin/bash
# usage: sbatch -p gpu --gres=gpu:1 validate_gpu.sh [R] [steps] [therm] [suffix]
# compares the GPU sampler with exact results: trap cases (exact_pa.py, N/Vbar/E)
# and the ideal Bose gas in a periodic box (exact_ideal.py, N/E/winding).
# suffix "_f" uses the single-precision binaries.
R=${1:-16384}; ST=${2:-2e5}; TH=${3:-5e4}; SFX=${4:-}
run() { # dim P beta mu omega kappa C
  d=$1; P=$2; b=$3; mu=$4; w=$5; k=$6; C=$7
  ex=$(python3 exact_pa.py --d $d --beta $b --mu $mu --omega $w --kappa $k --P $P --Nmax 30 2>/dev/null | head -1)
  mc=$(./worm_pimc_gpu_${d}d$SFX P=$P beta=$b mu=$mu omega=$w kappa=$k C=$C Lmax=8 Lbox=12 R=$R steps=$ST therm=$TH seed=7 | grep RESULT)
  echo "d=$d P=$P beta=$b mu=$mu w=$w kappa=$k C=$C"; echo "  exact: $ex"; echo "  GPU  : $mc"
}
ideal() { # dim beta mu L C
  ex=$(python3 exact_ideal.py --d $1 --beta $2 --mu $3 --L $4)
  mc=$(./worm_pimc_gpu_$1d$SFX pbc=1 omega=0 P=24 beta=$2 mu=$3 Lbox=$4 C=$5 Lmax=8 R=$R steps=$ST therm=$TH seed=7 | grep RESULT)
  echo "IDEAL d=$1 beta=$2 mu=$3 L=$4 C=$5"; echo "  exact: $ex"; echo "  GPU  : $mc"
}
nvidia-smi --query-gpu=name --format=csv,noheader
run 1 24 1 0 1 0 0.001
run 1 24 1 0 1 0.3 0.001
run 1 16 2 -0.5 1 0 0.001
run 2 24 1 0 1 0 0.001
run 2 24 1 -0.3 1 0.3 0.001
ideal 1 1 -0.2 4 1e-3
ideal 2 1 -0.3 4 1e-3
ideal 3 1 -0.5 4 1e-3
ideal 3 2 -0.1 3 1e-4
