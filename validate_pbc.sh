#!/bin/bash
# usage: sbatch -p cpu -c 12 validate_pbc.sh
# (a) regression of the trap cases incl. thermodynamic energy vs exact_pa.py
# (b) ideal Bose gas in a periodic box (N, E, winding/rho_s) vs exact_ideal.py
B=build_$SLURM_JOB_ID; mkdir -p $B
for d in 1 2 3; do g++ -O3 -march=native -std=c++17 -DDIM=$d worm_pimc.cpp -o $B/c${d}d || exit 1; done
trap_case() { # d P beta mu kappa seed
  ex=$(python3 exact_pa.py --d $1 --P $2 --beta $3 --mu $4 --omega 1 --kappa $5 --Nmax 30 | head -1)
  mc=$($B/c$1d P=$2 beta=$3 mu=$4 omega=1 kappa=$5 C=1e-3 Lmax=8 Lbox=12 steps=3e8 therm=1e6 blocks=40 seed=$6 | grep RESULT)
  printf "TRAP d=%s P=%s beta=%s mu=%s kappa=%s seed=%s\n  exact: %s\n  MC   : %s\n" "$@" "$ex" "$mc"
}
ideal_case() { # d beta mu L
  ex=$(python3 exact_ideal.py --d $1 --beta $2 --mu $3 --L $4)
  mc=$($B/c$1d pbc=1 omega=0 P=24 beta=$2 mu=$3 Lbox=$4 C=$5 Lmax=8 steps=3e8 therm=1e6 blocks=40 seed=11 | grep RESULT)
  printf "IDEAL d=%s beta=%s mu=%s L=%s C=%s\n  exact: %s\n  MC   : %s\n" "$@" "$ex" "$mc"
}
$B/c3d pbc=1 omega=0 P=24 beta=2 mu=-0.1 Lbox=3 C=1e-3 steps=2e6 therm=0 debug=1 | grep -E "INCONS|RESULT" > $B/dbg &
trap_case 1 24 1 0 0 11 > $B/t1 &
trap_case 1 24 1 0 0 23 > $B/t1b &
trap_case 1 24 1 0 0.3 11 > $B/t2 &
trap_case 2 24 1 -0.3 0.3 11 > $B/t3 &
trap_case 2 24 1 -0.3 0.3 23 > $B/t3b &
ideal_case 1 1 -0.2 4 1e-3 > $B/i1 &
ideal_case 2 1 -0.3 4 1e-3 > $B/i2 &
ideal_case 3 1 -0.5 4 1e-3 > $B/i3 &
ideal_case 3 2 -0.1 3 1e-3 > $B/i4 &
ideal_case 3 2 -0.1 3 1e-4 > $B/i5 &
wait
echo "debug run: $(cat $B/dbg)"
cat $B/t1 $B/t1b $B/t2 $B/t3 $B/t3b $B/i1 $B/i2 $B/i3 $B/i4 $B/i5
rm -rf $B
