#!/bin/bash
# usage: ./validate.sh  -> compares sampler with exact PA results
run() { # dim P beta mu omega kappa C
  d=$1; P=$2; b=$3; mu=$4; w=$5; k=$6; C=$7
  ex=$(python3 exact_pa.py --d $d --beta $b --mu $mu --omega $w --kappa $k --P $P --Nmax 30 2>/dev/null | head -1)
  mc=$(./worm_pimc_${d}d P=$P beta=$b mu=$mu omega=$w kappa=$k C=$C Lmax=8 Lbox=12 steps=3e7 therm=3e6 blocks=30 seed=7 | grep RESULT)
  echo "d=$d P=$P beta=$b mu=$mu w=$w kappa=$k C=$C"; echo "  exact: $ex"; echo "  MC   : $mc"
}
run 1 24 1 0 1 0 0.001
run 1 24 1 0 1 0.3 0.001
run 1 16 2 -0.5 1 0 0.001
run 2 24 1 0 1 0 0.001
run 2 24 1 -0.3 1 0.3 0.001
