#!/bin/bash
# Luttinger parameter of the clean Lieb-Liniger gas (pot=3) against the Bethe ansatz K(gamma) (exact_ll_gs.py):
# n = 1 (lam = 1/2), L = 20, mu = ground-state chemical potential, beta chosen so that beta v / L ~ 1
# (both <W^2> and <dN^2> are then of order K/pi). One CPU chain per array task (seed = task id).
# usage: sbatch -p cpu -c1 --array=1-8 validation/llK.sh <gamma: 1|2.5|8> <tau: 0.05|0.025>
# output: validation/out/llK/g<gamma>_t<tau>_s<seed>.txt
gm=$1; tau=$2; s=${SLURM_ARRAY_TASK_ID:-1}
case $gm in
  1)   g=1;   mu=0.71397; beta=21.8;;
  2.5) g=2.5; mu=1.44231; beta=14.6;;
  8)   g=8;   mu=2.83725; beta=9.5;;
esac
P=$(python3 -c "print(round($beta / $tau))")
unset CXX CC CXXFLAGS
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out/llK
g++ -O3 -march=native -std=c++17 -DDIM=1 worm_pimc.cpp -o $B/c || exit 1
C=${C:-$(python3 -c "print('%.3g' % (5e-8 * (292 / $P) ** 2))")}
STEPS=${STEPS:-$(python3 -c "print('%.3g' % (5e4 * $P))")}
o=validation/out/llK/g${gm}_t${tau}${TAG}_s$s.txt
echo "host=$(hostname)" > $o
$B/c pbc=1 omega=0 pot=3 g=$g Lbox=20 beta=$beta mu=$mu P=$P Lmax=$((P / 10)) C=$C N0=20 \
  steps=$STEPS therm=$(python3 -c "print('%.3g' % ($STEPS / 10))") seed=$((1000 * s + 13)) >> $o 2>&1
rm -rf $B
