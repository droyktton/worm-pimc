#!/bin/bash
# 1D contact interaction (pot=3, exact two-body pair-product action) against exact results:
#   tonks: hard core (g=inf) on a ring L=10, beta=2, mu=1           -> exact_tonks.py (finite L, incl. P(W))
#   g2:    g=2 (gamma ~ 2.5), L=20, beta=1, mu=1                     -> exact_yy.py (Yang-Yang, L -> inf)
#   tonksd<Vd>: hard core in the disorder potential Vd = 0.5|1|2, qc=4, dseed=1 (6 modes), same ring
#                                                                     -> exact_tonks.py with the same realisation
#   g05:   g=0.5 (gamma ~ 0.24), L=20, beta=1, mu=1                  -> exact_yy.py
# one CPU chain per array task (seed = task id).
# usage: sbatch -p cpu -c1 --array=1-8 validation/contact.sh <tonks|tonksd0.5|tonksd1|tonksd2|g2|g05> <P>
# output: validation/out/contact/<case>_P<P>_s<seed>.txt
case=$1; P=$2; s=${SLURM_ARRAY_TASK_ID:-1}
unset CXX CC CXXFLAGS
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out/contact
g++ -O3 -march=native -std=c++17 -DDIM=1 worm_pimc.cpp -o $B/c || exit 1
case $case in
  tonks) A="g=-1 Lbox=10 beta=2 mu=1 N0=4 C=$(python3 -c "print('%.3g' % (3e-4 * (20 / $P) ** 2))")";;
  tonksd*) A="g=-1 Lbox=10 beta=2 mu=1 N0=4 Vd=${case#tonksd} qc=4 dseed=1 C=$(python3 -c "print('%.3g' % (3e-4 * (20 / $P) ** 2))")";;
  g2)    A="g=2 Lbox=20 beta=1 mu=1 N0=16 C=$(python3 -c "print('%.3g' % (1e-4 * (20 / $P) ** 2))")";;
  g05)   A="g=0.5 Lbox=20 beta=1 mu=1 N0=42 C=$(python3 -c "print('%.3g' % (3e-5 * (20 / $P) ** 2))")";;
esac
o=validation/out/contact/${case}_P${P}_s$s.txt
echo "host=$(hostname)" > $o
$B/c pbc=1 omega=0 pot=3 $A P=$P Lmax=$((P / 4)) steps=${STEPS:-4e7} therm=${THERM:-4e6} seed=$((1000 * s + 11)) >> $o 2>&1
rm -rf $B
