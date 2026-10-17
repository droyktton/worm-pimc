#!/bin/bash
# Static structure factor S(q) (nsq) of the 1D contact gas (pot=3) against exact results and Luttinger theory:
#   t10:       hard core, L=10, beta=2, mu=1, P=40                      -> exact_tonks.py (finite ring)
#   t10d1:     same in the disorder Vd=1, qc=4, dseed=1, P=80          -> exact_tonks.py (same realisation)
#   t40b<b>:   hard core, L=40, n ~ 1 (mu = pi^2/2), beta = 1|3|10, tau = 0.05 -> exact_tonks.py
#   ll<g>:     Lieb-Liniger g = 2.5|8, n = 1, L=40, beta=10, tau=0.025  -> (K q / 2 pi n) coth(beta v q / 2)
# L=40: the contact action costs O(P N^2) per measurement, hence meas=200 (about 1-3 h per chain).
# one CPU chain per array task (seed = task id).
# usage: sbatch -p cpu -c1 --array=1-8 validation/sq.sh <case>
# output: validation/out/sq/<case>_s<seed>.txt
case=$1; s=${SLURM_ARRAY_TASK_ID:-1}
unset CXX CC CXXFLAGS
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out/sq
g++ -O3 -march=native -std=c++17 -DDIM=1 worm_pimc.cpp -o $B/c || exit 1
case $case in
  t10)    A="g=-1 Lbox=10 beta=2 mu=1 N0=4 P=40 Lmax=10 C=7.5e-5 nsq=12";;
  t10d1)  A="g=-1 Lbox=10 beta=2 mu=1 N0=4 P=80 Lmax=20 C=1.9e-5 Vd=1 qc=4 dseed=1 nsq=12";;
  t40b1)  A="g=-1 Lbox=40 beta=1 mu=4.9348 N0=40 P=20 Lmax=5 C=${C:-5e-6} nsq=32 meas=200 steps=${STEPS:-2e7} therm=${THERM:-2e6}";;
  t40b3)  A="g=-1 Lbox=40 beta=3 mu=4.9348 N0=40 P=60 Lmax=15 C=${C:-1e-6} nsq=32 meas=200 steps=${STEPS:-2e7} therm=${THERM:-2e6}";;
  t40b10) A="g=-1 Lbox=40 beta=10 mu=4.9348 N0=40 P=200 Lmax=50 C=${C:-1e-7} nsq=32 meas=200 steps=${STEPS:-2e7} therm=${THERM:-2e6}";;
  ll2.5)  A="g=2.5 Lbox=40 beta=10 mu=1.44231 N0=40 P=400 Lmax=40 C=${C:-1e-8} nsq=32 meas=200 steps=${STEPS:-1e7} therm=${THERM:-2e6}";;
  ll8)    A="g=8 Lbox=40 beta=10 mu=2.83725 N0=40 P=400 Lmax=40 C=${C:-1e-8} nsq=32 meas=200 steps=${STEPS:-1e7} therm=${THERM:-2e6}";;
esac
o=validation/out/sq/${case}${TAG}_s$s.txt
echo "host=$(hostname)" > $o
$B/c pbc=1 omega=0 pot=3 steps=${STEPS:-4e7} therm=${THERM:-4e6} $A seed=$((1000 * s + 17)) >> $o 2>&1
rm -rf $B
