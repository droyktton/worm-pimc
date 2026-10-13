#!/bin/bash
# usage: sbatch -p cpu -c 12 ref_lj_cpu.sh
# CPU reference for an interacting system (3D LJ, N0=27 lattice start, rho~0.7, T=1,
# lam=0.05, P=32, mu=0): 12 independent seeds, combined afterwards.
B=build_$SLURM_JOB_ID; mkdir -p $B
g++ -O3 -march=native -std=c++17 -DDIM=3 worm_pimc.cpp -o $B/c3d || exit 1
A="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=3.379 rc=1.6895 N0=27"
for s in $(seq 1 12); do
  ( echo "seed=$s $($B/c3d $A steps=${STEPS:-1e8} therm=${THERM:-1e7} blocks=20 seed=$s | grep RESULT | tr '\n' ' ')" ) > $B/o$s &
done
wait
cat $B/o*
rm -rf $B
