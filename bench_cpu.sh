#!/bin/bash
# usage: sbatch -p cpu -n1 -c1 bench_cpu.sh   -- CPU baseline (1 core) for the trap cases of bench_gpu.sh
# CPU baseline (single core) for the same cases as bench_gpu.sh
B=build_$SLURM_JOB_ID; mkdir -p $B
for d in 1 2; do g++ -O3 -march=native -std=c++17 -DDIM=$d worm_pimc.cpp -o $B/c${d}d; done
lscpu | grep "Model name"; nproc
echo "c1d $($B/c1d P=24 beta=1 mu=0 kappa=0 C=1e-3 Lmax=8 Lbox=12 steps=3e7 therm=3e6 seed=3 | grep RESULT)"
echo "c2d $($B/c2d P=24 beta=1 mu=-0.3 kappa=0.3 C=1e-3 Lmax=8 Lbox=12 steps=3e7 therm=3e6 seed=3 | grep RESULT)"
rm -rf $B
