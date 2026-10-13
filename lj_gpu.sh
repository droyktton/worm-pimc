#!/bin/bash
# usage: sbatch -p gpu --gres=gpu:1 lj_gpu.sh
# GPU (W=32; W=1 is bit-identical, see test_warp.sh) on the same interacting
# system as ref_lj_cpu.sh; plus a diagnostic rerun of a blank bench_N entry
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_$SLURM_JOB_ID; mkdir -p $B
nvcc -O3 -std=c++17 $ARCH -DDIM=3 worm_pimc_gpu.cu -o $B/g3d || exit 1
nvidia-smi --query-gpu=name --format=csv,noheader
A="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=3.379 rc=1.6895 N0=27 SM=96"
echo "W=32 $($B/g3d $A W=32 R=2048 steps=2e6 therm=1e6 seed=6 | grep RESULT | tr '\n' ' ')"
echo "--- diagnostic: bench_N blank entry (N0=125 double W=1)"
L=$(python3 -c "print((125/0.7)**(1/3))"); RC=$(python3 -c "print(min(2.5,$L/2))")
$B/g3d pbc=1 omega=0 pot=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=$L rc=$RC N0=125 W=1 R=15275 SM=219 NB=7008 steps=1600 therm=0 chunk=1600 seed=3; echo "exit=$?"
rm -rf $B
