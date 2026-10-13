#!/bin/bash
# usage: sbatch -p gpu --gres=gpu:1 lj_float.sh  -- float W=32 on the system of ref_lj_cpu.sh, plus N0=27 float rates
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_$SLURM_JOB_ID; mkdir -p $B
nvcc -O3 -std=c++17 $ARCH -DDIM=3 -DUSE_FLOAT worm_pimc_gpu.cu -o $B/gf || exit 1
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
A="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=3.379 rc=1.6895 N0=27 SM=96"
echo "LJ27 float W=32 $($B/gf $A W=32 R=2048 steps=2e6 therm=1e6 seed=8 2>&1 | grep -E 'RESULT|TIMING|error' | tr '\n' ' ')"
L=$(python3 -c "print('%.6f'%((27/0.7)**(1/3)))"); RC=$(python3 -c "print('%.6f'%(min(2.5,$L/2)))")
B2="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=$L rc=$RC N0=27 SM=72"
echo "N0=27 gf W=1  $($B/gf $B2 W=1  R=8192 steps=2e4 therm=2e4 chunk=1e4 seed=3 2>&1 | grep -E 'TIMING|error')"
echo "N0=27 gf W=32 $($B/gf $B2 W=32 R=1024 steps=7407 therm=2e4 chunk=1e4 seed=3 2>&1 | grep -E 'TIMING|error')"
rm -rf $B
