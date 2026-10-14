#!/bin/bash
# pilot for system B: <N>, Zfrac and cycles versus mu (GPU float, W=32, short)
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B
nvcc -O3 -std=c++17 $ARCH -DDIM=3 -DUSE_FLOAT worm_pimc_gpu.cu -o $B/g || exit 1
for mu in ${MUS:--3 -2 -1.5 -1 -0.5 0}; do
  for C in ${CS:-1e-5 1e-6}; do
    echo "mu=$mu C=$C $($B/g W=32 pbc=1 omega=0 pot=1 lam=0.5 P=32 beta=1 mu=$mu Lbox=${LB:-10} rc=2.5 C=$C Lmax=8 SM=256 N0=100 R=128 steps=3e5 therm=3e5 seed=5 2>&1 | grep -E 'RESULT|CYCLES|WARN' | tr '\n' ' ' | cut -c1-600)"
  done
done
rm -rf $B
