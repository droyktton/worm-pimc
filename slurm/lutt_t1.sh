#!/bin/bash
unset NVCC_PREPEND_FLAGS CXX CC CXXFLAGS
B=build_t$SLURM_JOB_ID; mkdir -p $B
nvcc -O3 -std=c++17 -arch=native -DDIM=1 -DUSE_FLOAT worm_pimc_gpu.cu -o $B/g || exit 1
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
for mu in 0.5 2; do
/usr/bin/time -f "wall=%e" $B/g pbc=1 omega=0 pot=2 lam=6.0596 Lbox=100 beta=10 P=400 Lmax=40 mu=$mu C=1e-3 SM=64 R=1024 steps=2e3 therm=1e3 N0=12 seed=5 2>&1 | grep -vE "^CYCLES"
done
rm -rf $B
