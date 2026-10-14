#!/bin/bash
# compute-sanitizer on both GPU modes with the cycle estimator (3D LJ N0=27 and 3D ideal gas, short)
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B
nvcc -O3 -std=c++17 $ARCH -DDIM=3 worm_pimc_gpu.cu -o $B/g3d || exit 1
SAN=/usr/local/cuda/bin/compute-sanitizer
L="pbc=1 omega=0 pot=1 lam=0.05 P=32 beta=1 mu=0 Lbox=3.379 rc=1.6895 N0=27 C=1e-2 SM=96 steps=200 therm=0 chunk=200 meas=5"
I="pbc=1 omega=0 P=24 beta=2 mu=-0.1 Lbox=3 C=1e-4 N0=5 SM=128 steps=2000 therm=0 chunk=1000 meas=5"
for w in 1 32; do
  echo "--- LJ W=$w";    $SAN --print-limit 2 $B/g3d W=$w R=8 $L 2>&1 | tail -1
  echo "--- ideal W=$w"; $SAN --print-limit 2 $B/g3d W=$w R=8 $I 2>&1 | tail -1
done
rm -rf $B
