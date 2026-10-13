#!/bin/bash
# usage: sbatch -p gpu --gres=gpu:1 test_warp.sh
# W=1 vs W=32: consistency checks (debug=1), compute-sanitizer, same-seed comparison
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_$SLURM_JOB_ID; mkdir -p $B
for d in 2 3; do nvcc -O3 -std=c++17 $ARCH -DDIM=$d worm_pimc_gpu.cu -o $B/g${d}d || exit 1; done
SAN=/usr/local/cuda/bin/compute-sanitizer
A="R=256 steps=2e4 therm=0 kappa=0.3 mu=-0.3 C=1e-3 chunk=1e3 debug=1"
for w in 1 32; do echo "--- 2D trap W=$w"; $B/g2d W=$w rmaj=0 $A | grep -E "RESULT|INCONS"; done
I="pbc=1 omega=0 P=24 beta=2 mu=-0.1 Lbox=3 C=1e-3 R=256 steps=2e4 therm=0 chunk=1e3 debug=1"
for w in 1 32; do echo "--- 3D ideal W=$w"; $B/g3d W=$w rmaj=0 $I | grep -E "RESULT|INCONS"; done
L="pbc=1 omega=0 pot=1 lam=0.05 P=32 beta=1 mu=0 Lbox=3.379 rc=1.6895 N0=27 C=1e-2 SM=96 R=64 steps=2e4 therm=0 chunk=1e3 debug=1"
for w in 1 32; do echo "--- 3D LJ N0=27 W=$w"; $B/g3d W=$w rmaj=0 $L | grep -E "RESULT|INCONS"; done
echo "--- 3D LJ N0=27 W=32 rmaj=1"; $B/g3d W=32 rmaj=1 $L | grep -E "RESULT|INCONS"
echo "--- sanitizer W=1";  $SAN --print-limit 2 $B/g3d W=1  R=32 $L steps=1e3 chunk=1e3 2>&1 | tail -1
echo "--- sanitizer W=32"; $SAN --print-limit 2 $B/g3d W=32 R=8  $L steps=1e3 chunk=1e3 2>&1 | tail -1
rm -rf $B
