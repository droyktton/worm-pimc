#!/bin/bash
# quick checks of the cycle estimator: builds, sum_j n_j = N, W=1 vs W=32 bit-identical, debug, sanitizer
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B
nvcc -O3 -std=c++17 $ARCH -DDIM=3 worm_pimc_gpu.cu -o $B/g3d || exit 1
g++ -O3 -march=native -std=c++17 -DDIM=3 worm_pimc.cpp -o $B/c3d || exit 1
python3 exact_ideal.py --d 3 --beta 2 --mu -0.1 --L 3
I="pbc=1 omega=0 P=24 beta=2 mu=-0.1 Lbox=3 C=1e-4"
echo "--- CPU"; $B/c3d $I steps=2e7 therm=2e6 debug=1 | grep -E "RESULT|CYCLES|INCONS"
for w in 1 32; do echo "--- GPU W=$w"; $B/g3d $I W=$w rmaj=0 R=256 steps=2e4 therm=0 chunk=1e3 debug=1 | grep -E "RESULT |CYCLES|INCONS"; done
echo "--- GPU W=1 R=16384"; $B/g3d $I W=1 R=16384 steps=1e5 therm=5e4 | grep -E "RESULT|CYCLES"
SAN=/usr/local/cuda/bin/compute-sanitizer
L="pbc=1 omega=0 pot=1 lam=0.05 P=32 beta=1 mu=0 Lbox=3.379 rc=1.6895 N0=27 C=1e-2 SM=96 steps=1e3 chunk=1e3"
echo "--- sanitizer W=1";  $SAN --print-limit 2 $B/g3d W=1  R=32 $L 2>&1 | tail -1
echo "--- sanitizer W=32"; $SAN --print-limit 2 $B/g3d W=32 R=8  $L 2>&1 | tail -1
rm -rf $B
