#!/bin/bash
# usage: sbatch -p gpu -w <node> --gres=gpu:1 bench_gpu.sh
# throughput (MC steps/s, all replicas) vs number of replicas R, double and float
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_$SLURM_JOB_ID; mkdir -p $B
for d in 1 2; do
  nvcc -O3 -std=c++17 $ARCH -DDIM=$d worm_pimc_gpu.cu -o $B/g${d}d || exit 1
  nvcc -O3 -std=c++17 $ARCH -DDIM=$d -DUSE_FLOAT worm_pimc_gpu.cu -o $B/g${d}f || exit 1
done
nvidia-smi --query-gpu=name --format=csv,noheader
for exe in g1d g1f g2d g2f; do
  case $exe in g1*) args="P=24 beta=1 mu=0 kappa=0";; g2*) args="P=24 beta=1 mu=-0.3 kappa=0.3";; esac
  for R in 512 2048 8192 32768 65536; do
    out=$($B/$exe $args C=1e-3 Lmax=8 Lbox=12 R=$R steps=2e4 therm=2e4 SM=32 seed=3 | grep RESULT)
    echo "$exe R=$R $out"
  done
done
rm -rf $B
