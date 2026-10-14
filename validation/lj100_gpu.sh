#!/bin/bash
# GPU (W=32) side of the N ~ 100 GPU-versus-CPU test; two starts (N0) as equilibration check.
# usage: sbatch -p gpu --gres=gpu:1 validation/lj100_gpu.sh <A|B> <double|float> [N0 ...]
# output: validation/out/lj100_<sys>_gpu_<prec>_N0<n>.txt
. validation/lj100_params.sh $1 || exit 1
prec=$2; shift 2; starts=${@:-100 64}
ARCH=${ARCH:-"-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"}
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out
F=""; [ $prec = float ] && F=-DUSE_FLOAT
nvcc -O3 -std=c++17 $ARCH -DDIM=3 $F worm_pimc_gpu.cu -o $B/g || exit 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
for n0 in $starts; do
  o=validation/out/lj100_${SYS_NAME}_gpu_${prec}_N0$n0${TAG}.txt
  ./$B/g W=32 $SYS N0=$n0 R=${R:-512} steps=${STEPS:-1e7} therm=${THERM:-1e7} seed=$((7 + n0 + ${SEEDOFF:-0})) > $o 2>&1
  cat $o
done
rm -rf $B
