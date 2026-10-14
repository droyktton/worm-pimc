#!/bin/bash
# CPU side of the N ~ 100 GPU-versus-CPU test: one independent seed per core, half of them
# from an N0=100 lattice and half from N0=64 (equilibration check).
# usage: sbatch -p cpu -N1 --exclusive validation/lj100_cpu.sh <A|B> [nseeds]
# output: validation/out/lj100_<sys>_cpu/seed<k>.txt
. validation/lj100_params.sh $1 || exit 1
ns=${2:-56}
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B
D=validation/out/lj100_${SYS_NAME}_cpu; mkdir -p $D
g++ -O3 -march=native -std=c++17 -DDIM=3 worm_pimc.cpp -o $B/c || exit 1
for s in $(seq 1 $ns); do
  n0=$(( s % 2 ? 100 : 64 ))
  ./$B/c $SYS N0=$n0 steps=${STEPS:-2e8} therm=${THERM:-2e7} blocks=20 meas=${MEAS:-20} seed=$((1000 + s)) > $D/seed$s.txt 2>&1 &
done
wait
grep -h RESULT $D/seed*.txt | head -4
rm -rf $B
