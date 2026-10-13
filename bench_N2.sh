#!/bin/bash
# usage: sbatch -p gpu -w <node> --gres=gpu:1 -c 2 bench_N2.sh
# (a) float validation of the interacting system of ref_lj_cpu.sh (W=32)
# (b) production throughput vs N after thermalisation (TIMING prod_rate):
#     3D LJ, rho = 0.7, T = 1, lam = 0.05, P = 32, N0 lattice start.
ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120"
B=build_$SLURM_JOB_ID; mkdir -p $B
g++ -O3 -march=native -std=c++17 -DDIM=3 worm_pimc.cpp -o $B/cpu || exit 1
nvcc -O3 -std=c++17 $ARCH -DDIM=3 worm_pimc_gpu.cu -o $B/gd || exit 1
nvcc -O3 -std=c++17 $ARCH -DDIM=3 -DUSE_FLOAT worm_pimc_gpu.cu -o $B/gf || exit 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; lscpu | grep "Model name"
A="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=3.379 rc=1.6895 N0=27 SM=96"
echo "LJ27 float W=32 $($B/gf $A W=32 R=2048 steps=2e6 therm=1e6 seed=8 | grep RESULT | tr '\n' ' ')"
for N in 8 27 64 125 216 512; do
  read L RC SM TH TC TG <<< $(python3 -c "
N=$N; L=(N/0.7)**(1/3); print(f'{L:.6f} {min(2.5,L/2):.6f} {int(1.5*N)+32} 20000 {int(max(2e5, 6e6/N))} {int(max(2e3, 2e5/N))}')")
  A="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=$L rc=$RC N0=$N"
  r=$($B/cpu $A steps=$TC therm=$TH seed=3 | grep TIMING); echo "N0=$N CPU1core   $r"
  for exe in gd gf; do
    for W in 1 32; do
      [ $W = 1 ] && [ $N -gt 27 ] && continue
      R=8192; [ $W = 32 ] && R=1024
      o=$($B/$exe $A W=$W R=$R SM=$SM steps=$TG therm=$TH chunk=1e4 seed=3 2>&1 | grep -E "TIMING|error|RESULT N")
      echo "N0=$N $exe W=$W R=$R  $(echo "$o" | tr '\n' ' ' | sed -E 's/RESULT N= ([0-9.]+).*overflow=([0-9]+)/N=\1 ovf=\2/')"
    done
  done
done
rm -rf $B
