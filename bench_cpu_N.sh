#!/bin/bash
# usage: sbatch -p cpu -c 1 bench_cpu_N.sh   -- CPU (1 core) production rate vs N, same system as bench_N2.sh
B=build_$SLURM_JOB_ID; mkdir -p $B
g++ -O3 -march=native -std=c++17 -DDIM=3 worm_pimc.cpp -o $B/cpu || exit 1
lscpu | grep "Model name"
for N in 8 27 64 125 216 512; do
  read L RC TC <<< $(python3 -c "
N=$N; L=(N/0.7)**(1/3); print(f'{L:.6f} {min(2.5,L/2):.6f} {int(max(2e5, 6e6/N))}')")
  echo "N0=$N CPU1core $($B/cpu pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=0 C=1e-2 Lmax=8 Lbox=$L rc=$RC N0=$N steps=$TC therm=20000 seed=3 | grep TIMING)"
done
rm -rf $B
