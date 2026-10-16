#!/bin/bash
unset CXX CC CXXFLAGS
B=build_c$SLURM_JOB_ID; mkdir -p $B
g++ -O3 -march=native -std=c++17 -DDIM=1 worm_pimc.cpp -o $B/c || exit 1
$B/c pbc=1 omega=0 pot=2 lam=6.0596 Lbox=100 beta=10 P=400 Lmax=40 mu=1 C=1e-7 N0=19 steps=2e7 therm=1e6 seed=3 2>&1 | grep -vE "^CYCLES"
rm -rf $B
