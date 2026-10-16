#!/bin/bash
# Clean 1D 4He (Aziz HFDHE2, lam=6.0596 K A^2) in a periodic box, grand canonical, CPU code: one long
# chain per array task (seed = task id), for the Luttinger parameter K = pi sqrt(<W^2><dN^2>) (RESULT3),
# density N/L, energy and cycles. The GPU replicas (W=1) are ~4 cores fast here and each replica gets too
# few steps to relax N and W (insert acceptance < 1%), so long CPU chains are used instead.
# usage: sbatch -p cpu -c1 --array=1-8 validation/luttinger.sh <L> <beta> <P> <mu>
# env: C (default 1e-7 (400/P)^2, Zfrac ~ 0.1-0.4), LMAX (default P/10), N0 (default 0.15 L),
#      STEPS (default 5e4 P), THERM (default STEPS/10), TAG
# output: validation/out/lutt/L<L>_b<beta>_P<P>_mu<mu><TAG>_s<seed>.txt
L=$1; beta=$2; P=$3; mu=$4; s=${SLURM_ARRAY_TASK_ID:-1}
unset CXX CC CXXFLAGS
B=build_${SLURM_JOB_ID:-$$}; mkdir -p $B validation/out/lutt
g++ -O3 -march=native -std=c++17 -DDIM=1 worm_pimc.cpp -o $B/c || exit 1
C=${C:-$(python3 -c "print('%.3g' % (1e-7 * (400 / $P) ** 2))")}
LMAX=${LMAX:-$(( P / 10 ))}
N0=${N0:-$(python3 -c "print(round(0.15 * $L))")}
STEPS=${STEPS:-$(python3 -c "print('%.3g' % (5e4 * $P))")}
THERM=${THERM:-$(python3 -c "print('%.3g' % ($STEPS / 10))")}
o=validation/out/lutt/L${L}_b${beta}_P${P}_mu${mu}${TAG}_s$s.txt
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)" > $o
$B/c pbc=1 omega=0 pot=2 lam=6.0596 Lbox=$L beta=$beta P=$P Lmax=$LMAX mu=$mu C=$C N0=$N0 \
  steps=$STEPS therm=$THERM seed=$((1000 * s + 7)) >> $o 2>&1
rm -rf $B
