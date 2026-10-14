# Systems of the GPU-versus-CPU test with N ~ 100 (sourced by lj100_gpu.sh / lj100_cpu.sh)
#  A: dense quantum LJ liquid, lam = 0.05, T = 1, rho ~ 0.63 (no exchange)
#  B: degenerate LJ gas, lam = 0.5, T = 1 (thermal wavelength 2.5 sigma), rho ~ 0.18: superfluid
#     crossover (rho_s/rho ~ 0.4), long exchange cycles
SYS_NAME=$1
case $1 in
  A) SYS="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.05 P=32 beta=1 mu=1 Lbox=5.4 rc=2.5 C=1e-4 Lmax=8 SM=192" ;;
  B) SYS="pbc=1 omega=0 pot=1 eps=1 sig=1 lam=0.5 P=32 beta=1 mu=3 Lbox=8 rc=2.5 C=1e-7 Lmax=8 SM=224" ;;
  *) echo "system must be A or B" >&2; exit 1 ;;
esac
