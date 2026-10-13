#!/bin/bash
# Submit the three demos and, once all of them succeed, regenerate the figures.
# usage (from the repository root): bash slurm/run_all_demos.sh
set -e
cd "$(dirname "$0")/.."
j1=$(sbatch --parsable slurm/demo_trap.sbatch)
j2=$(sbatch --parsable slurm/demo_superfluid.sbatch)
j3=$(sbatch --parsable slurm/demo_lj_eos.sbatch)
j4=$(sbatch --parsable --dependency=afterok:$j1:$j2:$j3 slurm/make_figures.sbatch)
echo "submitted: demo_trap=$j1 demo_superfluid=$j2 demo_lj_eos=$j3 make_figures=$j4 (after the demos)"
echo "follow with: squeue -u \$USER ; logs in slurm/logs/, results in demos/out/"
