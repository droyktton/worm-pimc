#!/bin/bash
#SBATCH --job-name=demo_trap
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4   # 4 seeds in parallel
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.log
# Demo 1 (CPU): trapped bosons vs exact. Submit from the repository root: sbatch slurm/demo_trap.sbatch
cd "$SLURM_SUBMIT_DIR" || exit 1
bash demos/demo_trap.sh
