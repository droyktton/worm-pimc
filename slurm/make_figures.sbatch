#!/bin/bash
#SBATCH --job-name=make_figures
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x_%j.log
# Regenerate docs/figures/*.png from demos/out/*.csv and docs/data/. Submit from the repository root.
cd "$SLURM_SUBMIT_DIR" || exit 1
python3 docs/make_figures.py && ls -l docs/figures
