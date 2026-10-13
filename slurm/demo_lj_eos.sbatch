#!/bin/bash
#SBATCH --job-name=demo_lj_eos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=node210
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%j.log
# Demo 3 (GPU, W=32): quantum Lennard-Jones equation of state. Submit from the repository root.
# node210 is excluded: its four GPUs hold a vLLM server outside SLURM (~1 GB free each).
cd "$SLURM_SUBMIT_DIR" || exit 1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
bash demos/demo_lj_eos.sh
