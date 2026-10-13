# Running on a SLURM cluster

Batch scripts for the demos. Submit them **from the repository root**, so that paths and `slurm/logs/` resolve.
Every job compiles the code on its compute node: nothing runs on the login node.

```bash
bash slurm/run_all_demos.sh            # all demos, then the figures (job dependency)
sbatch slurm/demo_trap.sbatch          # Demo 1, CPU partition
sbatch slurm/demo_superfluid.sbatch    # Demo 2, one GPU
sbatch slurm/demo_lj_eos.sbatch        # Demo 3, one GPU (W=32)
sbatch slurm/make_figures.sbatch       # regenerate docs/figures/
squeue -u $USER                        # follow; logs in slurm/logs/, results in demos/out/
```

- **Partitions.** The scripts assume partitions `cpu` and `gpu` and `--gres=gpu:1`. Adjust the `#SBATCH` lines for another cluster.
- **GPU node excluded.** The GPU scripts exclude `node210`: its GPUs are occupied by a service running outside SLURM.
- **Smaller GPUs.** On GPUs with little free memory, lower the number of replicas, e.g. `sbatch --export=ALL,R=2048 slurm/demo_superfluid.sbatch`. The program aborts with a clear message if the requested replicas do not fit in device memory.
- **Compute capability.** Binaries target sm_86 (RTX 30xx, A10) and sm_120 (RTX 50xx). For other GPUs, set e.g. `--export=ALL,ARCH="-arch=sm_89"`.
- **More scripts.** The validation and benchmark scripts in the repository root (`validate_*.sh`, `bench_*.sh`, `ref_lj_cpu.sh`, `lj_*.sh`, `test_warp.sh`) are also batch jobs. The `sbatch` line to use is in each header.
