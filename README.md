# Worm-algorithm PIMC for bosons in continuous space (CPU + CUDA)

Based on Boninsegni, Prokof'ev, Svistunov, PRL 96, 070601 (2006). Units ħ = m = 1 by default
(λ = ħ²/2m is a parameter).
Grand-canonical ensemble, primitive approximation with P imaginary-time slices, Z/G sectors.
Moves: Open/Close, Insert/Remove, Advance/Recede (head and tail), Swap, Staging.

Full documentation (models, method, estimators, GPU implementation, demos, validation, performance, references):
**[docs/worm_pimc.md](docs/worm_pimc.md)**.

## Quick start on a desktop or laptop

Requirements:

- **CPU code:** a C++17 compiler (g++ ≥ 7 or clang).
- **GPU code:** an NVIDIA GPU and the CUDA toolkit (`nvcc` ≥ 11.5 for `-arch=native`).
- **Exact references and demos:** Python 3.
- **Figures:** matplotlib.

Platform notes:

- **Linux:** works as is.
- **Windows:** use WSL2. The CUDA toolkit for WSL gives GPU access.
- **macOS:** CPU code only (no CUDA).

```bash
git clone https://github.com/droyktton/worm-pimc.git && cd worm-pimc

# CPU: 1D bosons in a harmonic trap (a few seconds), and the comparison with the exact result
g++ -O3 -march=native -std=c++17 -DDIM=1 worm_pimc.cpp -o worm_pimc_1d
./worm_pimc_1d P=24 beta=1 mu=0 kappa=0.3 C=1e-3 steps=3e7 therm=1e6
python3 exact_pa.py --d 1 --beta 1 --mu 0 --kappa 0.3 --P 24

# GPU: compile for the GPU in this machine; float is much faster on GeForce cards
nvcc -O3 -std=c++17 -arch=native -DDIM=3 -DUSE_FLOAT worm_pimc_gpu.cu -o worm_pimc_gpu_3d_f
# ideal Bose gas in a periodic box, compared with the exact result
./worm_pimc_gpu_3d_f pbc=1 omega=0 P=24 beta=2 mu=-0.1 Lbox=3 C=1e-4 R=4096 steps=1e5 therm=2e4
python3 exact_ideal.py --d 3 --beta 2 --mu -0.1 --L 3

# demos (each one compiles what it needs; results in demos/out/)
bash demos/demo_trap.sh                                        # CPU, a few minutes
ARCH=-arch=native R=2048 bash demos/demo_superfluid.sh         # GPU
ARCH=-arch=native R=128  bash demos/demo_lj_eos.sh             # GPU, replica-per-warp mode
python3 docs/make_figures.py                                   # figures in docs/figures/
```

Hints for small machines:

- **GPU memory.** Lower the number of replicas `R` on laptop GPUs. The program checks free device memory and says how much it needs.
- **CPU-only machines.** Run the CPU code. To use several cores, start independent runs with different `seed=` and average them.

## Running on a SLURM cluster

See [slurm/README.md](slurm/README.md): `bash slurm/run_all_demos.sh` submits all demos and then the figures.
Compile and run on compute nodes, not on the login node.

## Usage (CPU)
    g++ -O2 -std=c++17 -DDIM=1 worm_pimc.cpp -o worm_pimc_1d    # DIM = 1, 2, 3
    ./worm_pimc_1d P=24 beta=1 mu=0 C=1e-3 Lmax=8 omega=1 kappa=0 steps=3e7 therm=3e6
    ./validate.sh      # compare with exact solutions

Parameters: `P`, `beta`, `mu`, `C`, `Lmax` (≤ P−2), `Lbox` (box for Insert; box side with pbc),
`omega` (harmonic trap), `kappa` (harmonic pair coupling), `steps`, `therm`, `meas`, `blocks`,
`debug=1` (consistency check), `seed`.

Further options (CPU and GPU):
- `pbc=1`: periodic box of side `Lbox`, with the minimum image used for links and pairs.
  - Open/Close/Swap use the free propagator summed over periodic images, and sample the image with probability ∝ G0.
  - Staging keeps the actual end-to-end displacement of the segment, so the winding number is preserved.
- `lam`: ħ²/2m.
- `pot`: pair potential.
  - `0`: harmonic, with coupling `kappa`.
  - `1`: Lennard-Jones, with parameters `eps` and `sig`.
  - `2`: Aziz HFDHE2 for ⁴He, in K and Å (use `lam=6.0596`).
- `rc`: pair cutoff, must be ≤ Lbox/2.
- `N0`: start from N0 straight closed worldlines on a lattice.

Output (Z-sector averages, jackknife errors):
- `N`; `Vbar` (mean potential energy); `E` (thermodynamic energy estimator).
- `D2`: ⟨D²⟩, where D is the total winding displacement; `rhos` = ⟨D²⟩/(2λdβ⟨N⟩).
- `Zfrac` (fraction of time in the Z sector) and acceptance ratios.

`C` only affects efficiency, not the results (checked for C = 0.1…0.001); aim for Zfrac ~ 0.1–0.5.

## Exact references
- `exact_pa.py`: bosons in a harmonic trap ± harmonic pair coupling.
  - Exact Z_N by a sum over cycle types, both for the *same* primitive discretisation (no Trotter error) and for the continuum.
  - Returns N, Vbar and E.
- `exact_ideal.py`: ideal Bose gas in a periodic box, with exact N, E, ⟨D²⟩ and ρs.
  - The primitive approximation is exact for free particles, so this is a strict test of the pbc code at any P.

Original CPU validation (3e7 steps, C=1e-3; exact PA vs MC):

| case | N exact | N MC | Vbar exact | Vbar MC |
|---|---|---|---|---|
| 1D β=1 μ=0 κ=0 | 1.9672 | 1.996(21) | 0.8120 | 0.825(8) |
| 1D κ=0.3 | 1.2392 | 1.2400(63) | 0.5991 | 0.6041(34) |
| 1D β=2 μ=−0.5 | 0.1782 | 0.1785(11) | 0.05690 | 0.05690(39) |
| 2D κ=0 | 1.1868 | 1.180(12) | 1.1608 | 1.151(11) |
| 2D μ=−0.3 κ=0.3 | 0.6091 | 0.6078(54) | 0.6380 | 0.6360(56) |

The periodic-box code (`validate_pbc.sh`) matches `exact_ideal.py` in 1D/2D/3D for N, E, ⟨D²⟩ and ρs.
For example, 3D β=2 L=3 gives ρs = 0.9591(16), against 0.95819 exact. With the extended moves, the trap cases also reproduce the exact E.

## GPU (CUDA): `worm_pimc_gpu.cu`
    nvcc -O3 -std=c++17 -arch=sm_86 -DDIM=3 [-DUSE_FLOAT] worm_pimc_gpu.cu -o worm_pimc_gpu_3d
    ./worm_pimc_gpu_3d W=32 R=1024 steps=1e5 therm=2e4 ...   # same parameters as the CPU code

The GPU code uses the same moves and acceptance ratios as the CPU code.
- **Replicas.** It runs `R` independent replicas; `steps` and `therm` are counted *per replica*.
- **`W=1`:** one thread runs one replica. Use it for few particles (N ≲ 10).
- **`W=32`:** one warp runs one replica. Use it for many particles.
  - The O(N) sums are split over the lanes and reduced with warp shuffles: the potential of a bead against its slice, the Swap normalisations and the measurements.
  - The reduced result is broadcast from lane 0. Every lane holds an identical copy of the RNG, so all lanes take identical decisions.
- **Reproducibility.** With the same seed, `W=1` and `W=32` give bit-identical results.
- **Errors.** Error bars come from a jackknife over groups of replicas.
- **Capacity.** `SM` sets the maximum beads per slice and `NB` the maximum beads per replica.
  - A move exceeding them is rejected, so those configurations get zero weight: this samples a truncated ensemble.
  - The rejections are reported as `overflow`; increase `SM`/`NB` unless the count is negligible.
- **Debugging.** `debug=1` checks the consistency of the linked lists between kernel chunks.
- **Inlining.** The move functions are `__forceinline__` on purpose. As ABI calls, ptxas (CUDA 13.0 and 13.3, -O1 and above) corrupted the local frame of the `W=1` kernel; the build is fine with `-G`.

### Validation
The scripts are `validate_gpu.sh`, `validate_pbc.sh`, `ref_lj_cpu.sh`, `lj_gpu.sh`, `lj_float.sh` and `test_warp.sh`.
- **Against exact results.** Trap cases (N, Vbar, E) and the periodic ideal gas in 1D/2D/3D (N, E, ⟨D²⟩, ρs) agree in double and float precision. The error bars are ~5–10× smaller than in the CPU tests.
- **Interacting system.** 3D Lennard-Jones (N ≈ 21), GPU vs CPU (12 seeds):

| | CPU | GPU W=32 double | GPU W=32 float |
|---|---|---|---|
| N | 21.1214(88) | 21.1151(50) | 21.1167(65) |
| Vbar | −51.742(47) | −51.710(26) | −51.724(33) |
| E | 2.951(58) | 2.973(27) | 2.953(21) |

### Performance
Benchmark: 3D LJ at ρσ³ = 0.7, T = 1, P = 32, thermalised production, total MC steps/s (`bench_N2.sh`, `bench_cpu_N.sh`):

| N | 1 core Xeon Gold 6330 | RTX 5070 Ti W=32 double | RTX 5070 Ti W=32 float | float vs 1 core |
|---|---|---|---|---|
| 8 | 2.0e6 | 7.4e6 | 5.3e7 | 27× |
| 27 | 4.2e5 | 6.3e6 | 4.8e7 | 115× |
| 64 | 1.6e5 | 5.1e6 | 3.3e7 | 204× |
| 125 | 8.8e4 | 3.2e6 | 2.0e7 | 223× |
| 216 | 6.2e4 | 1.8e6 | 9.5e6 | 153× |
| 512 | 3.0e4 | 8.2e5 | 2.4e6 | 79× |

- **Trap (N ≈ 1–2, `bench_gpu.sh`):** `W=1` float reaches ~2–3e8 steps/s, against 6.5e6 on one core.
- **Precision.** FP64 throughput on GeForce cards is 1/64 of FP32. Use `-DUSE_FLOAT` (validated) unless double precision is needed.

**To do:**
- Cell lists for large N; the rate currently drops ~1/N.
- Reduce the cost of thermalising every replica.
