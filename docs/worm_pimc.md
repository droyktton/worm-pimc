# Worm-algorithm path-integral Monte Carlo for bosons — technical documentation

This document describes the physical models, the method, the implementation (CPU and CUDA), how to run it, the demos, the validation against exact results, and the performance measurements. All formulas below are the ones implemented in `worm_pimc.cpp` and `worm_pimc_gpu.cu`.

**Contents**

1. [Models](#1-models)
2. [Path-integral representation](#2-path-integral-representation)
3. [The worm algorithm](#3-the-worm-algorithm)
4. [Estimators](#4-estimators)
5. [Exact reference solutions](#5-exact-reference-solutions)
6. [GPU implementation](#6-gpu-implementation)
7. [Usage](#7-usage)
8. [Demos](#8-demos)
9. [Validation](#9-validation)
10. [Performance](#10-performance)
11. [Limitations and outlook](#11-limitations-and-outlook)
12. [References](#12-references)

---

## 1. Models

We consider $N$ identical bosons of mass $m$ in $d = 1, 2, 3$ dimensions (compile-time `DIM`):

```math
H \;=\; -\lambda \sum_{i=1}^{N} \nabla_i^2 \;+\; \sum_{i=1}^{N} V_{\rm ext}(\mathbf r_i) \;+\; \sum_{i<j} v\big(|\mathbf r_i - \mathbf r_j|\big),
\qquad \lambda = \frac{\hbar^2}{2m}.
```

The default is $\hbar = m = 1$, that is $\lambda = 1/2$ (parameter `lam`).

**External potential** (parameter `omega`, $V_{\rm ext} = 0$ when `omega=0`):

```math
V_{\rm ext}(\mathbf r) = \tfrac12\,\omega^2 r^2 .
```

**Pair potentials** (parameter `pot`):

| `pot` | potential | parameters |
|---|---|---|
| 0 | harmonic coupling $v(r) = \tfrac12 \kappa r^2$ | `kappa` (no pair term if `kappa=0`) |
| 1 | Lennard-Jones $v(r) = 4\varepsilon\left[(\sigma/r)^{12} - (\sigma/r)^6\right]$ | `eps`, `sig` |
| 2 | Aziz HFDHE2 for ⁴He (K, Å) | built in; use `lam=6.0596` K Å² |

The Aziz HFDHE2 potential [5] is

```math
v(r) = \epsilon\left[A\,e^{-\alpha x} - F(x)\left(\frac{C_6}{x^6} + \frac{C_8}{x^8} + \frac{C_{10}}{x^{10}}\right)\right],
\qquad x = \frac{r}{r_m},
\qquad F(x) = \begin{cases} e^{-(D/x - 1)^2} & x < D \\ 1 & x \ge D \end{cases}
```

with $\epsilon = 10.8$ K, $r_m = 2.9673$ Å, $A = 0.5448504\times10^6$, $\alpha = 13.353384$, $D = 1.241314$, $C_6 = 1.3732412$, $C_8 = 0.4253785$ and $C_{10} = 0.1781$.

All pair potentials are **truncated** (not shifted) at the cutoff `rc`: $v(r) = 0$ for $r \gt r_c$. No tail corrections are applied.

**Boundary conditions.** Two options:

- **Open space** (`pbc=0`). This is typically combined with the trap. `Lbox` then only defines the region where Insert places new worms.
- **Periodic cubic box** of side $L$ = `Lbox` (`pbc=1`). The minimum-image convention applies to every pair distance and every imaginary-time link, and $r_c \le L/2$ is required.

---

## 2. Path-integral representation

**Grand partition function.** In the grand-canonical ensemble at temperature $T = 1/\beta$ and chemical potential $\mu$,

```math
\Xi = \sum_{N} \frac{e^{\beta\mu N}}{N!} \sum_{\mathcal P\in S_N} \int d\mathbf R\; \langle \mathbf R \,|\, e^{-\beta H} \,|\, \mathcal P \mathbf R\rangle ,
```

where $\mathcal P$ runs over the permutations of the particle labels.

**Primitive approximation.** Imaginary time is split into $P$ slices of width $\tau = \beta/P$ (parameter `P`), and each short-time density matrix is replaced by

```math
\rho(\mathbf R, \mathbf R'; \tau) \;\approx\; \prod_i \rho_0(\mathbf r_i, \mathbf r'_i; \tau)\; e^{-\tau V(\mathbf R)} ,
\qquad
\rho_0(\mathbf r, \mathbf r'; t) = (4\pi\lambda t)^{-d/2} \exp\!\left[-\frac{|\mathbf r - \mathbf r'|^2}{4\lambda t}\right],
```

with a systematic error of $O(\tau^2)$ in thermodynamic averages [2].

**Configurations.** A configuration is a set of *beads*. Each bead lives on a time slice $k \in \lbrace 0,\dots,P-1\rbrace$, carries a position, and is joined by *links* to a predecessor on slice $k-1$ and a successor on slice $k+1$ (indices mod $P$). Closed chains of links are the worldlines; a worldline winding $n$ times around imaginary time represents an exchange cycle of $n$ particles.

**Configuration weight.** The weight of a configuration $X$ is

```math
W(X) = C^{\,[X\in G]}\; \prod_{\text{beads}} e^{\mu\tau}\; \prod_{\text{links}} \rho_0(\Delta\mathbf r; \tau)\; \prod_{k=0}^{P-1} e^{-\tau V_k},
\qquad
V_k = \sum_{b\in k} V_{\rm ext}(\mathbf r_b) + \sum_{\substack{b<b'\\ b,b'\in k}} v(|\mathbf r_b-\mathbf r_{b'}|).
```

In the sector of closed worldlines (the **Z sector**) the number of beads is $N_b = NP$, so $\prod e^{\mu\tau} = e^{\beta\mu N}$ and $W$ is the primitive-approximation integrand of $\Xi$.

---

## 3. The worm algorithm

The worm algorithm of Boninsegni, Prokof'ev and Svistunov [1,2] enlarges the configuration space with a **G sector**. A G configuration contains exactly one open worldline (the *worm*) running from the *tail* (Masha, $\mathcal M$: no predecessor) to the *head* (Ira, $\mathcal I$: no successor), in addition to any number of closed worldlines.

- The G sector carries the arbitrary weight factor $C$ (parameter `C`). Its configurations sample the one-body Matsubara Green function.
- **All averages are taken in the Z sector.** $C$ only tunes the fraction of time spent in Z, $Z_{\rm frac}$ (printed as `Zfrac`); a value of 0.05–0.5 is efficient.
- Permutation sampling is local and ergodic: exchange cycles are built by the Swap move while the worm is open, and particle-number fluctuations come from Insert/Remove and Advance/Recede.

### 3.1 Moves and acceptance probabilities

**Move selection.** At each step one of 12 options is chosen with equal probability:

- Open, Close, Insert, Remove;
- Advance and Recede, for the head and for the tail separately (4 options);
- Swap;
- Staging (3 of the 12 options).

The two members of every complementary pair have the same selection probability, so selection probabilities cancel in the acceptance ratios.

**Notation.**

- $\Delta V_{\rm new}$ ($\Delta V_{\rm old}$) is the potential energy of the beads that are added (removed). Each bead contributes $`U(b) = V_{\rm ext}(\mathbf r_b) + \sum_{b'\in k_b,\, b'\ne b} v(\lvert \mathbf r_b - \mathbf r_{b'}\rvert )`$, and for a re-sampled segment the difference new − old is used.
- $\bar\rho_0$ is the image-summed free propagator (Section 3.2). Without pbc, $\bar\rho_0 = \rho_0$.
- Acceptance is $A = \min(1, \cdot)$ of the quantities below.

| Move | Proposal | Acceptance ratio |
|---|---|---|
| **Open** (Z→G) | Pick a bead $\mathcal I$ uniformly among the $N_b$ beads and $m\in\lbrace 1,\dots,L_{\max}\rbrace$ uniformly; delete the $m-1$ beads following $\mathcal I$. The bead $m$ links ahead becomes the tail $\mathcal M$. | $`\dfrac{C\, N_b\, L_{\max}\; e^{-(m-1)\mu\tau}\; e^{+\tau\Delta V_{\rm old}}}{\bar\rho_0(\mathbf r_{\mathcal I}, \mathbf r_{\mathcal M}; m\tau)}`$ |
| **Close** (G→Z) | Allowed if the slice gap $g$ from $\mathcal I$ to $\mathcal M$ is $\le L_{\max}$. Choose a periodic image of $\mathcal M$ with probability $\propto \rho_0$, then generate the $g-1$ beads with a Lévy bridge. | $`\dfrac{\bar\rho_0(\mathbf r_{\mathcal I}, \mathbf r_{\mathcal M}; g\tau)\; e^{(g-1)\mu\tau}\; e^{-\tau\Delta V_{\rm new}}}{C\, N_b'\, L_{\max}}`$, with $N_b'$ the number of beads after closing |
| **Insert** (Z→G) | Uniform slice ($P$ choices); $\ell \in\lbrace 0,\dots,L_{\max}-1\rbrace$ links uniformly; first bead uniform in the box of volume $V=L^d$; the remaining $\ell$ beads by free Gaussian steps. | $`C\, V\, P\, L_{\max}\; e^{(\ell+1)\mu\tau}\; e^{-\tau\Delta V_{\rm new}}`$ |
| **Remove** (G→Z) | Only if the worm has $\le L_{\max}$ beads and the tail lies inside the box; delete the whole worm. | inverse of Insert, with $\Delta V_{\rm old}$ |
| **Advance** (head or tail) | $m\in\lbrace 1,\dots,L_{\max}\rbrace$ new beads by free Gaussian steps from the head (forward) or the tail (backward). | $`e^{m\mu\tau}\, e^{-\tau\Delta V_{\rm new}}`$ |
| **Recede** (head or tail) | Delete $m\in\lbrace 1,\dots,L_{\max}\rbrace$ beads from the head or tail; the worm must keep at least one bead. | $`e^{-m\mu\tau}\, e^{+\tau\Delta V_{\rm old}}`$ |
| **Swap** | Choose $m\in\lbrace 1,\dots,L_{\max}\rbrace$. The candidates $\Omega$ are the beads $j$ on slice $k_{\mathcal I}+m$ whose $m$-th predecessor $q$ exists. Pick $j$ with probability $\bar\rho_0(\mathbf r_{\mathcal I},\mathbf r_j;m\tau)/\Sigma_{\mathcal I}$; delete the $m-1$ beads between $q$ and $j$; bridge $\mathcal I \to j$; $q$ becomes the new head. | $`\dfrac{\Sigma_{\mathcal I}}{\Sigma_q}\, e^{-\tau(\Delta V_{\rm new} - \Delta V_{\rm old})}`$, where $\Sigma_x = \sum_{j\in\Omega} \bar\rho_0(\mathbf r_x, \mathbf r_j; m\tau)$ |
| **Staging** (any sector) | Pick a random bead $a$ and $m\in\lbrace 2,\dots,L_{\max}\rbrace$ links; the segment must not contain the head. Re-sample the $m-1$ interior beads with a Lévy bridge, keeping both endpoints and the segment's end-to-end displacement. | $e^{-\tau(\Delta V_{\rm new} - \Delta V_{\rm old})}$ |

In every move the kinetic weight cancels exactly against the proposal density; only potential energy, chemical potential and normalisations remain. For Insert and Advance, the free Gaussian steps reproduce $\prod\rho_0$. For bridges, see below.

**Lévy bridge** [3]. The $m-1$ interior points between $\mathbf a$ (slice 0) and the endpoint $\mathbf b$ (slice $m$) are generated sequentially:

```math
\mathbf r_j = \mathbf r_{j-1} + \frac{\mathbf b - \mathbf r_{j-1}}{m-j+1} + \sqrt{2\lambda\tau\,\frac{m-j}{m-j+1}}\;\boldsymbol\xi_j,
\qquad \boldsymbol\xi_j \sim \mathcal N(0, 1)^d .
```

The joint density of the bridge is $`\prod_{\rm links}\rho_0 \,/\, \rho_0(\mathbf b - \mathbf a; m\tau)`$.

### 3.2 Periodic boundary conditions

With pbc every link displacement is taken as the minimum image. This requires $\sqrt{2\lambda\tau} \ll L/2$.

The displacement over $m \le L_{\max}$ links, however, need not be small compared with $L$. Its periodic images then all contribute. The code therefore uses the image-summed propagator, which factorises over Cartesian components:

```math
\bar\rho_0(\mathbf r, \mathbf r'; t) = \prod_{\alpha=1}^{d} \frac{1}{\sqrt{4\pi\lambda t}} \sum_{n=-2}^{2} \exp\!\left[-\frac{(x_\alpha + nL)^2}{4\lambda t}\right],
\qquad x_\alpha = \mathrm{mimg}(r'_\alpha - r_\alpha).
```

- **Close and Swap** pick the image $n$ of each component with probability $\propto$ its term, and then build the bridge towards that unwrapped point. The probability of proposing a given path is then $\prod\rho_0(\text{links}) / \bar\rho_0$, which is why $\bar\rho_0$ appears in the acceptance ratios of Open, Close and Swap.
- **Staging** keeps the *actual* end-to-end displacement of the old segment, i.e. the sum of its minimum-image links. Staging therefore never changes the winding number, and the move stays symmetric.

An earlier version used only the nearest image. It violated detailed balance whenever $\sqrt{2\lambda m\tau}$ was not negligible against $L/2$: for the 3D ideal gas at $\beta=2$, $L=3$ it gave $E = 1.69$, against the exact $0.142$. Sections 3.2 and 9.2 document the fix and its validation.

---

## 4. Estimators

All estimators are accumulated in the Z sector every `meas` steps.

| Quantity | Estimator |
|---|---|
| Particle number | $N = N_b / P$ |
| Potential energy | $\bar V = \frac1P\sum_{k} V_k$ |
| Thermodynamic energy | $`E = \frac{1}{P}\left[\frac{d\,N_b}{2\tau} - \sum_{\rm links}\frac{\lvert\Delta\mathbf r\rvert^2}{4\lambda\tau^2}\right] + \bar V`$ |
| Winding | $\mathbf D = \sum_{\rm links}\Delta\mathbf r$ (minimum image), reported as $D^2 = \lvert\mathbf D\rvert^2$ |
| Superfluid fraction | $`\rho_s/\rho = \dfrac{\langle D^2\rangle}{2\lambda\, d\,\beta\,\langle N\rangle}`$ |
| Exchange cycles | $n_j$ = number of particles in permutation cycles of length $j$, for $j = 1,\dots,11$; the last bin collects $j\ge12$, and $\sum_j n_j = N$ |

**Energy.** The thermodynamic estimator is $-\partial_\beta$ of $\ln W$ at fixed $P$ and fugacity. In the Z sector the number of links equals $N_b$, so its average equals the internal energy of the discretised system *exactly* (no Trotter error relative to the PA reference).

**Exchange cycles.** A closed worldline of $j$ particles has $jP$ beads and crosses slice 0 exactly $j$ times. The estimator walks the loop from every bead of slice 0 and adds 1 to bin $j = \ell/P$, where $\ell$ is the number of beads in the loop, so each cycle of length $j$ contributes $j$ to $n_j$. The number of bins is the compile-time constant `NCYC` = 12.

**Superfluid fraction.** This is the Pollock–Ceperley winding-number estimator [4], $\rho_s/\rho = m\langle W^2\rangle L^2/(d\hbar^2\beta N)$ with $\mathbf D = L\mathbf W$. In the grand-canonical ensemble the ratio of averages is used.

**Statistical errors.**

- **CPU:** the production run is split into `blocks` consecutive blocks.
- **GPU:** the replicas are split into `blocks` groups. Replicas are independent, so the groups are uncorrelated by construction.
- In both cases every average is a ratio estimator $\sum O / \sum n_Z$, and errors (including that of the derived $\rho_s/\rho$) come from a delete-one jackknife over blocks or groups.

---

## 5. Exact reference solutions

### 5.1 Harmonic trap with harmonic pair coupling (`exact_pa.py`)

**Normal modes.** For $V_{\rm ext} = \frac12\omega^2 r^2$ and $v = \frac12\kappa r^2$ the Hamiltonian is quadratic. It has a centre-of-mass mode of frequency $\omega$ and $N-1$ relative modes of frequency $\omega_r = \sqrt{\omega^2 + N\kappa}$, per Cartesian component.

**Canonical partition function.** $Z_N = \frac{1}{N!}\sum_{\mathcal P}\mathrm{tr}(\mathcal P e^{-\beta H})$ is evaluated as a sum over cycle types (integer partitions of $N$, with weight $1/\prod_c c^{m_c} m_c!$). A permutation acts on the coordinates through an orthogonal matrix whose eigenvalues are roots of unity $e^{2\pi i k/c}$, one set per cycle. Each (mode, eigenvalue $\theta$) pair contributes a factor:

- **continuum:** $\dfrac{e^{-\beta\omega_k/2}}{1 - e^{i\theta}e^{-\beta\omega_k}}$;
- **primitive approximation with $P$ slices:** $\dfrac{\lambda_0^P}{1 - e^{i\theta}\varrho^P}$. Here $\lambda_0\varrho^n$ are the eigenvalues of the PA transfer kernel of a 1D oscillator (Mehler formula), with $\varrho + 1/\varrho = 2 + \tau^2\omega_k^2$.

The PA version describes *exactly the same discretisation the Monte Carlo samples*, so the comparison carries no Trotter error.

**Grand-canonical averages.**

- $\Xi = \sum_{N\le N_{\max}} e^{\beta\mu N} Z_N$, and $\langle N\rangle$ follows directly from it.
- $\bar V$ follows from the derivative with respect to a scale factor $s$ of the potential ($\omega^2\to s\omega^2$, $\kappa\to s\kappa$).
- $E = -\partial_\beta\ln\Xi|_\mu + \mu\langle N\rangle$, at fixed $P$.

The weight of $N = N_{\max}$ is printed to check the truncation.

**Exchange cycles.** Each term of the sum over cycle types is the weight of that cycle type in $Z_N$. Hence $`\langle n_j\rangle = \Xi^{-1}\sum_N e^{\beta\mu N}\sum_{\rm types} j\, m_j\, Z_N^{\rm (type)}`$ is exact too, both for the PA and in the continuum (`--cycles`, with `--J` bins).

### 5.2 Ideal Bose gas in a periodic box (`exact_ideal.py`)

**Why it is exact for the PA.** For free particles the primitive approximation is exact. The only difference from the sampled weight is the minimum-image truncation of single links, of order $e^{-L^2/16\lambda\tau}$.

**Cycle expansion.** The grand potential is

```math
\ln\Xi = \sum_{j\ge1}\frac{e^{\beta\mu j}}{j}\, z(j)^d,
\qquad
z(j) = \sum_{w\in\mathbb Z} z(j,w),
\qquad
z(j,w) = \frac{L}{\sqrt{4\pi\lambda j\beta}}\, e^{-(wL)^2/4\lambda j\beta}.
```

Here $z(j,w)$ is the 1D single-particle trace at imaginary time $j\beta$ with winding number $w$.

**Observables.** Exchange cycles of length $j$ and winding vector $\mathbf w$ are independent Poisson variables, which gives

```math
\langle N\rangle = \sum_j e^{\beta\mu j} z(j)^d,
\qquad
\langle D^2\rangle = \sum_j \frac{e^{\beta\mu j}}{j}\, d\, L^2 \Big(\sum_w w^2 z(j,w)\Big)\, z(j)^{d-1},
\qquad
E = -\partial_\beta\ln\Xi|_\mu + \mu\langle N\rangle .
```

The mean number of particles in cycles of length $j$ is the $j$-th term of $\langle N\rangle$, $\langle n_j\rangle = e^{\beta\mu j} z(j)^d$ (`--cycles`).

---

## 6. GPU implementation

`worm_pimc_gpu.cu` runs `R` **independent replicas** (Markov chains) of exactly the same algorithm. The configuration of every replica lives in fixed-capacity arrays in global memory:

- per bead: positions, slice, predecessor, successor, and positions inside the per-slice and active lists;
- per slice: the list of beads on it;
- per replica: the list of active beads and a free list.

The capacities are `SM` beads per slice and `NB` beads per replica. Each replica has its own Philox counter-based RNG stream (cuRAND, sub-sequence = replica index), so runs are reproducible for a given `seed`.

**Two parallelisation modes** (runtime parameter `W`):

- **`W=1`, one thread = one replica.** Suited to few particles (traps, $N \lesssim 10$). Arrays are laid out *[element][replica]*, so neighbouring threads access neighbouring words.
- **`W=32`, one warp = one replica.** Suited to many particles. All 32 lanes execute the same control flow with *identical copies* of the replica's RNG state, so they draw the same random numbers and take the same decisions without communicating.
  - The $O(N)$ work is split over the lanes and combined with warp-shuffle reductions: the potential of a bead against the other beads of its slice, the Swap normalisations $\Sigma_{\mathcal I}$ and $\Sigma_q$ (with an online log-sum-exp per lane, then a prefix sum over lanes to select the candidate), and the measurements.
  - Every reduced value is **broadcast from lane 0**, so all lanes hold bit-identical values and never diverge in their accept/reject decisions.
  - List updates are done by all lanes with identical values, separated by `__syncwarp()` fences (read phase / write phase).
  - Arrays are laid out *[replica][element]*, so the lanes read consecutive beads of a slice (coalesced).

**Reproducibility.** With the same seed, `W=1` and `W=32` produce **bit-identical** results; this is checked by `test_warp.sh`.

**Capacity limits.** A move that would exceed `SM` or `NB` is rejected. Such configurations simply get zero weight, so detailed balance holds within a truncated configuration space. These rejections are counted and reported as `overflow`. A non-negligible count means `SM`/`NB` must be increased.

**Precision.**

- Compile with `-DUSE_FLOAT` for single precision. On GeForce GPUs FP64 runs at 1/64 of the FP32 rate.
- Accumulators are always double.
- All validation tests pass in single precision too (Section 9).

**Debugging.** `debug=1` runs a list-consistency check kernel after every chunk of `chunk` steps.

**Compiler note.** The move functions are declared `__forceinline__`. With them compiled as ABI calls, ptxas from CUDA 13.0 and 13.3 (at `-O1` and above) corrupted the local stack frame of the `W=1` kernel; compute-sanitizer reported misaligned and out-of-bounds local stores, and the kernel was correct with `-G`. Inlining avoids the problem and also shrinks the stack frame.

---

## 7. Usage

### 7.1 Building

```bash
# CPU (DIM = 1, 2 or 3)
g++ -O3 -march=native -std=c++17 -DDIM=3 worm_pimc.cpp -o worm_pimc_3d

# GPU: sm_86 = RTX 30xx / A10, sm_120 = RTX 50xx; add -DUSE_FLOAT for single precision
nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 \
     -DDIM=3 -DUSE_FLOAT worm_pimc_gpu.cu -o worm_pimc_gpu_3d_f
```

On a SLURM cluster, compile and run on compute nodes (`srun`/`sbatch`), not on the login node. Ready-made batch scripts for the demos are in `slurm/` (see `slurm/README.md`). The validation and benchmark scripts in the repository root are batch jobs too; the `sbatch` line is in each header.

### 7.2 Parameters

Every parameter is given on the command line as `key=value`. Defaults are in brackets.

| Parameter | Meaning | CPU | GPU |
|---|---|---|---|
| `P` [24] | number of time slices; $\tau = \beta/P$ | ✓ | ✓ |
| `beta` [1], `mu` [0] | inverse temperature, chemical potential | ✓ | ✓ |
| `lam` [0.5] | $\lambda = \hbar^2/2m$ | ✓ | ✓ |
| `C` [1] | G-sector weight | ✓ | ✓ |
| `Lmax` [8] | maximum segment length of the moves (at most $P-2$; on the GPU also less than `LCAP` = 64) | ✓ | ✓ |
| `Lbox` [12] | Insert region (open space) or box side (`pbc=1`) | ✓ | ✓ |
| `pbc` [0] | periodic box | ✓ | ✓ |
| `omega` [1] | trap frequency (0 = no trap) | ✓ | ✓ |
| `pot` [0], `kappa` [0], `eps` [1], `sig` [1] | pair potential (Section 1) | ✓ | ✓ |
| `rc` [∞, or $L/2$ with pbc] | pair cutoff | ✓ | ✓ |
| `N0` [0] | start from `N0` straight closed worldlines on a simple lattice | ✓ | ✓ |
| `steps` [CPU 2e7 / GPU 1e6], `therm` [2e6 / 1e5] | production / thermalisation steps (**per replica** on the GPU) | ✓ | ✓ |
| `meas` [20] | measure every `meas` steps | ✓ | ✓ |
| `blocks` [CPU 20 / GPU 32] | jackknife blocks (CPU) or replica groups (GPU) | ✓ | ✓ |
| `seed` [12345] | RNG seed | ✓ | ✓ |
| `debug` [0] | consistency checks | ✓ | ✓ |
| `W` [1] | 1 = replica per thread, 32 = replica per warp | | ✓ |
| `R` [4096] | number of replicas | | ✓ |
| `SM` [64], `NB` [`SM`·`P`] | capacity: beads per slice, beads per replica | | ✓ |
| `chunk` [1e5] | steps per kernel launch | | ✓ |
| `tpb` [128] | threads per block (multiple of 32) | | ✓ |
| `rmaj` [0 for W=1, 1 for W=32] | memory layout | | ✓ |
| `dev` [0], `stack` [driver default] | device index, per-thread stack size | | ✓ |

If a parameter is given more than once, the last occurrence wins.

### 7.3 Output

```
RESULT N= <N> +- err   Vbar= ... +- ...   Zfrac= ...   time=...s  [GPU: rate=... steps/s  overflow=...]
TIMING therm=...s prod=...s prod_rate=... steps/s
RESULT2 E= ... +- ...   D2= ... +- ...   rhos= ... +- ...
CYCLES n1= ... +- ... n2= ... +- ... ... n11= ... +- ... n12+= ... +- ...
acceptance: open=... close=... insert=... remove=... advH=... recH=... advT=... recT=... swap=... stage=...
```

`rate` counts all steps, including thermalisation. `prod_rate` counts production only, summed over all replicas on the GPU.

### 7.4 Choosing parameters

- **`C`:** aim for `Zfrac` between 0.05 and 0.5.
- **`Lmax`:** keep $\sqrt{2\lambda\tau}$ (one link) well below $L/2$. Larger values improve the efficiency of the open/close/swap moves until their acceptance drops.
- **`P`:** the primitive approximation converges as $\tau^2$. Check convergence in $P$ for interacting systems.
- **Thermalisation on the GPU:** each replica must be thermalised separately.
  - With many short replicas, an insufficient `therm` produces a bias that the replica-group error bars **cannot detect**.
  - Example: the ideal Bose gas at $T=0.5$, where the condensate occupation fluctuates strongly and slowly, gave $\langle N\rangle = 7.77(6)$ instead of 8 with `therm=5e4`; with `therm=5e5` it gave 8.02(5).
  - Check by repeating with a larger `therm` (or a different `N0`).
  - For large $N$ use fewer, longer replicas, and start from `N0` near the expected particle number.
- **`SM`:** use $`\gtrsim 1.5\,N + 32`$ and check that `overflow=0`.

---

## 8. Demos

The demos are in `demos/`. Run them from the repository root with `sbatch`; their outputs are written to `demos/out/`. The results below come from the runs on the cabib cluster (RTX 5070 Ti / RTX 3080 Ti, Xeon Gold 6330).

Each demo is a plain bash script that compiles what it needs and writes its results to `demos/out/`.

| Demo | Run on a workstation | Run on the cluster |
|---|---|---|
| 1. Trapped bosons (CPU) | `bash demos/demo_trap.sh` | `sbatch slurm/demo_trap.sbatch` |
| 2. Superfluid fraction of the ideal Bose gas (GPU, `W=1`) | `ARCH=-arch=native bash demos/demo_superfluid.sh` | `sbatch slurm/demo_superfluid.sbatch` |
| 3. Quantum Lennard-Jones equation of state (GPU, `W=32`) | `ARCH=-arch=native bash demos/demo_lj_eos.sh` | `sbatch slurm/demo_lj_eos.sbatch` |

- **All demos at once:** `bash slurm/run_all_demos.sh` submits the three demos and then `slurm/make_figures.sbatch`, which runs `docs/make_figures.py` once they finish.
- **Smaller GPUs:** set the number of replicas with `R=...`.

### 8.1 Demo 1 — bosons in a harmonic trap (CPU)

Two systems, each compared with the exact primitive-approximation result of the *same* discretisation (`exact_pa.py`): $\omega=1$ with harmonic pair coupling $\kappa=0.3$, at $\beta=1$, $P=24$. Each case runs 4 independent seeds of $10^8$ steps in parallel, and the error is the standard error over the seeds. Output (`demos/out/demo_trap.txt`):

```
d=1 beta=1 mu=0 kappa=0.3   (4 seeds x 1e8 steps)
  N     exact = 1.23924   MC = 1.23820 +- 0.00176   (-0.6 sigma)
  Vbar  exact = 0.59909   MC = 0.59748 +- 0.00098   (-1.6 sigma)
  E     exact = 1.19819   MC = 1.19468 +- 0.00393   (-0.9 sigma)
d=2 beta=1 mu=-0.3 kappa=0.3   (4 seeds x 1e8 steps)
  N     exact = 0.60910   MC = 0.60846 +- 0.00152   (-0.4 sigma)
  Vbar  exact = 0.63804   MC = 0.63827 +- 0.00187   (+0.1 sigma)
  E     exact = 1.27609   MC = 1.27614 +- 0.00576   (+0.0 sigma)
```

A separate run of the 1D case with 8 seeds gives N = 1.23982(117), V̄ = 0.59938(91), E = 1.19708(231): +0.5σ, +0.3σ and −0.5σ. The scatter between seeds agrees with the error bars reported by the program.

### 8.2 Demo 2 — superfluid fraction of the ideal Bose gas (GPU)

**Setup.** Ideal bosons ($\lambda=1/2$) in a periodic cubic box $L=4$. At each temperature the chemical potential is fixed by the exact solution so that $\langle N\rangle = 8$ (density $\rho = 1/8$; the bulk condensation temperature is $T_c = 4\pi\lambda(\rho/\zeta(3/2))^{2/3} \approx 0.83$). In this finite box the superfluid fraction rises smoothly from 0 to 1 across $T_c$. It is measured with the winding-number estimator and compared with the exact cycle expansion (`exact_ideal.py`).

**Run parameters.** GPU float, `W=1`, 8192 replicas × $2\times10^5$ steps after $5\times10^5$ thermalisation steps, starting from $N_0=8$; $P=32$, $C=10^{-5}$, `SM=256` (no overflows).

![Superfluid fraction and energy of the ideal Bose gas](figures/superfluid.png)

| T | μ (exact, ⟨N⟩=8) | N | E (exact) | E (MC) | ρs/ρ (exact) | ρs/ρ (MC) |
|---|---|---|---|---|---|---|
| 3.333 | −4.1451 | 7.9960(88) | 37.694 | 37.648(52) | 0.00027 | 0.00029(2) |
| 2.000 | −1.2337 | 7.9925(94) | 20.786 | 20.758(32) | 0.02008 | 0.02027(21) |
| 1.333 | −0.3770 | 8.0099(129) | 10.925 | 10.924(26) | 0.19399 | 0.19403(61) |
| 1.000 | −0.1852 | 8.0029(210) | 5.765 | 5.749(17) | 0.44131 | 0.44101(111) |
| 0.800 | −0.1220 | 7.9792(234) | 3.253 | 3.250(13) | 0.61672 | 0.61401(123) |
| 0.667 | −0.0921 | 8.0000(285) | 1.956 | 1.967(10) | 0.73044 | 0.73105(128) |
| 0.500 | −0.0631 | 8.0681(325) | 0.805 | 0.799(8) | 0.85761 | 0.85777(104) |
| 0.333 | −0.0399 | 8.0162(375) | 0.182 | 0.186(4) | 0.95351 | 0.95435(87) |

All points agree with the exact result within about 2σ; the error bars are smaller than the markers in the figure.

**Thermalisation lesson.** The first version of this demo used `therm=5e4` and started from an empty box. Its low-temperature points were biased low ($\langle N\rangle = 7.87(4)$ at $T=0.5$). Near and below $T_c$ the condensate occupation fluctuates strongly and slowly, and every replica must thermalise on its own (Section 7.4).

### 8.3 Demo 3 — equation of state of a quantum Lennard-Jones fluid (GPU, replica-per-warp)

**Setup.** Lennard-Jones bosons with quantum parameter $\lambda = \hbar^2/(2m\sigma^2\varepsilon) = 0.05$ (for comparison, ⁴He with LJ parameters σ = 2.556 Å, ε = 10.22 K has ≈ 0.09, and Ne ≈ 0.004) at $T = \varepsilon/k_B$, $P=32$, in a periodic box $L=4.5\sigma$ ($r_c = L/2$). The chemical potential is scanned across the gas–liquid crossover.

**Equilibration check.** Every point is run from two very different initial states, lattices of $N_0=8$ and $N_0=64$ particles. The script stops if the two results differ by more than 3σ.

**Run parameters.** GPU float, replica-per-warp mode (`W=32`), 256 replicas × $10^6$ steps after $4\times10^6$ thermalisation steps, $C=10^{-4}$ (Zfrac 0.33–0.79). About 19 minutes in total on an RTX 5070 Ti.

![Equation of state of the quantum Lennard-Jones fluid](figures/lj_eos.png)

| μ/ε | ⟨N⟩ from N₀=8 | ⟨N⟩ from N₀=64 | difference | ρσ³ | E/(Nε) |
|---|---|---|---|---|---|
| −8 | 0.0614(10) | 0.0616(10) | −0.1σ | 0.000675(8) | 1.49(3) |
| −6 | 0.4628(36) | 0.4633(36) | −0.1σ | 0.00508(3) | 1.481(13) |
| −4 | 4.140(25) | 4.131(21) | +0.3σ | 0.0454(2) | 1.299(8) |
| −3 | 17.939(86) | 17.948(99) | −0.1σ | 0.1969(7) | 0.706(4) |
| −2 | 40.095(50) | 40.085(49) | +0.2σ | 0.4399(4) | −0.039(3) |
| −1 | 48.747(49) | 48.667(44) | +1.2σ | 0.5345(4) | −0.243(3) |
| 0 | 53.914(46) | 53.879(40) | +0.6σ | 0.5915(3) | −0.311(3) |
| 1 | 57.680(46) | 57.664(55) | +0.2σ | 0.6329(4) | −0.316(3) |
| 2 | 60.550(61) | 60.606(55) | −0.7σ | 0.6648(5) | −0.286(3) |

**Reading the results.**

- **Dilute gas.** The energy per particle tends to $\frac32 k_BT$, the classical ideal-gas value: $E/N = 1.49(3)$ at $\mu=-8$.
- **Crossover.** Between $\mu\approx-4$ and $-2$ the density jumps by an order of magnitude. This is the rounded gas–liquid transition of a small box.
- **Liquid.** The liquid branch reaches $\rho\sigma^3 \approx 0.66$ with negative, nearly constant energy per particle.
- **Statistics.** All 18 runs are free of capacity overflows.

**Thermalisation lesson.** A first version of this demo used $C=10^{-2}$, which gave Zfrac ≈ 0.005, and $4\times10^5$ thermalisation steps, starting only from $N_0=64$. Near the crossover it was far from equilibrium: at $\mu=-3$ it gave $\langle N\rangle = 22.7$ instead of 17.9. The two-start check is what exposed it.

---

## 9. Validation

Every test compares against an exact result for the *same* discretisation, so any deviation beyond statistics is a bug. Numbers in parentheses are one standard error in the last digits.

### 9.1 Harmonic trap (exact primitive approximation, `exact_pa.py`)

GPU, `W=1`, $P=24$ ($P=16$ for $\beta=2$), $C=10^{-3}$, $L_{\max}=8$, $2\times10^5$ steps per replica after $5\times10^4$ thermalisation steps (`validate_gpu.sh`).

| case | | exact | GPU double (5070 Ti, R=16384) | GPU float (3080 Ti, R=32768) |
|---|---|---|---|---|
| 1D, β=1, μ=0, κ=0 | N | 1.96720 | 1.96895(162) | 1.96582(97) |
| | V̄ | 0.81199 | 0.81254(74) | 0.81219(35) |
| | E | 1.62398 | 1.62305(212) | 1.62418(115) |
| 1D, β=1, μ=0, κ=0.3 | N | 1.23924 | 1.23943(64) | 1.23929(54) |
| | V̄ | 0.59909 | 0.59904(42) | 0.59886(27) |
| | E | 1.19819 | 1.19909(91) | 1.19786(84) |
| 1D, β=2, μ=−0.5, κ=0 | N | 0.17821 | 0.17823(9) | 0.17821(8) |
| | V̄ | 0.05690 | 0.05693(3) | 0.05691(3) |
| | E | 0.11380 | 0.11380(9) | 0.11384(7) |
| 2D, β=1, μ=0, κ=0 | N | 1.18680 | 1.18734(137) | 1.18603(76) |
| | V̄ | 1.16075 | 1.16103(137) | 1.16102(85) |
| | E | 2.32149 | 2.32306(368) | 2.32518(232) |
| 2D, β=1, μ=−0.3, κ=0.3 | N | 0.60910 | 0.60899(43) | 0.60927(34) |
| | V̄ | 0.63804 | 0.63795(51) | 0.63853(31) |
| | E | 1.27609 | 1.27725(136) | 1.27616(90) |

The CPU code reproduces the same cases. With $3\times10^8$ steps and two seeds, it gives for example E = 1.6275(47) and 1.6346(66) for the first case, and 1.2730(51) and 1.2760(54) for the last one (`validate_pbc.sh`).

### 9.2 Ideal Bose gas in a periodic box (`exact_ideal.py`)

$P=24$, $\lambda=1/2$; $C=10^{-3}$ ($10^{-4}$ for the last case).

| case | | exact | GPU double | GPU float | CPU ($3\times10^8$ steps) |
|---|---|---|---|---|---|
| 1D, β=1, μ=−0.2, L=4 | N | 5.15466 | 5.16344(672) | 5.15210(603) | 5.18519(2684) |
| | E | 0.83120 | 0.82920(331) | 0.83666(226) | 0.84297(979) |
| | ρs/ρ | 0.58353 | 0.58352(87) | 0.58308(65) | 0.58672(274) |
| 2D, β=1, μ=−0.3, L=4 | N | 4.26085 | 4.26063(400) | 4.26307(312) | 4.26419(1167) |
| | E | 2.20385 | 2.20934(446) | 2.19731(391) | 2.19262(1740) |
| | ρs/ρ | 0.38456 | 0.38409(51) | 0.38518(50) | 0.38553(197) |
| 3D, β=1, μ=−0.5, L=4 | N | 3.66741 | 3.66985(303) | 3.66685(255) | 3.66968(937) |
| | E | 4.03538 | 4.04762(727) | 4.02361(697) | 4.05700(2615) |
| | ρs/ρ | 0.18730 | 0.18838(63) | 0.18762(33) | 0.18695(227) |
| 3D, β=2, μ=−0.1, L=3 | N | 4.57995 | 4.57219(889) | 4.57853(413) | 4.57868(2127) |
| | E | 0.14221 | 0.13884(230) | 0.14245(136) | 0.13567(787) |
| | ρs/ρ | 0.95819 | 0.95834(47) | 0.95832(24) | 0.95909(155) |

Of the 36 Monte Carlo numbers above, only one deviates by more than 2σ (2.4σ, float 1D energy), which is what statistical fluctuations predict. All 30 GPU numbers in the trap table lie within 1.6σ.

- **Overflows.** In the 1D case and in the 3D β=2 case, a few hundred to ~1500 moves (out of about $4\times10^9$) were rejected for exceeding `SM`=64. This samples a truncated ensemble, and the effect is negligible here.
- **The image-sum fix.** The last case is the one that exposed the nearest-image bug (Section 3.2): before the fix the CPU gave E = 1.69 and ρs/ρ = 0.81.

### 9.3 Trotter scan (`validation/trotter.sh`)

This test checks the discretisation itself. The setup is the trap with harmonic coupling ($\omega=1$, $\kappa=0.3$, $\beta=2$) at $P = 4, 6, 8, 12, 16, 24, 32, 48, 64$, so that $\tau$ runs from 0.5 to 0.031.

- **Runs.** GPU `W=1`, double precision, $R = 32768$ replicas × $4\times10^5$ steps after $2\times10^5$ thermalisation steps, $L_{\max} = \min(P-2, 8)$.
- **G-sector weight.** $C$ is scaled as $(16/P)^2$ so that Zfrac stays roughly constant. With a fixed $C$, Zfrac fell to ~0.007 at $P=64$ and the replicas were under-thermalised.
- **Two references.** At every $P$ the Monte Carlo is compared with the exact PA result at the *same* $P$ (a strict test). The $\tau\to0$ extrapolation of the Monte Carlo is compared with the exact continuum result.

| d | μ | C at P=16 | MC vs exact PA, N at all 9 P: χ²/dof (p) | largest deviation (N, V̄, E and n₁…n₆₊, all P) | N at τ→0: MC / exact | E at τ→0: MC / exact |
|---|---|---|---|---|---|---|
| 1 | 0.5 | 3·10⁻³ | 12.0/9 (0.21) | −2.6σ (n₅, P=4) | 1.63239(32) / 1.63284 | 1.12858(30) / 1.12874 |
| 2 | 1.5 | 3·10⁻⁴ | 12.0/9 (0.22) | +2.2σ (V̄, P=8) | 2.76198(45) / 2.76133 | 3.92677(77) / 3.92561 |
| 3 | 2.0 | 10⁻⁴ | 17.6/9 (0.04) | +2.4σ (n₂, P=8) | 2.00425(34) / 2.00506 | 4.10473(84) / 4.10512 |

The τ→0 columns come from a weighted fit $a + b\tau^2$ to the points with $\tau \le 1/6$ ($P \ge 12$).

![Trotter scan](figures/val_trotter.png)

- **The PA is resolved exactly.** At $P=4$ the Trotter error of $\langle N\rangle$ is 0.07–0.18, i.e. several hundred standard errors, and the Monte Carlo still agrees with the PA at the same $P$. Across all $P$ and observables (243 numbers) no deviation exceeds 2.6σ.
- **The τ² law and the continuum limit.** The extrapolations agree with the exact continuum within 1.6σ, except $\langle N\rangle$ in 3D (−2.3σ).
- **The 3D case.** The 3D scan has p = 0.04 for N, and its extrapolated N is 2.3σ low. Both are borderline. A second independent 3D scan (`TAG=_run2 SEEDOFF=1000`) has been prepared to decide whether this is a fluctuation; it has not been run yet.

### 9.4 Exchange cycles of the ideal gas (`validation/cycles_ideal.sh`)

This test checks the permutation sampling directly: the full distribution $\langle n_j\rangle$ against the exact cycle expansion (Section 5.2).

- **Runs.** GPU `W=1` in double and float, $P=24$, $\lambda=1/2$, $L_{\max}=8$, $R=16384$ replicas × $4\times10^5$ steps after $2\times10^5$ thermalisation steps.
- **Cases.** The four cases include one close to condensation: 3D, $\beta=1.25$, $L=4$, $\langle N\rangle = 8$, where 14% of the particles are in cycles longer than 11.
- **Test statistic.** Each χ² is over 15 numbers: N, E, ρs/ρ and the 12 bins $n_1 \ldots n_{\ge12}$.

| case | C | Zfrac | double: χ²/dof (p), max dev. | float: χ²/dof (p), max dev. |
|---|---|---|---|---|
| 1D, β=1, μ=−0.2, L=4 | 10⁻³ | 0.080 | 18.0/15 (0.26), 1.8σ | 15.0/15 (0.45), 1.9σ |
| 2D, β=1, μ=−0.3, L=4 | 10⁻³ | 0.032 | 19.0/15 (0.21), 2.9σ | 15.6/15 (0.41), 2.4σ |
| 3D, β=2, μ=−0.1, L=3 | 10⁻⁴ | 0.114 | 12.6/15 (0.64), 1.9σ | 10.9/15 (0.76), 1.5σ |
| 3D, β=1.25, μ=−0.122, L=4 | 10⁻⁵ | 0.293 | 12.6/15 (0.63), 1.6σ | 13.2/15 (0.59), 1.7σ |

No overflows occurred in any of these runs. Of the 120 numbers, four lie beyond 2σ (all in the 2D case, at most 2.9σ), against about six expected for Gaussian errors.

![Exchange-cycle occupations](figures/val_cycles.png)

The table with every bin is printed by `python3 validation/analyze.py cycles`.

### 9.5 Interacting systems: GPU versus CPU

No exact result exists for an interacting many-body system, so here the GPU code is compared with the independent CPU implementation.

**Small system (N ≈ 21).** 3D Lennard-Jones: $\lambda=0.05$, $T=1$, $P=32$, $\mu=0$, $L=3.379\sigma$, $r_c=L/2$, start from $N_0=27$. The CPU reference is 12 independent seeds × $1.1\times10^8$ steps (`ref_lj_cpu.sh`). The GPU runs use `W=32` with 2048 replicas × $2\times10^6$ steps (`lj_gpu.sh`, `lj_float.sh`).

| | CPU | GPU W=32 double | GPU W=32 float |
|---|---|---|---|
| N | 21.1214(88) | 21.1151(50) | 21.1167(65) |
| V̄ | −51.742(47) | −51.710(26) | −51.724(33) |
| E | 2.951(58) | 2.973(27) | 2.953(21) |

**N ≈ 100 (`validation/lj100_*.sh`).** Both systems are 3D Lennard-Jones ($\varepsilon = \sigma = 1$, $r_c = 2.5\sigma$) at $T=1$ with $P=32$:

| system | λ | μ | L | ρσ³ | ρs/ρ | C | SM |
|---|---|---|---|---|---|---|---|
| A: dense liquid, no exchange | 0.05 | 1 | 5.4σ | 0.64 | 0 | 10⁻⁴ | 192 |
| B: degenerate gas (thermal wavelength 2.5σ), superfluid crossover | 0.5 | 3 | 8σ | 0.18 | 0.41 | 10⁻⁷ | 224 |

- **CPU reference.** 56 independent seeds × $2\times10^8$ steps after $2\times10^7$ (A) or $4\times10^7$ (B) thermalisation steps. Half of the seeds start from an $N_0=100$ lattice, half from $N_0=64$. The error is the standard error over seeds.
- **GPU runs.** `W=32`, 512 replicas × $10^7$ steps after $10^7$ thermalisation steps, in double and in float, each from $N_0=100$ and from $N_0=64$.
- **Equilibration check.** The two starts agree in every column (`analyze.py lj100` prints them separately).

The table combines the two starts of each precision (inverse-variance mean).

| | A: CPU | A: GPU double | A: GPU float | B: CPU | B: GPU double | B: GPU float |
|---|---|---|---|---|---|---|
| N | 101.433(13) | 101.413(13) | 101.394(13) | 90.590(11) | 90.583(14) | 90.593(13) |
| V̄ | −353.827(91) | −353.701(95) | −353.600(95) | −47.188(12) | −47.181(16) | −47.185(15) |
| E | −45.583(77) | −45.583(67) | −45.627(72) | 140.721(92) | 140.538(88) | 140.850(87) |
| ρs/ρ | 0 | 0 | 0 | 0.41232(85) | 0.41471(80) | 0.41231(71) |
| n₁ | = N | = N | = N | 63.766(26) | 63.706(23) | 63.783(24) |
| n₂ | 0 | 0 | 0 | 3.8507(73) | 3.8397(62) | 3.8373(68) |
| n₍≥12₎ | 0 | 0 | 0 | 12.901(31) | 12.987(24) | 12.927(29) |

![GPU versus CPU at N ≈ 100](figures/val_lj100.png)

- **System A** is a dense liquid at $\lambda = 0.05$ with no exchange. All numbers agree within 2.1σ (float N). Note that V̄ is shifted by +1.0σ (double) and +1.7σ (float), in the same direction in both precisions.
- **System B, float.** All 16 numbers agree with the CPU: χ²/dof = 1.1/4 for N, V̄, E, ρs and 8.3/12 for the cycle bins.
- **System B, double.** χ²/dof is 6.5/4 for N, V̄, E, ρs and 15.7/12 for the cycle bins. The largest deviations are ρs/ρ (+2.0σ) and $n_{\ge12}$ (+2.2σ). Both come from the $N_0=100$ run (+2.3σ and +2.6σ); the $N_0=64$ run agrees (χ² = 2.4/4).
- **What this means.** Long cycles and the winding are the slowest observables, so an under-thermalised run or a statistical fluctuation are the most likely explanations. Float, which uses the same code path, agrees. Two more independent double and float runs of B (`TAG=_s1`, `_s2`) have been prepared to settle it.

### 9.6 Internal consistency

`test_warp.sh` checks the GPU code itself:

- Same-seed runs of `W=1`, `W=32` and `W=32` with the other memory layout give bit-identical results, in the trap (2D, κ=0.3), the 3D ideal gas and 3D Lennard-Jones (N≈27).
- The list-consistency checks (`debug=1`) pass.
- `compute-sanitizer` reports 0 errors in both modes.

### 9.7 Reproducing the validation

The scripts in `validation/` write their raw output to `validation/out/`. Run them from the repository root:

```bash
sbatch -p gpu --gres=gpu:1 validation/trotter.sh 1 0.5 3e-3      # also: 2 1.5 3e-4 and 3 2.0 1e-4
sbatch -p gpu --gres=gpu:1 validation/cycles_ideal.sh
sbatch -p cpu -N1 --exclusive validation/lj100_cpu.sh A           # B: add --export=ALL,THERM=4e7,MEAS=100
sbatch -p gpu --gres=gpu:1 validation/lj100_gpu.sh A double       # A|B, double|float
python3 validation/analyze.py      # summary tables (markdown); trotter | cycles | lj100
python3 validation/figures.py      # docs/figures/val_*.png
```

- **Extra runs.** `TAG` (a suffix of the output file) and `SEEDOFF` (a seed offset) add independent runs. `analyze.py` combines the extra GPU runs of `lj100_gpu.sh` with the others, and reports an extra Trotter scan as a separate table.
- **Run times.** One Trotter scan takes 1–2 h on one GPU, and one $N\approx100$ GPU run takes 0.4–3 h, depending on the card and the precision. The CPU reference takes a 56-core node for about 6 h (A) or 3 h (B).

---

## 10. Performance

**Metric:** total Monte Carlo steps per second, summed over replicas on the GPU. A step is one attempted move.

**Why steps/s is a fair comparison:**

- The GPU replicas sample the same Markov chain as the CPU code. In the long-run limit, error² × time scales with steps/s.
- The rates are measured during production only (`TIMING prod_rate`), after every replica has been thermalised.
- The overhead that steps/s does *not* capture is thermalisation: each GPU replica needs its own. It is amortised by long production runs.

### 10.1 Many particles: 3D Lennard-Jones, `W=32`

Setup: $\rho\sigma^3=0.7$, $T=1$, $\lambda=0.05$, $P=32$, $r_c=\min(2.5\sigma, L/2)$, start from an $N_0$ lattice, $2\times10^4$ thermalisation steps per replica. Scripts: `bench_N2.sh` (GPU, R=1024) and `bench_cpu_N.sh` (one core of a Xeon Gold 6330, `-O3 -march=native`).

| N | CPU, 1 core | RTX 5070 Ti, double | RTX 5070 Ti, float | float / 1 core | double / 1 core |
|---|---|---|---|---|---|
| 8 | 2.0e6 | 7.4e6 | 5.3e7 | 27× | 3.8× |
| 27 | 4.2e5 | 6.3e6 | 4.8e7 | 115× | 15× |
| 64 | 1.6e5 | 5.1e6 | 3.3e7 | 204× | 32× |
| 125 | 8.8e4 | 3.2e6 | 2.0e7 | 223× | 37× |
| 216 | 6.2e4 | 1.8e6 | 9.5e6 | 153× | 29× |
| 512 | 3.0e4 | 8.2e5 | 2.4e6 | 79× | 27× |

![Throughput versus N](figures/performance.png)

- **Mode choice.** For $N \gtrsim 10$, `W=32` is much faster than `W=1`: for example at $N=27$ in double, 6.3e6 against 1.1e6 steps/s.
- **Scaling.** The rate falls roughly as $1/N$ for large $N$ because the cost per move is $O(N)$. Cell lists are the planned remedy.
- **Node equivalence.** A Gold 6330 node has 28 cores, so one RTX 5070 Ti in float is worth about 7–8 such nodes at $N\approx 64$–125.

### 10.2 Few particles: harmonic trap, `W=1`

Setup: $P=24$, $C=10^{-3}$, total steps/s including thermalisation, saturated at $R \gtrsim 32768$ replicas (`bench_gpu.sh`, `bench_cpu.sh`).

| system | CPU, 1 core (Gold 6330) | RTX 3080 Ti double / float | RTX 5070 Ti double / float |
|---|---|---|---|
| 1D, κ=0 (N≈2) | 6.5e6 | 6.7e7 / 1.7e8 | 8.2e7 / 2.0e8 |
| 2D, κ=0.3 (N≈0.6) | 6.7e6 | 5.1e7 / — | 8.3e7 / 3.2e8 |

Throughput saturates at about 32k replicas.

### 10.3 Hardware notes

- **FP64.** GeForce GPUs execute FP64 at 1/64 of the FP32 rate, hence the 3–7× gain of `-DUSE_FLOAT`. Single precision passes every validation test above.
- **Memory.** Device memory per replica is about $`NB\,(d\cdot{\rm sizeof(real)} + 28) + 4P(SM+1)`$ bytes. The program checks the free memory before allocating, and refuses to start if it does not fit.
- **Shared GPUs.** On shared nodes, other processes may hold device memory.

---

## 11. Limitations and outlook

- **Discretisation:** the primitive approximation has an $O(\tau^2)$ error. Higher-order actions (pair-product, Chin, Takahashi–Imada) would allow much larger $\tau$ for strongly repulsive cores such as ⁴He.
- **Potentials:** they are truncated without tail corrections. The Aziz potential is implemented but has not yet been validated against published ⁴He results.
- **Large $N$:** the cost of a move grows as $O(N)$ (all beads of a slice). Cell lists per slice would make it $O(1)$ for short-range potentials; this is the next step for $N \gtrsim 500$.
- **Thermalisation:** with many replicas, thermalisation is paid once per replica. Long production runs with few replicas amortise it.
- **Capacity:** capacities are fixed at launch (`SM`, `NB`), and `LCAP` = 64 bounds `Lmax` at compile time.

---

## 12. References

1. M. Boninsegni, N. Prokof'ev and B. Svistunov, *Worm algorithm for continuous-space path integral Monte Carlo simulations*, Phys. Rev. Lett. **96**, 070601 (2006).
2. M. Boninsegni, N. V. Prokof'ev and B. V. Svistunov, *Worm algorithm and diagrammatic Monte Carlo: A new approach to continuous-space path integral Monte Carlo simulations*, Phys. Rev. E **74**, 036701 (2006).
3. D. M. Ceperley, *Path integrals in the theory of condensed helium*, Rev. Mod. Phys. **67**, 279 (1995).
4. E. L. Pollock and D. M. Ceperley, *Path-integral computation of superfluid densities*, Phys. Rev. B **36**, 8343 (1987).
5. R. A. Aziz, V. P. S. Nain, J. S. Carley, W. L. Taylor and G. T. McConville, *An accurate intermolecular potential for helium*, J. Chem. Phys. **70**, 4330 (1979).
6. E. L. Pollock and D. M. Ceperley, *Simulation of quantum many-body systems by path-integral methods*, Phys. Rev. B **30**, 2555 (1984).
