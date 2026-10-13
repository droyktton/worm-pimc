# Worm algorithm PIMC (bosones en el continuo) — versión mínima validada

Basado en Boninsegni, Prokof'ev, Svistunov, PRL 96, 070601 (2006). ħ=m=1.
Gran canónico, aproximación primitiva con P slices, sectores Z/G, movimientos:
Open/Close, Insert/Remove, Advance/Recede (head y tail), Swap, Staging.

## Uso
    g++ -O2 -std=c++17 -DDIM=1 worm_pimc.cpp -o worm_pimc_1d    # DIM=1,2,3
    ./worm_pimc_1d P=24 beta=1 mu=0 C=1e-3 Lmax=8 omega=1 kappa=0 steps=3e7 therm=3e6
    ./validate.sh      # compara con soluciones exactas
Parámetros: P, beta, mu, C, Lmax (<= P-2), Lbox (caja para Insert), omega (trampa),
kappa (acoplamiento armónico de a pares), steps, therm, meas, blocks, debug=1 (chequeo de consistencia), seed.
Salida: N, Vbar (energía potencial media), Zfrac (fracción de pasos en sector Z), aceptancias.

## Validación (exact_pa.py)
Bosones en trampa armónica ± acoplamiento armónico de pares: Z_N exacta (suma sobre
tipos de ciclo) para la MISMA discretización primitiva (sin error de Trotter) y para el continuo.
Resultados (3e7 pasos, C=1e-3; exacto PA vs MC):
| caso | N exacto | N MC | Vbar exacto | Vbar MC |
|---|---|---|---|---|
| 1D β=1 μ=0 κ=0 | 1.9672 | 1.996(21) | 0.8120 | 0.825(8) |
| 1D κ=0.3 | 1.2392 | 1.2400(63) | 0.5991 | 0.6041(34) |
| 1D β=2 μ=-0.5 | 0.1782 | 0.1785(11) | 0.05690 | 0.05690(39) |
| 2D κ=0 | 1.1868 | 1.180(12) | 1.1608 | 1.151(11) |
| 2D μ=-0.3 κ=0.3 | 0.6091 | 0.6078(54) | 0.6380 | 0.6360(56) |
Los resultados no dependen de C (probado C=0.1…0.001); C solo afecta la eficiencia (apuntar a Zfrac ~ 0.1–0.5).
Limitación: el test es en trampa; no incluye interacciones de contacto/Lennard-Jones (agregar potencial en U()).

## Opciones agregadas (CPU y GPU)
`pbc=1` caja periódica de lado `Lbox` (imagen mínima en links y pares; Open/Close/Swap usan el
propagador libre sumado sobre imágenes y eligen la imagen ∝ G0; Staging conserva el desplazamiento
real del segmento, i.e. el winding), `lam` (ħ²/2m), `pot=0` armónico de pares `kappa` / `1` LJ (`eps`,`sig`)
/ `2` Aziz HFDHE2 para ⁴He (K, Å; `lam=6.0596`), cutoff `rc` (≤ Lbox/2), `N0` (arranque con N0 líneas de
mundo cerradas en red). Observables nuevos: energía termodinámica `E`, winding `D2` y `rhos` = <D²>/(2λdβ<N>).
`exact_ideal.py`: gas de Bose ideal en caja periódica (N, E, D², ρs exactos; la PA es exacta para partículas libres).

## GPU (CUDA): `worm_pimc_gpu.cu`
    nvcc -O3 -std=c++17 -arch=sm_86 -DDIM=3 [-DUSE_FLOAT] worm_pimc_gpu.cu -o worm_pimc_gpu_3d
    ./worm_pimc_gpu_3d W=32 R=1024 steps=1e5 therm=2e4 ...   # mismos parámetros que la CPU
Mismos movimientos y aceptaciones que la CPU; réplicas independientes (`R`, `steps`/`therm` son POR réplica).
- `W=1`: un hilo = una réplica (pocas partículas, N ≲ 10).
- `W=32`: un warp = una réplica; las sumas O(N) (U de una cuenta contra su slice, normalizaciones del Swap,
  medición) se reparten entre lanes y se reducen con shuffles (resultado broadcast desde el lane 0, así todos
  los lanes toman decisiones idénticas con copias idénticas del RNG).
Con la misma semilla W=1 y W=32 dan resultados bit a bit idénticos. Errores: jackknife sobre grupos de réplicas.
Capacidad fija `SM` (cuentas por slice) y `NB` (por réplica): un movimiento que la excede se rechaza
(ensamble truncado; se reporta `overflow`). `debug=1` chequea consistencia de listas entre chunks.
Nota: los movimientos son `__forceinline__` a propósito (ptxas 13.0/13.3 corrompía el frame local de W=1).

Validación (scripts `validate_gpu.sh`, `validate_pbc.sh`, `ref_lj_cpu.sh` + `lj_gpu.sh`/`lj_float.sh`):
trampa (N, Vbar, E) y gas ideal periódico 1D/2D/3D (N, E, D², ρs) contra exacto, en double y float,
con errores ~5–10× menores que los tests CPU; LJ 3D (N≈21) GPU vs CPU dentro de 1σ (double y float).

Rendimiento (LJ 3D ρσ³=0.7, T=1, P=32, producción termalizada; pasos MC/s totales; `bench_N2.sh`, `bench_cpu_N.sh`):
| N | 1 core Xeon Gold 6330 | 5070 Ti W=32 double | 5070 Ti W=32 float | float vs 1 core |
|---|---|---|---|---|
| 8 | 2.0e6 | 7.4e6 | 5.3e7 | 27× |
| 27 | 4.2e5 | 6.3e6 | 4.8e7 | 115× |
| 64 | 1.6e5 | 5.1e6 | 3.3e7 | 204× |
| 125 | 8.8e4 | 3.2e6 | 2.0e7 | 223× |
| 216 | 6.2e4 | 1.8e6 | 9.5e6 | 153× |
| 512 | 3.0e4 | 8.2e5 | 2.4e6 | 79× |
Trampa (N≈1–2, `bench_gpu.sh`): W=1 float ~2–3e8 pasos/s vs 6.5e6 en 1 core.
FP64 en GeForce es 1/64 de FP32: usar `-DUSE_FLOAT` (validado) salvo que haga falta double.
Pendiente: listas de celdas para N grande (la tasa cae ~1/N), y costo de termalizar cada réplica.
