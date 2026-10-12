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

## Plan GPU (CUDA) — NO implementado ni probado (sin nvcc/GPU en el entorno)
Una cadena worm es secuencial, así que la paralelización natural es:
1. **Réplicas independientes**: un bloque (o warp) por réplica, estado en memoria compartida/global (posiciones SoA, listas prev/next), RNG Philox por réplica. Escala casi linealmente; promedio al final.
2. **Dentro de la réplica** (N grande): reducciones paralelas para energía de pares al insertar/mover segmentos y para las sumas de candidatos del Swap; staging sobre slices en paralelo por hilos.
3. Para interacciones de pares a N grande, celdas/Verlet por slice.
Pasos sugeridos: portar `Sim` a struct de arrays, kernel `run_replica`, tests contra `validate.sh`.
