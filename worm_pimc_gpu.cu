// worm_pimc_gpu.cu -- GPU (CUDA) version of worm_pimc.cpp.
//
// Two parallelisation modes (same moves / acceptance ratios as the CPU code):
//  W=1  (few particles): one CUDA thread = one independent Markov chain
//       (replica).  Arrays laid out [element][replica] so that neighbouring
//       threads touch neighbouring words.
//  W=32 (many particles): one warp = one replica.  All lanes run the same
//       control flow with identical copies of the RNG (hence identical
//       decisions); the O(N) sums (potential of a bead against its slice,
//       Swap normalisations, measurements) are split over the lanes and
//       reduced with shuffles, the result broadcast from lane 0 so that every
//       lane holds bit-identical values.  All lanes perform the same (identical)
//       stores; list updates are fenced with __syncwarp.  Layout
//       [replica][element] so that lanes read consecutive beads of a slice.
//
// Averages: per-replica Z-sector sums of the observables (N, Vbar, energy E,
// winding D^2, exchange-cycle histogram n_j, N^2) are combined as ratio estimators; error bars from a jackknife
// over groups of replicas.  Same options as the CPU code (pbc, pot, lam, disorder Vd/qc/dseed, ...);
// the disorder coefficients are generated on the host exactly as in the CPU code.
//
// Build:  nvcc -O3 -std=c++17 -arch=sm_86 -DDIM=1 worm_pimc_gpu.cu -o worm_pimc_gpu_1d
//         (-DUSE_FLOAT for single precision)
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <curand_kernel.h>

#ifndef DIM
#define DIM 1
#endif
#ifndef LCAP
#define LCAP 64            // compile-time bound on Lmax+1 (local scratch arrays)
#endif
#ifndef NCYC
#define NCYC 12            // exchange-cycle-length bins (last one: j >= NCYC)
#endif
#ifndef DMAX
#define DMAX 512           // maximum number of disorder modes
#endif

#ifdef USE_FLOAT
typedef float real;
#define RLOG logf
#define REXP expf
#define RSQRT sqrtf
#define RFABS fabsf
#define RRINT ::rintf
#define RSINCOS sincosf
#else
typedef double real;
#define RLOG log
#define REXP exp
#define RSQRT sqrt
#define RFABS fabs
#define RRINT ::rint
#define RSINCOS sincos
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(3); } } while (0)

static const int NMOVE = 10;
enum { O_N, O_V, O_E, O_D2, O_C, O_N2 = O_C + NCYC, NOBS };
#define NIMG 2

struct Params {
    int P, Lmax, R, NB, SM;          // R replicas, NB beads/replica, SM beads/slice
    real tau, mu, C, Lbox, Vbox, omega, kappa, lam;
    real logC, logVbox, logP, logLmax;
    int pbc, pot;                    // periodic box; 0 harmonic pair, 1 LJ, 2 Aziz
    int rmaj;                        // layout: 0 [elem][replica], 1 [replica][elem]
    real eps, sig, rc2;
    int ndis; real kdis;             // disorder modes, 2 pi / Lbox
};

__constant__ real dcoef[2 * DMAX];   // disorder c_k = (Re, Im), k = 1..ndis

// device state (configuration arrays indexed by Rep::I and friends; statistics [k * R + rep])
struct State {
    real* r;                          // [(bead*DIM + d)]
    int *slice, *prev, *next, *posS, *posA;
    int* sl;                          // [(s*SM + k)]
    int* nsl;                         // [s]
    int* act;                         // [k]
    int* freeL;                       // [k]
    int *nact, *nfree, *nB, *head, *tail, *worm;
    curandStatePhilox4_32_10_t* rng;
    // statistics
    double* sumO;                     // [obs]
    long long *nZ, *nAll, *overflow;
    long long *att, *acc;             // [move]
};

#define FULL 0xffffffffu

// NOTE: the move functions are __forceinline__ on purpose: with them compiled
// as ABI calls, ptxas (CUDA 13.0 and 13.3, -O1 and above) corrupted the local
// frame of the W=1 kernel (misaligned / out-of-bounds local stores reported by
// compute-sanitizer; fine with -G).  Inlining also lowers the stack frame.

template <int W>
struct Rep {
    const Params& p;
    const State& S;
    int me, lane;
    curandStatePhilox4_32_10_t st;
    int nact, nfree, nB, head, tail, worm;
    long long att[NMOVE], acc[NMOVE], overflow;

    __device__ Rep(const Params& p_, const State& S_, int me_, int lane_ = 0) : p(p_), S(S_), me(me_), lane(lane_) {}

    __device__ inline size_t I(int i) const { return p.rmaj ? (size_t)me * p.NB + i : (size_t)i * p.R + me; }
    __device__ inline real& R_(int b, int d) const {
        return S.r[p.rmaj ? ((size_t)me * p.NB + b) * DIM + d : ((size_t)b * DIM + d) * p.R + me];
    }
    __device__ inline int& slice(int b) const { return S.slice[I(b)]; }
    __device__ inline int& prev(int b) const { return S.prev[I(b)]; }
    __device__ inline int& next(int b) const { return S.next[I(b)]; }
    __device__ inline int& posS(int b) const { return S.posS[I(b)]; }
    __device__ inline int& posA(int b) const { return S.posA[I(b)]; }
    __device__ inline int& sl(int s, int k) const {
        return S.sl[p.rmaj ? ((size_t)me * p.P + s) * p.SM + k : ((size_t)s * p.SM + k) * p.R + me];
    }
    __device__ inline int& nsl(int s) const { return S.nsl[p.rmaj ? (size_t)me * p.P + s : (size_t)s * p.R + me]; }

    // ---- warp helpers (no-ops for W=1); results broadcast from lane 0
    __device__ inline void SYNC() const { if constexpr (W > 1) __syncwarp(); }
    template <class T> __device__ inline T wsum(T x) const {
        if constexpr (W > 1) {
            for (int o = W / 2; o > 0; o >>= 1) x += __shfl_xor_sync(FULL, x, o);
            x = __shfl_sync(FULL, x, 0);
        }
        return x;
    }
    template <class T> __device__ inline T wmax(T x) const {
        if constexpr (W > 1) {
            for (int o = W / 2; o > 0; o >>= 1) x = max(x, __shfl_xor_sync(FULL, x, o));
            x = __shfl_sync(FULL, x, 0);
        }
        return x;
    }
    __device__ inline int& act(int k) const { return S.act[I(k)]; }
    __device__ inline int& freeL(int k) const { return S.freeL[I(k)]; }

    __device__ inline real uni() {
#ifdef USE_FLOAT
        return curand_uniform(&st);
#else
        return curand_uniform_double(&st);
#endif
    }
    __device__ inline real gauss() {
#ifdef USE_FLOAT
        return curand_normal(&st);
#else
        return curand_normal_double(&st);
#endif
    }
    __device__ inline int rint(int n) { int k = (int)(uni() * n); return k >= n ? n - 1 : k; }
    __device__ inline bool accept(real logA) { return logA >= 0 || RLOG(uni()) < logA; }
    __device__ inline int wrap(int s) const { s %= p.P; return s < 0 ? s + p.P : s; }

    __device__ inline void getr(int b, real* x) const { for (int d = 0; d < DIM; d++) x[d] = R_(b, d); }
    __device__ inline real mimg(real x) const { return p.pbc ? x - p.Lbox * RRINT(x / p.Lbox) : x; }
    __device__ inline void wrapr(real* x) const { if (p.pbc) for (int d = 0; d < DIM; d++) x[d] = mimg(x[d]); }
    __device__ inline real d2(const real* a, const real* b) const {
        real s = 0; for (int d = 0; d < DIM; d++) { real x = mimg(a[d] - b[d]); s += x * x; } return s;
    }
    __device__ inline real d2b(const real* a, int b) const {
        real s = 0; for (int d = 0; d < DIM; d++) { real x = mimg(a[d] - R_(b, d)); s += x * x; } return s;
    }
    __device__ inline real vext(const real* x) const {
        real s = 0;
        if (p.omega != 0) { for (int d = 0; d < DIM; d++) s += x[d] * x[d]; s = real(0.5) * p.omega * p.omega * s; }
        if (p.ndis) for (int d = 0; d < DIM; d++) s += vdis(x[d]);
        return s;
    }
    // disorder along one axis: Re sum_k c_k z^k, z = exp(i kdis x) (Horner), as in the CPU code
    __device__ inline real vdis(real x) const {
        real zr, zi, sr = 0, si = 0;
        RSINCOS(p.kdis * x, &zi, &zr);
        for (int k = p.ndis - 1; k >= 0; k--) {
            real tr = sr + dcoef[2 * k], ti = si + dcoef[2 * k + 1];
            sr = tr * zr - ti * zi; si = tr * zi + ti * zr;
        }
        return sr;
    }
    __device__ static real aziz(real r) {   // HFDHE2 (K, Angstrom)
        const real eps = 10.8, rm = 2.9673, A = 0.5448504e6, alpha = 13.353384, D = 1.241314;
        const real C6 = 1.3732412, C8 = 0.4253785, C10 = 0.178100;
        real x = r / rm, x2 = x * x, x6 = x2 * x2 * x2;
        real F = x < D ? REXP(-(D / x - 1) * (D / x - 1)) : real(1);
        return eps * (A * REXP(-alpha * x) - F * (C6 / x6 + C8 / (x6 * x2) + C10 / (x6 * x2 * x2)));
    }
    __device__ inline real vpair(real r2) const {
        if (r2 > p.rc2) return 0;
        switch (p.pot) {
            case 1: { real s6 = p.sig * p.sig / r2; s6 = s6 * s6 * s6; return 4 * p.eps * (s6 * s6 - s6); }
            case 2: return aziz(RSQRT(r2));
            default: return real(0.5) * p.kappa * r2;
        }
    }
    __device__ inline bool pairs() const { return p.pot != 0 || p.kappa != 0; }
    __device__ __forceinline__ real U(const real* x, int s, int excl) const {
        real e = 0;
        if (pairs()) {
            int n = nsl(s);
            for (int k = lane; k < n; k += W) { int b = sl(s, k); if (b != excl) e += vpair(d2b(x, b)); }
            e = wsum(e);
        }
        return vext(x) + e;
    }
    __device__ __forceinline__ real Ub(int b) const { real x[DIM]; getr(b, x); return U(x, slice(b), b); }
    __device__ inline real logG0(real dist2, int m) const {
        real t = m * p.tau;
        return -real(0.5) * DIM * RLOG(real(4.0 * M_PI) * p.lam * t) - dist2 / (real(4) * p.lam * t);
    }
    // free propagator over m links summed over periodic images (see CPU code)
    __device__ real logG0i(const real* a, const real* b, int m) const {
        if (!p.pbc) return logG0(d2(a, b), m);
        real t4 = 4 * p.lam * m * p.tau, s = -real(0.5) * DIM * RLOG(real(M_PI) * t4);
        for (int d = 0; d < DIM; d++) {
            real x = mimg(b[d] - a[d]), z = 0;
            for (int n = -NIMG; n <= NIMG; n++) { real y = x + n * p.Lbox; z += REXP(-y * y / t4); }
            s += RLOG(z);
        }
        return s;
    }
    __device__ __forceinline__ void pickImage(const real* a, const real* b, int m, real* bt) {
        if (!p.pbc) { for (int d = 0; d < DIM; d++) bt[d] = b[d]; return; }
        real t4 = 4 * p.lam * m * p.tau;
        for (int d = 0; d < DIM; d++) {
            real x = mimg(b[d] - a[d]), w[2 * NIMG + 1], z = 0;
            for (int n = -NIMG; n <= NIMG; n++) { real y = x + n * p.Lbox; w[n + NIMG] = REXP(-y * y / t4); z += w[n + NIMG]; }
            real u = uni() * z; int k = 0;
            while (k < 2 * NIMG && u >= w[k]) { u -= w[k]; k++; }
            bt[d] = a[d] + x + (k - NIMG) * p.Lbox;
        }
    }

    // capacity check for adding beads on slices s0, s0+dir, ... (k beads); a move
    // exceeding NB/SM is rejected, i.e. such configurations get zero weight
    __device__ bool room(int s0, int dir, int k) {
        if (nact + k > p.NB) { overflow++; return false; }
        if (k > p.P) { overflow++; return false; }
        for (int j = 0; j < k; j++) if (nsl(wrap(s0 + dir * j)) + 1 > p.SM) { overflow++; return false; }
        return true;
    }
    // list updates: read phase, fence, write phase, fence (W>1: every lane
    // stores the same values, so no lane may read while another one writes)
    __device__ __forceinline__ int newBead(const real* x, int s) {
        SYNC();
        int id = nfree > 0 ? freeL(nfree - 1) : nB;
        int k = nsl(s);
        SYNC();
        if (nfree > 0) nfree--; else nB++;
        for (int d = 0; d < DIM; d++) R_(id, d) = x[d];
        slice(id) = s; prev(id) = -1; next(id) = -1;
        posS(id) = k; sl(s, k) = id; nsl(s) = k + 1;
        posA(id) = nact; act(nact) = id; nact++;
        SYNC();
        return id;
    }
    __device__ __forceinline__ void killBead(int id) {
        SYNC();
        int s = slice(id), q = posS(id), k = nsl(s) - 1, last = sl(s, k);
        int qa = posA(id), lastA = act(nact - 1);
        SYNC();
        sl(s, q) = last; posS(last) = q; nsl(s) = k;
        act(qa) = lastA; posA(lastA) = qa; nact--;
        freeL(nfree++) = id;
        SYNC();
    }

    __device__ __forceinline__ void bridge(const real* a, const real* b, int m, real (*out)[DIM]) {
        real pv[DIM]; for (int d = 0; d < DIM; d++) pv[d] = a[d];
        for (int j = 1; j <= m - 1; j++) {
            int rem = m - j + 1;
            real sd = RSQRT(real(2) * p.lam * p.tau * (real)(m - j) / (real)rem);
            for (int d = 0; d < DIM; d++) { out[j - 1][d] = pv[d] + (b[d] - pv[d]) / rem + sd * gauss(); pv[d] = out[j - 1][d]; }
        }
        for (int j = 0; j < m - 1; j++) wrapr(out[j]);
    }
    __device__ __forceinline__ void extend(const real* a, int n, real (*out)[DIM]) {
        real pv[DIM]; for (int d = 0; d < DIM; d++) pv[d] = a[d];
        real sd = RSQRT(real(2) * p.lam * p.tau);
        for (int j = 0; j < n; j++)
            for (int d = 0; d < DIM; d++) { out[j][d] = pv[d] + sd * gauss(); pv[d] = out[j][d]; }
        for (int j = 0; j < n; j++) wrapr(out[j]);
    }

    // ---------------------------------------------------------------- moves
    __device__ __forceinline__ void open_move() {
        if (worm || nact == 0) return;
        att[0]++;
        int Nb = nact;
        int b0 = act(rint(Nb));
        int m = 1 + rint(p.Lmax);
        int ch[LCAP + 1]; ch[0] = b0;
        for (int j = 1; j <= m; j++) ch[j] = next(ch[j - 1]);
        real dV = 0;
        for (int j = 1; j <= m - 1; j++) dV += Ub(ch[j]);
        real a[DIM], b[DIM]; getr(b0, a); getr(ch[m], b);
        real logA = p.logC + RLOG((real)Nb) + p.logLmax + p.tau * dV - (m - 1) * p.mu * p.tau - logG0i(a, b, m);
        if (!accept(logA)) return;
        for (int j = 1; j <= m - 1; j++) killBead(ch[j]);
        next(b0) = -1; prev(ch[m]) = -1;
        head = b0; tail = ch[m]; worm = 1;
        acc[0]++;
    }
    __device__ __forceinline__ void close_move() {
        if (!worm) return;
        int sI = slice(head), sM = slice(tail);
        int g = wrap(sM - sI - 1) + 1;
        if (g > p.Lmax) return;
        att[1]++;
        real np[LCAP][DIM], a[DIM], b[DIM], bt[DIM]; getr(head, a); getr(tail, b);
        pickImage(a, b, g, bt);
        bridge(a, bt, g, np);
        real dV = 0;
        for (int k = 1; k <= g - 1; k++) dV += U(np[k - 1], wrap(sI + k), -1);
        int NbAfter = nact + (g - 1);
        real logA = logG0i(a, b, g) + (g - 1) * p.mu * p.tau - p.tau * dV - p.logC - RLOG((real)NbAfter) - p.logLmax;
        if (!accept(logA)) return;
        if (!room(sI + 1, 1, g - 1)) return;
        int cur = head;
        for (int k = 1; k <= g - 1; k++) { int nb = newBead(np[k - 1], wrap(sI + k)); next(cur) = nb; prev(nb) = cur; cur = nb; }
        next(cur) = tail; prev(tail) = cur;
        head = tail = -1; worm = 0;
        acc[1]++;
    }
    __device__ __forceinline__ void insert_move() {
        if (worm) return;
        att[2]++;
        int sM = rint(p.P), ell = rint(p.Lmax);
        real r0[DIM]; for (int d = 0; d < DIM; d++) r0[d] = (uni() - real(0.5)) * p.Lbox;
        real np[LCAP][DIM]; extend(r0, ell, np);
        real dV = U(r0, sM, -1);
        for (int k = 1; k <= ell; k++) dV += U(np[k - 1], wrap(sM + k), -1);
        real logA = p.logC + p.logVbox + p.logP + p.logLmax + (ell + 1) * p.mu * p.tau - p.tau * dV;
        if (!accept(logA)) return;
        if (!room(sM, 1, ell + 1)) return;
        int cur = newBead(r0, sM); tail = cur;
        for (int k = 1; k <= ell; k++) { int nb = newBead(np[k - 1], wrap(sM + k)); next(cur) = nb; prev(nb) = cur; cur = nb; }
        head = cur; worm = 1;
        acc[2]++;
    }
    __device__ __forceinline__ void remove_move() {
        if (!worm) return;
        int b = head, ell = 0;
        while (b != tail) { if (ell >= p.Lmax - 1) return; b = prev(b); ell++; }
        for (int d = 0; d < DIM; d++) if (RFABS(R_(tail, d)) > real(0.5) * p.Lbox) return;
        att[3]++;
        real dV = 0; b = head;
        while (true) { dV += Ub(b); if (b == tail) break; b = prev(b); }
        real logA = -(p.logC + p.logVbox + p.logP + p.logLmax + (ell + 1) * p.mu * p.tau - p.tau * dV);
        if (!accept(logA)) return;
        b = head;
        while (true) { int q = prev(b); bool last = (b == tail); killBead(b); if (last) break; b = q; }
        head = tail = -1; worm = 0;
        acc[3]++;
    }
    __device__ __forceinline__ void advance_move(int dir) {
        if (!worm) return;
        int idx = dir > 0 ? 4 : 6;
        att[idx]++;
        int m = 1 + rint(p.Lmax);
        int e = dir > 0 ? head : tail;
        real np[LCAP][DIM], a[DIM]; getr(e, a); extend(a, m, np);
        real dV = 0; int s0 = slice(e);
        for (int k = 1; k <= m; k++) dV += U(np[k - 1], wrap(s0 + dir * k), -1);
        real logA = m * p.mu * p.tau - p.tau * dV;
        if (!accept(logA)) return;
        if (!room(s0 + dir, dir, m)) return;
        int cur = e;
        for (int k = 1; k <= m; k++) {
            int nb = newBead(np[k - 1], wrap(s0 + dir * k));
            if (dir > 0) { next(cur) = nb; prev(nb) = cur; } else { prev(cur) = nb; next(nb) = cur; }
            cur = nb;
        }
        if (dir > 0) head = cur; else tail = cur;
        acc[idx]++;
    }
    __device__ __forceinline__ void recede_move(int dir) {
        if (!worm) return;
        int idx = dir > 0 ? 5 : 7;
        int m = 1 + rint(p.Lmax);
        int e = dir > 0 ? head : tail, other = dir > 0 ? tail : head;
        int rem[LCAP]; int b = e;
        for (int t = 0; t < m; t++) { if (b == other) return; rem[t] = b; b = dir > 0 ? prev(b) : next(b); }
        att[idx]++;
        real dV = 0;
        for (int t = 0; t < m; t++) dV += Ub(rem[t]);
        real logA = -m * p.mu * p.tau + p.tau * dV;
        if (!accept(logA)) return;
        for (int t = 0; t < m; t++) killBead(rem[t]);
        if (dir > 0) { head = b; next(b) = -1; } else { tail = b; prev(b) = -1; }
        acc[idx]++;
    }
    // walk m steps back from j; returns -1 if the chain ends earlier
    __device__ inline int back(int j, int m) const {
        for (int t = 0; t < m; t++) { j = prev(j); if (j < 0) return -1; }
        return j;
    }
    // common tail of the swap: new bridge head -> j, acceptance, relinking
    __device__ __forceinline__ void swap_finish(int m, int h, int s, int j, int q, const real* rh, real dlogsum) {
        int oldc[LCAP];
        { int x = j; for (int t = 0; t < m - 1; t++) { x = prev(x); oldc[t] = x; } }
        real np[LCAP][DIM], rj[DIM], bt[DIM]; getr(j, rj);
        pickImage(rh, rj, m, bt);
        bridge(rh, bt, m, np);
        real dV = 0;
        for (int k = 0; k < m - 1; k++) {
            int ob = oldc[m - 2 - k]; int sk = wrap(s + 1 + k);
            real ro[DIM]; getr(ob, ro);
            dV += U(np[k], sk, ob) - U(ro, sk, ob);
        }
        real logA = dlogsum - p.tau * dV;
        if (!accept(logA)) return;
        for (int t = 0; t < m - 1; t++) killBead(oldc[t]);   // frees room first
        int cur = h;
        for (int k = 0; k < m - 1; k++) { int nb = newBead(np[k], wrap(s + 1 + k)); next(cur) = nb; prev(nb) = cur; cur = nb; }
        next(cur) = j; prev(j) = cur;
        next(q) = -1; head = q;
        acc[8]++;
    }
    // Swap: candidates are the beads on slice s+m whose chain reaches back m
    // links; with W>1 each lane handles a contiguous chunk of the slice list
    __device__ __forceinline__ void swap_move() {
        if (!worm) return;
        int m = 1 + rint(p.Lmax);
        int h = head, s = slice(h), sp = wrap(s + m);
        int n = nsl(sp);
        real rh[DIM]; getr(h, rh);
        int chunk = (n + W - 1) / W, k0 = lane * chunk, k1 = min(n, k0 + chunk);
        // pass 1: normalisation from the head (online log-sum-exp per lane)
        real mxl = -1e30f, sml = 0; int ncl = 0;
        for (int k = k0; k < k1; k++) {
            int jj = sl(sp, k); if (back(jj, m) < 0) continue;
            ncl++;
            real rj[DIM]; getr(jj, rj); real e = logG0i(rh, rj, m);
            if (e > mxl) { sml = sml * REXP(mxl - e) + 1; mxl = e; } else sml += REXP(e - mxl);
        }
        int nc = wsum(ncl);
        if (nc == 0) return;
        att[8]++;
        real mx = wmax(mxl);
        real part = ncl ? sml * REXP(mxl - mx) : real(0);
        real sum1 = wsum(part);
        // pass 2: choose candidate (prefix sums over lanes, then within the owner's chunk)
        real u = uni() * sum1;
        real inc = part;
        int owner = 0;
        if constexpr (W > 1) {
            for (int o = 1; o < W; o <<= 1) { real t = __shfl_up_sync(FULL, inc, o); if (lane >= o) inc += t; }
            unsigned b1 = __ballot_sync(FULL, ncl > 0 && u < inc);
            unsigned b2 = __ballot_sync(FULL, ncl > 0);
            owner = b1 ? __ffs(b1) - 1 : 31 - __clz(b2);     // fallback: last candidate
        }
        int j = -1, q = -1;
        if (lane == owner) {
            real cs = inc - part;
            for (int k = k0; k < k1; k++) {
                int jj = sl(sp, k); int qq = back(jj, m); if (qq < 0) continue;
                j = jj; q = qq;
                real rj[DIM]; getr(jj, rj);
                cs += REXP(logG0i(rh, rj, m) - mx);
                if (u < cs) break;
            }
        }
        if constexpr (W > 1) { j = __shfl_sync(FULL, j, owner); q = __shfl_sync(FULL, q, owner); }
        real logsum1 = mx + RLOG(sum1);
        // pass 3: reverse normalisation from q
        real rq[DIM]; getr(q, rq);
        real mx2l = -1e30f, s2l = 0;
        for (int k = k0; k < k1; k++) {
            int jj = sl(sp, k); if (back(jj, m) < 0) continue;
            real rj[DIM]; getr(jj, rj); real e = logG0i(rq, rj, m);
            if (e > mx2l) { s2l = s2l * REXP(mx2l - e) + 1; mx2l = e; } else s2l += REXP(e - mx2l);
        }
        real mx2 = wmax(mx2l);
        real sum2 = wsum(s2l > 0 ? s2l * REXP(mx2l - mx2) : real(0));
        real logsum2 = mx2 + RLOG(sum2);
        swap_finish(m, h, s, j, q, rh, logsum1 - logsum2);
    }
    __device__ __forceinline__ void stage_move() {
        if (nact == 0 || p.Lmax < 2) return;
        int a = act(rint(nact));
        int m = 2 + rint(p.Lmax - 1);
        int ch[LCAP + 1]; ch[0] = a;
        for (int j = 1; j <= m; j++) { int nx = next(ch[j - 1]); if (nx < 0) return; ch[j] = nx; }
        att[9]++;
        // keep the segment's actual end-to-end displacement (winding preserved)
        real np[LCAP][DIM], ra[DIM], bt[DIM]; getr(a, ra);
        for (int d = 0; d < DIM; d++) bt[d] = ra[d];
        for (int j = 1; j <= m; j++) for (int d = 0; d < DIM; d++) bt[d] += mimg(R_(ch[j], d) - R_(ch[j - 1], d));
        bridge(ra, bt, m, np);
        real dV = 0;
        for (int j = 1; j <= m - 1; j++) {
            int x = ch[j]; int sk = slice(x);
            real ro[DIM]; getr(x, ro);
            dV += U(np[j - 1], sk, x) - U(ro, sk, x);
        }
        if (!accept(-p.tau * dV)) return;
        for (int j = 1; j <= m - 1; j++) for (int d = 0; d < DIM; d++) R_(ch[j], d) = np[j - 1][d];
        acc[9]++;
    }

    __device__ __forceinline__ void step() {
        SYNC();
        switch (rint(12)) {
            case 0: open_move(); break;
            case 1: close_move(); break;
            case 2: insert_move(); break;
            case 3: remove_move(); break;
            case 4: advance_move(+1); break;
            case 5: recede_move(+1); break;
            case 6: advance_move(-1); break;
            case 7: recede_move(-1); break;
            case 8: swap_move(); break;
            default: stage_move(); break;
        }
    }

    __device__ __forceinline__ void measure(double* o) const {
        SYNC();
        double e = 0;
        for (int s = 0; s < p.P; s++) {
            int n = nsl(s);
            for (int i = lane; i < n; i += W) {
                int bi = sl(s, i); real ri[DIM]; getr(bi, ri);
                e += vext(ri);
                if (pairs()) for (int k = i + 1; k < n; k++) e += vpair(d2b(ri, sl(s, k)));
            }
        }
        double k2 = 0, D[DIM];
        for (int d = 0; d < DIM; d++) D[d] = 0;
        for (int k = lane; k < nact; k += W) {
            int b = act(k), c = next(b);
            for (int d = 0; d < DIM; d++) { double x = mimg(R_(c, d) - R_(b, d)); k2 += x * x; D[d] += x; }
        }
        e = wsum(e); k2 = wsum(k2);
        double D2 = 0; for (int d = 0; d < DIM; d++) { D[d] = wsum(D[d]); D2 += D[d] * D[d]; }
        // exchange cycles: walking a loop of j particles (P*j beads) from each of
        // its j beads on slice 0 adds j to bin j
        for (int k = 0; k < NCYC; k++) o[O_C + k] = 0;
        int n0 = nsl(0);
        for (int i = lane; i < n0; i += W) {
            int b0 = sl(0, i), c = b0, len = 0;
            do { c = next(c); len++; } while (c != b0);
            o[O_C + min(len / p.P, NCYC) - 1] += 1;
        }
        if constexpr (W > 1) for (int k = 0; k < NCYC; k++) o[O_C + k] = wsum(o[O_C + k]);
        double tau = p.tau;
        o[O_N] = nact / (double)p.P;
        o[O_V] = e / p.P;
        o[O_E] = (0.5 * DIM * nact / tau - k2 / (4.0 * p.lam * tau * tau)) / p.P + e / p.P;
        o[O_D2] = D2;
        o[O_N2] = o[O_N] * o[O_N];
    }

    __device__ void load() {
        st = S.rng[me];
        nact = S.nact[me]; nfree = S.nfree[me]; nB = S.nB[me];
        head = S.head[me]; tail = S.tail[me]; worm = S.worm[me];
        for (int i = 0; i < NMOVE; i++) { att[i] = 0; acc[i] = 0; }
        overflow = 0;
    }
    __device__ void store() {
        if (lane != 0) return;
        S.rng[me] = st;
        S.nact[me] = nact; S.nfree[me] = nfree; S.nB[me] = nB;
        S.head[me] = head; S.tail[me] = tail; S.worm[me] = worm;
        for (int i = 0; i < NMOVE; i++) { S.att[i * p.R + me] += att[i]; S.acc[i * p.R + me] += acc[i]; }
        S.overflow[me] += overflow;
    }
};

__global__ void init_kernel(Params p, State S, unsigned long long seed) {
    int me = blockIdx.x * blockDim.x + threadIdx.x;
    if (me >= p.R) return;
    curand_init(seed, me, 0, &S.rng[me]);
    S.nact[me] = 0; S.nfree[me] = 0; S.nB[me] = 0; S.head[me] = -1; S.tail[me] = -1; S.worm[me] = 0;
    Rep<1> X(p, S, me);
    for (int s = 0; s < p.P; s++) X.nsl(s) = 0;
    for (int k = 0; k < NOBS; k++) S.sumO[k * p.R + me] = 0;
    S.nZ[me] = 0; S.nAll[me] = 0; S.overflow[me] = 0;
    for (int i = 0; i < NMOVE; i++) { S.att[i * p.R + me] = 0; S.acc[i * p.R + me] = 0; }
}

// optional initial state: N0 particles as straight closed worldlines on a
// simple lattice filling the box [-Lbox/2, Lbox/2)^d
__global__ void lattice_kernel(Params p, State S, int N0) {
    int me = blockIdx.x * blockDim.x + threadIdx.x;
    if (me >= p.R) return;
    Rep<1> X(p, S, me);
    X.load();
    int ns = (int)ceil(pow((double)N0, 1.0 / DIM) - 1e-9);
    real a = p.Lbox / ns;
    for (int i = 0; i < N0; i++) {
        real x[DIM]; int c = i;
        for (int d = 0; d < DIM; d++) { x[d] = -real(0.5) * p.Lbox + (c % ns + real(0.5)) * a; c /= ns; }
        int first = X.newBead(x, 0), cur = first;
        for (int s = 1; s < p.P; s++) { int nb = X.newBead(x, s); X.next(cur) = nb; X.prev(nb) = cur; cur = nb; }
        X.next(cur) = first; X.prev(first) = cur;
    }
    X.store();
}

// runs steps [it0, it0+n) of the global schedule; measurements after 'therm'
template <int W>
__global__ void run_kernel(Params p, State S, long long it0, long long n, long long therm, int meas) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int me = gid / W, lane = gid % W;
    if (me >= p.R) return;                       // whole warps exit together
    Rep<W> X(p, S, me, lane);
    X.load();
    double so[NOBS]; for (int k = 0; k < NOBS; k++) so[k] = 0;
    long long nZ = 0, nAll = 0;
    for (long long it = it0; it < it0 + n; it++) {
        X.step();
        if (it >= therm && ((it - therm) % meas == 0)) {
            nAll++;
            if (!X.worm) { double o[NOBS]; X.measure(o); for (int k = 0; k < NOBS; k++) so[k] += o[k]; nZ++; }
        }
    }
    X.store();
    if (lane != 0) return;
    for (int k = 0; k < NOBS; k++) S.sumO[k * p.R + me] += so[k];
    S.nZ[me] += nZ; S.nAll[me] += nAll;
}

// consistency check (debug): returns 0 if fine
__global__ void check_kernel(Params p, State S, int* bad) {
    int me = blockIdx.x * blockDim.x + threadIdx.x;
    if (me >= p.R) return;
    Rep<1> X(p, S, me);
    X.load();
    int err = 0;
    long cnt = 0; for (int s = 0; s < p.P; s++) cnt += X.nsl(s);
    if (cnt != X.nact) err = 1;
    int nh = 0, nt = 0;
    for (int k = 0; k < X.nact && !err; k++) {
        int id = X.act(k);
        if (X.posA(id) != k) err = 2;
        if (X.sl(X.slice(id), X.posS(id)) != id) err = 3;
        int nx = X.next(id), pv = X.prev(id);
        if (nx >= 0) { if (X.prev(nx) != id || X.slice(nx) != (X.slice(id) + 1) % p.P) err = 4; } else nh++;
        if (pv >= 0) { if (X.next(pv) != id) err = 5; } else nt++;
    }
    if (!err) {
        if (X.worm) { if (nh != 1 || nt != 1 || X.next(X.head) != -1 || X.prev(X.tail) != -1) err = 6; }
        else if (nh != 0 || nt != 0) err = 7;
    }
    if (err) atomicMax(bad, err);
}

// disorder coefficients c_k = Vd sqrt(2/M) exp(i phi_k), phi_k uniform from splitmix64(dseed) (as in worm_pimc.cpp)
static void disorder(int M, double Vd, unsigned long long seed, std::vector<double>& c) {
    c.resize(2 * M);
    for (int k = 0; k < M; k++) {
        unsigned long long z = (seed += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31;
        double ph = 2.0 * M_PI * (double)(z >> 11) * 0x1p-53, a = Vd * std::sqrt(2.0 / M);
        c[2 * k] = a * std::cos(ph); c[2 * k + 1] = a * std::sin(ph);
    }
}

static double getd(int argc, char** argv, const char* key, double def) {
    size_t n = strlen(key);
    double v = def;                       // the last occurrence wins
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], key, n) && argv[i][n] == '=') v = atof(argv[i] + n + 1);
    return v;
}

template <class T> static T* dalloc(size_t n) {
    T* x; cudaError_t e = cudaMalloc(&x, n * sizeof(T));
    if (e != cudaSuccess) {
        size_t fr, tot; cudaMemGetInfo(&fr, &tot);
        fprintf(stderr, "cudaMalloc(%zu bytes) failed: %s (free %zu / total %zu)\n", n * sizeof(T), cudaGetErrorString(e), fr, tot);
        exit(3);
    }
    return x;
}

int main(int argc, char** argv) {
    Params p;
    p.P = (int)getd(argc, argv, "P", 24);
    double beta = getd(argc, argv, "beta", 1.0);
    p.mu = getd(argc, argv, "mu", 0.0);
    p.C = getd(argc, argv, "C", 1.0);
    p.Lmax = (int)getd(argc, argv, "Lmax", 8);
    p.Lbox = getd(argc, argv, "Lbox", 12.0);
    p.omega = getd(argc, argv, "omega", 1.0);
    p.kappa = getd(argc, argv, "kappa", 0.0);
    double lam = getd(argc, argv, "lam", 0.5);
    p.pbc = (int)getd(argc, argv, "pbc", 0);
    p.pot = (int)getd(argc, argv, "pot", 0);
    p.eps = getd(argc, argv, "eps", 1.0);
    p.sig = getd(argc, argv, "sig", 1.0);
    double rc = getd(argc, argv, "rc", p.pbc ? 0.5 * (double)p.Lbox : 1e300);
    double Vd = getd(argc, argv, "Vd", 0.0), qc = getd(argc, argv, "qc", 2.0);
    unsigned long long dseed = (unsigned long long)getd(argc, argv, "dseed", 1);
    p.R = (int)getd(argc, argv, "R", 4096);              // replicas
    p.NB = (int)getd(argc, argv, "NB", 0);                // bead capacity per replica
    p.SM = (int)getd(argc, argv, "SM", 64);               // capacity per slice
    double nsteps = getd(argc, argv, "steps", 1e6);       // steps PER REPLICA
    double ntherm = getd(argc, argv, "therm", 1e5);
    int meas = (int)getd(argc, argv, "meas", 20);
    int ngroups = (int)getd(argc, argv, "blocks", 32);    // jackknife groups of replicas
    long long chunk = (long long)getd(argc, argv, "chunk", 1e5);
    int debug = (int)getd(argc, argv, "debug", 0);
    int tpb = (int)getd(argc, argv, "tpb", 128);
    int W = (int)getd(argc, argv, "W", 1);                 // 1: replica per thread, 32: replica per warp
    p.rmaj = (int)getd(argc, argv, "rmaj", W > 1 ? 1 : 0);
    int N0 = (int)getd(argc, argv, "N0", 0);              // initial particles (lattice)
    unsigned long long seed = (unsigned long long)getd(argc, argv, "seed", 12345);
    int dev = (int)getd(argc, argv, "dev", 0);
    if (p.Lmax > p.P - 2) { fprintf(stderr, "need Lmax <= P-2\n"); return 1; }
    if (p.Lmax + 1 > LCAP) { fprintf(stderr, "need Lmax+1 <= LCAP=%d (recompile with -DLCAP=...)\n", LCAP); return 1; }
    if (p.NB <= 0) p.NB = p.SM * p.P;
    if (ngroups > p.R) ngroups = p.R;
    if (ngroups < 2) { fprintf(stderr, "need R >= 2\n"); return 1; }
    if (W != 1 && W != 32) { fprintf(stderr, "W must be 1 or 32\n"); return 1; }
    if (tpb % 32) { fprintf(stderr, "tpb must be a multiple of 32\n"); return 1; }
    if (N0 > 0 && (N0 > p.SM || (long)N0 * p.P > p.NB)) { fprintf(stderr, "N0 exceeds SM/NB capacity\n"); return 1; }
    if (p.pbc && rc > 0.5 * (double)p.Lbox * (1 + 1e-6)) { fprintf(stderr, "need rc <= Lbox/2 with pbc\n"); return 1; }
#ifdef USE_FLOAT
    p.rc2 = rc < 1e18 ? (real)(rc * rc) : (real)1e37;
#else
    p.rc2 = rc < 1e150 ? rc * rc : 1e300;
#endif
    p.lam = lam;
    p.tau = beta / p.P;
    p.Vbox = pow(p.Lbox, DIM);
    p.logC = log(p.C); p.logVbox = log(p.Vbox); p.logP = log((double)p.P); p.logLmax = log((double)p.Lmax);

    CK(cudaSetDevice(dev));
    p.ndis = 0; p.kdis = 2.0 * M_PI / (double)p.Lbox;
    if (Vd != 0.0) {
        p.ndis = (int)std::floor(qc * (double)p.Lbox / (2.0 * M_PI));
        if (p.ndis < 1 || p.ndis > DMAX) { fprintf(stderr, "need 1 <= qc Lbox / 2 pi <= DMAX=%d\n", DMAX); return 1; }
        std::vector<double> c; disorder(p.ndis, Vd, dseed, c);
        std::vector<real> cr(c.begin(), c.end());
        CK(cudaMemcpyToSymbol(dcoef, cr.data(), cr.size() * sizeof(real)));
    }
    size_t stack = (size_t)getd(argc, argv, "stack", 0);
    if (stack) CK(cudaDeviceSetLimit(cudaLimitStackSize, stack));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));

    size_t R = p.R;
    {   // device memory estimate (configuration arrays dominate)
        double need = (double)R * ((double)p.NB * (DIM * sizeof(real) + 7 * sizeof(int)) + (double)p.P * (p.SM + 1) * sizeof(int)
                                   + NOBS * 8 + 2 * NMOVE * 8 + 128);
        size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
        if (need > 0.95 * fr) {
            fprintf(stderr, "error: need ~%.2f GB of device memory, %.2f GB free; reduce R, NB or SM\n", need / 1e9, fr / 1e9);
            return 1;
        }
    }
    State S;
    S.r = dalloc<real>((size_t)p.NB * DIM * R);
    S.slice = dalloc<int>(p.NB * R); S.prev = dalloc<int>(p.NB * R); S.next = dalloc<int>(p.NB * R);
    S.posS = dalloc<int>(p.NB * R); S.posA = dalloc<int>(p.NB * R);
    S.sl = dalloc<int>((size_t)p.P * p.SM * R); S.nsl = dalloc<int>(p.P * R);
    S.act = dalloc<int>(p.NB * R); S.freeL = dalloc<int>(p.NB * R);
    S.nact = dalloc<int>(R); S.nfree = dalloc<int>(R); S.nB = dalloc<int>(R);
    S.head = dalloc<int>(R); S.tail = dalloc<int>(R); S.worm = dalloc<int>(R);
    S.rng = dalloc<curandStatePhilox4_32_10_t>(R);
    S.sumO = dalloc<double>(NOBS * R);
    S.nZ = dalloc<long long>(R); S.nAll = dalloc<long long>(R); S.overflow = dalloc<long long>(R);
    S.att = dalloc<long long>(NMOVE * R); S.acc = dalloc<long long>(NMOVE * R);
    int* dbad = dalloc<int>(1);

    int nblk = (p.R + tpb - 1) / tpb;
    init_kernel<<<nblk, tpb>>>(p, S, seed);
    CK(cudaGetLastError());
    if (N0 > 0) { lattice_kernel<<<nblk, tpb>>>(p, S, N0); CK(cudaGetLastError()); }
    long long nthr = (long long)p.R * W;
    int nblkW = (int)((nthr + tpb - 1) / tpb);

    long long total = (long long)nsteps, th = (long long)ntherm, all = th + total;
    CK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    // chunks never straddle the end of thermalisation, so production can be timed alone
    double tth = 0;
    for (long long it = 0; it < all; ) {
        if (it == th) { CK(cudaDeviceSynchronize()); tth = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
        long long n = std::min(chunk, (it < th ? th : all) - it);
        if (W == 1) run_kernel<1><<<nblkW, tpb>>>(p, S, it, n, th, meas);
        else        run_kernel<32><<<nblkW, tpb>>>(p, S, it, n, th, meas);
        CK(cudaGetLastError());
        if (debug) {
            CK(cudaMemset(dbad, 0, sizeof(int)));
            check_kernel<<<nblk, tpb>>>(p, S, dbad);
            int bad; CK(cudaMemcpy(&bad, dbad, sizeof(int), cudaMemcpyDeviceToHost));
            if (bad) { fprintf(stderr, "INCONSISTENT (code %d) after step %lld\n", bad, it + n); return 2; }
        }
        it += n;
    }
    CK(cudaDeviceSynchronize());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::vector<double> sO(NOBS * R); std::vector<long long> nZ(R), nAll(R), ovf(R), att(NMOVE * R), acc(NMOVE * R);
    CK(cudaMemcpy(sO.data(), S.sumO, NOBS * R * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(nZ.data(), S.nZ, R * sizeof(long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(nAll.data(), S.nAll, R * sizeof(long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ovf.data(), S.overflow, R * sizeof(long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(att.data(), S.att, NMOVE * R * sizeof(long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(acc.data(), S.acc, NMOVE * R * sizeof(long long), cudaMemcpyDeviceToHost));

    // jackknife over groups of replicas (ratio estimators <O> = sum O / sum nZ);
    // derived: rho_s/rho = <D2>/(2 lam d beta <N>), <dN^2>, <W^2> = <D2>/(d Lbox^2),
    // K = pi sqrt(<W^2> <dN^2>)
    enum { Q_RS = NOBS, Q_DN2, Q_W2, Q_K, NQ };
    const double Lb = p.Lbox;
    auto derive = [&](const double* m, double* q) {
        for (int k = 0; k < NOBS; k++) q[k] = m[k];
        q[Q_RS] = m[O_N] > 0 ? m[O_D2] / (2.0 * lam * DIM * beta * m[O_N]) : 0.0;
        q[Q_DN2] = m[O_N2] - m[O_N] * m[O_N];
        q[Q_W2] = m[O_D2] / (DIM * Lb * Lb);
        q[Q_K] = M_PI * sqrt(std::max(0.0, q[Q_W2] * q[Q_DN2]));
    };
    std::vector<double> gO(ngroups * NOBS, 0), gZ(ngroups, 0);
    double TO[NOBS] = {0}, TZ = 0, TA = 0; long long OV = 0;
    for (size_t i = 0; i < R; i++) {
        int g = (int)(i * ngroups / R);
        for (int k = 0; k < NOBS; k++) { gO[g * NOBS + k] += sO[k * R + i]; TO[k] += sO[k * R + i]; }
        gZ[g] += nZ[i]; TZ += nZ[i]; TA += nAll[i]; OV += ovf[i];
    }
    double m[NOBS], mean[NQ], err[NQ] = {0};
    for (int k = 0; k < NOBS; k++) m[k] = TO[k] / TZ;
    derive(m, mean);
    for (int g = 0; g < ngroups; g++) {
        double mj[NOBS], qj[NQ];
        for (int k = 0; k < NOBS; k++) mj[k] = (TO[k] - gO[g * NOBS + k]) / (TZ - gZ[g]);
        derive(mj, qj);
        for (int k = 0; k < NQ; k++) err[k] += (qj[k] - mean[k]) * (qj[k] - mean[k]);
    }
    for (int k = 0; k < NQ; k++) err[k] = sqrt(err[k] * (ngroups - 1.0) / ngroups);

    const char* names[10] = {"open", "close", "insert", "remove", "advH", "recH", "advT", "recT", "swap", "stage"};
    printf("GPU=%s W=%d N0=%d DIM=%d %s P=%d beta=%g mu=%g C=%g Lmax=%d Lbox=%g omega=%g kappa=%g lam=%g pbc=%d pot=%d eps=%g sig=%g rc=%g R=%d steps/rep=%.3g therm=%.3g\n",
           prop.name, W, N0, DIM, sizeof(real) == 4 ? "float" : "double", p.P, beta, (double)p.mu, (double)p.C, p.Lmax,
           (double)p.Lbox, (double)p.omega, (double)p.kappa, lam, p.pbc, p.pot, (double)p.eps, (double)p.sig, rc,
           p.R, nsteps, ntherm);
    printf("RESULT N= %.5f +- %.5f   Vbar= %.5f +- %.5f   Zfrac= %.4f   time=%.2fs   rate=%.3g steps/s   overflow=%lld\n",
           mean[O_N], err[O_N], mean[O_V], err[O_V], TA ? TZ / TA : 0.0, secs, (double)all * p.R / secs, OV);
    printf("TIMING therm=%.2fs prod=%.2fs prod_rate=%.4g steps/s\n", th > 0 ? tth : 0.0, secs - tth,
           secs > tth ? (double)total * p.R / (secs - tth) : 0.0);
    printf("RESULT2 E= %.5f +- %.5f   D2= %.5f +- %.5f   rhos= %.5f +- %.5f\n",
           mean[O_E], err[O_E], mean[O_D2], err[O_D2], mean[Q_RS], err[Q_RS]);
    printf("RESULT3 dN2= %.5f +- %.5f   W2= %.5e +- %.2e   K= %.5f +- %.5f\n",
           mean[Q_DN2], err[Q_DN2], mean[Q_W2], err[Q_W2], mean[Q_K], err[Q_K]);
    if (p.ndis) printf("DISORDER Vd=%g qc=%g dseed=%llu modes=%d\n", Vd, qc, dseed, p.ndis);
    printf("CYCLES");
    for (int k = 0; k < NCYC; k++)
        printf(" n%d%s= %.7f +- %.7f", k + 1, k == NCYC - 1 ? "+" : "", mean[O_C + k], err[O_C + k]);
    printf("\n");
    printf("acceptance:");
    for (int i = 0; i < NMOVE; i++) {
        long long a = 0, c = 0; for (size_t k = 0; k < R; k++) { a += att[i * R + k]; c += acc[i * R + k]; }
        printf(" %s=%.3f", names[i], a ? (double)c / a : 0.0);
    }
    printf("\n");
    if (OV) fprintf(stderr, "WARNING: %lld moves rejected for exceeding NB/SM: sampling a truncated ensemble (increase NB/SM unless negligible)\n", OV);
    return 0;
}
