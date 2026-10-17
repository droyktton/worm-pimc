// worm_pimc.cpp -- Worm algorithm for continuous-space path-integral Monte Carlo
// (bosons, grand-canonical, primitive approximation).
//
// Reference: M. Boninsegni, N. Prokof'ev, B. Svistunov, PRL 96, 070601 (2006);
//            PRE 74, 036701 (2006).
//
// Units: hbar = m = 1  =>  lambda = hbar^2/(2m) = 1/2.
// Weight of a configuration (primitive approximation, P slices, tau = beta/P):
//     W = C^{[worm]} * prod_beads exp(mu*tau) * prod_links G0(r,r';tau)
//         * exp(-tau * sum_slices V(R_k))
// with G0(r,r';t) = (4 pi lambda t)^(-d/2) exp(-|r-r'|^2/(4 lambda t)).
//
// Sectors: Z (closed loops only) and G (one open worldline: tail "Masha" ->
// head "Ira").  All averages are taken in the Z sector.
//
// Moves (every pair satisfies detailed balance; selection probabilities of
// the two members of a pair are equal):
//   Open/Close, Insert/Remove, Advance/Recede (head and tail), Swap (head),
//   Staging (Levy bridge on a segment, any sector).
//
// Options: pbc=1 (periodic box of side Lbox, minimum image on links and
// pairs), pot=0 harmonic pair kappa / 1 Lennard-Jones (eps,sig) / 2 Aziz
// HFDHE2 for 4He (K, Angstrom; use lam=6.0596), pair cutoff rc, 3 contact
// g delta(x_i - x_j) in 1D (Lieb-Liniger) with the exact two-body pair-product action:
// each pair of links on the same slice carries exp(-u), u = -ln[rho_rel/rho0_rel] (see ucontact).
// Disorder (Vd > 0): V(x) = Vd sqrt(2/M) sum_{k=1..M} cos(2 pi k x/Lbox + phi_k)
// added along each axis, M = floor(qc Lbox / 2 pi) modes up to wavevector qc,
// random phases from dseed (splitmix64, identical in the GPU code); its spatial
// variance is exactly Vd^2.
// Z-sector estimators: N, Vbar, thermodynamic energy E, winding <D^2>
// (D = sum of link displacements; rho_s/rho = <D^2>/(2 lam d beta <N>)),
// exchange cycles n_j = number of particles in permutation cycles of length j
// (j = 1..NCYC-1, last bin j >= NCYC; sum_j n_j = N), <N^2> and the Luttinger
// parameter K = pi sqrt(<W^2> <dN^2>) (W = D/Lbox per axis; meaningful in 1D, pbc).
// With nsq > 0 (pbc): rho_q = sum_i exp(i q x_i) on each slice for q = 2 pi m/Lbox along x, m = 1..nsq;
// <|rho_q|^2> and <rho_q> averaged over slices give S(q) = <|rho_q|^2>/<N> and the connected
// S_c(q) = (<|rho_q|^2> - |<rho_q>|^2)/<N> (they differ only with disorder).
//
// Build:  g++ -O2 -std=c++17 -DDIM=1 worm_pimc.cpp -o worm_pimc_1d
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <map>
#include <vector>
#include <array>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>
#include <complex>

#ifndef DIM
#define DIM 1
#endif
#ifndef NCYC
#define NCYC 12            // cycle-length bins (last one: j >= NCYC)
#endif
#ifndef NSQ
#define NSQ 32             // maximum number of wavevectors of the static structure factor (parameter nsq)
#endif

typedef std::array<double, DIM> Vec;

struct Bead {
    double r[DIM];
    int slice, prev, next, posS, posA;
};

struct Sim {
    // ---- parameters
    int P = 24, Lmax = 8;
    double beta = 1.0, tau = 0.0, mu = 0.0, C = 1.0, Lbox = 12.0;
    double omega = 1.0, kappa = 0.0;   // external trap, harmonic pair coupling
    double lam = 0.5;                  // hbar^2/(2m)
    double Vbox = 1.0;
    int pbc = 0, pot = 0;              // periodic box; pair potential type
    double eps = 1.0, sig = 1.0, rc = 1e300, rc2 = 1e300;
    double gc = 0.0;                   // contact strength (pot=3); gc < 0: hard core (g = infinity)
    int nsq = 0;                       // structure-factor wavevectors (0 = off)
    int ndis = 0;                      // disorder modes; dc = (Re, Im) of c_k, k = 1..ndis
    std::vector<double> dc;

    std::mt19937_64 rng;
    std::normal_distribution<double> nd{0.0, 1.0};
    std::uniform_real_distribution<double> ud{0.0, 1.0};

    // ---- configuration
    std::vector<Bead> B;
    std::vector<int> freeList;
    std::vector<std::vector<int>> sl;   // beads per slice
    std::vector<int> act;               // all active beads
    int head = -1, tail = -1;
    bool worm = false;

    // ---- statistics
    static const int NMOVE = 10;
    long att[NMOVE] = {0}, acc[NMOVE] = {0};

    void init() {
        tau = beta / P;
        Vbox = std::pow(Lbox, DIM);
        rc2 = rc < 1e150 ? rc * rc : 1e300;
        sl.assign(P, {});
    }

    double uni() { return ud(rng); }
    int rint(int n) { int k = (int)(uni() * n); return k >= n ? n - 1 : k; }
    double gauss() { return nd(rng); }

    // ---- physics helpers
    inline double mimg(double x) const { return pbc ? x - Lbox * std::nearbyint(x / Lbox) : x; }
    inline void wrapr(double* r) const { if (pbc) for (int d = 0; d < DIM; d++) r[d] = mimg(r[d]); }
    inline double d2(const double* a, const double* b) const {
        double s = 0;
        for (int d = 0; d < DIM; d++) { double x = mimg(a[d] - b[d]); s += x * x; }
        return s;
    }
    inline double vext(const double* r) const {
        double s = 0;
        if (omega != 0.0) {
            for (int d = 0; d < DIM; d++) s += r[d] * r[d];
            s = 0.5 * omega * omega * s;
        }
        if (ndis) for (int d = 0; d < DIM; d++) s += vdis(r[d]);
        return s;
    }
    // disorder along one axis: Re sum_k c_k z^k, z = exp(2 pi i x/Lbox) (Horner)
    double vdis(double x) const {
        double zr = std::cos(2.0 * M_PI * x / Lbox), zi = std::sin(2.0 * M_PI * x / Lbox), sr = 0, si = 0;
        for (int k = ndis - 1; k >= 0; k--) {
            double tr = sr + dc[2 * k], ti = si + dc[2 * k + 1];
            sr = tr * zr - ti * zi; si = tr * zi + ti * zr;
        }
        return sr;
    }
    static double aziz(double r) {          // HFDHE2, Aziz et al. JCP 70, 4330 (1979); K, Angstrom
        const double eps = 10.8, rm = 2.9673, A = 0.5448504e6, alpha = 13.353384, D = 1.241314;
        const double C6 = 1.3732412, C8 = 0.4253785, C10 = 0.178100;
        double x = r / rm, x2 = x * x, x6 = x2 * x2 * x2;
        double F = x < D ? std::exp(-(D / x - 1.0) * (D / x - 1.0)) : 1.0;
        return eps * (A * std::exp(-alpha * x) - F * (C6 / x6 + C8 / (x6 * x2) + C10 / (x6 * x2 * x2)));
    }
    inline double vpair(double r2) const {
        if (r2 > rc2) return 0.0;
        switch (pot) {
            case 1: { double s6 = sig * sig / r2; s6 = s6 * s6 * s6; return 4.0 * eps * (s6 * s6 - s6); }
            case 2: return aziz(std::sqrt(r2));
            default: return 0.5 * kappa * r2;
        }
    }
    bool pairs() const { return pot == 3 ? false : (pot != 0 || kappa != 0.0); }
    bool linkact() const { return pot == 3; }

    // ---- contact interaction: exact relative propagator of H = -D d^2/dr^2 + g delta(r), D = 2 lam,
    // rho/rho0 = 1 - k sqrt(pi D t) exp(-(|r||r'| + r r')/(2 D t)) erfcx((|r| + |r'| + 2 k D t)/(2 sqrt(D t))),
    // k = g/(2D); the exponential is 1 when the pair crosses (r r' < 0) and exp(-|r||r'|/(D t)) otherwise.
    static double erfcx(double z) {
        if (z < 25.0) return std::exp(z * z) * std::erfc(z);
        double iz2 = 1.0 / (z * z);
        return (1.0 - 0.5 * iz2 * (1.0 - 1.5 * iz2 * (1.0 - 2.5 * iz2))) / (z * std::sqrt(M_PI));
    }
    // g < 0 means g = infinity (hard core, Tonks-Girardeau): rho/rho0 = 1 - exp(-|r||r'|/(D t)), 0 if crossing
    double ucontact(double r, double rp, double t, double g) const {
        if (g < 0) return r * rp > 0 ? -std::log1p(-std::exp(-std::fabs(r * rp) / (2.0 * lam * t))) : INFINITY;
        double D = 2.0 * lam, k = g / (2.0 * D), Dt = D * t, sq = std::sqrt(Dt);
        double ar = std::fabs(r), arp = std::fabs(rp);
        double w = r * rp > 0 ? std::exp(-ar * arp / Dt) : 1.0;
        double f = k * std::sqrt(M_PI) * sq * w * erfcx((ar + arp + 2.0 * k * Dt) / (2.0 * sq));
        return -std::log1p(-f);
    }
    // relative coordinate of the pair at slice s (r) and s+1 (rp), followed continuously along the link
    inline void relpair(const double* a, const double* ap, const double* c, const double* cp, double& r, double& rp) const {
        r = mimg(a[0] - c[0]);
        rp = r + mimg((ap[0] - cp[0]) - (a[0] - c[0]));
    }
    // pair action of the link ra (slice s) -> rb (slice s+1) with every complete link on slice s,
    // except the one starting at bead excl
    double linkU(const double* ra, const double* rb, int s, int excl) const {
        double e = 0, r, rp;
        for (int c : sl[s]) {
            if (c == excl || B[c].next < 0) continue;
            relpair(ra, rb, B[c].r, B[B[c].next].r, r, rp);
            e += ucontact(r, rp, tau, gc);
        }
        return e;
    }
    // same for the existing link that starts at bead b (excluding itself)
    double linkUb(int b) const { return linkU(B[b].r, B[B[b].next].r, B[b].slice, b); }

    // potential felt by a bead at position r on slice s (others on that slice,
    // excluding bead 'excl'); pairs are attributed to the bead being added/removed
    double U(const double* r, int s, int excl) const {
        double e = vext(r);
        if (pairs()) {
            for (int b : sl[s]) if (b != excl) e += vpair(d2(r, B[b].r));
        }
        return e;
    }

    double logG0(double dist2, int m) const {
        double t = m * tau;
        return -0.5 * DIM * std::log(4.0 * M_PI * lam * t) - dist2 / (4.0 * lam * t);
    }
    // with pbc, free propagator over m links summed over periodic images of b
    // (factorises per dimension); without pbc identical to logG0
    static const int NIMG = 2;
    double logG0i(const double* a, const double* b, int m) const {
        if (!pbc) return logG0(d2(a, b), m);
        double t4 = 4.0 * lam * m * tau, s = -0.5 * DIM * std::log(M_PI * t4);
        for (int d = 0; d < DIM; d++) {
            double x = mimg(b[d] - a[d]), z = 0;
            for (int n = -NIMG; n <= NIMG; n++) { double y = x + n * Lbox; z += std::exp(-y * y / t4); }
            s += std::log(z);
        }
        return s;
    }
    // unwrapped target for a bridge a -> b over m links: periodic image chosen
    // with probability proportional to its free propagator (pairs with logG0i)
    void pickImage(const double* a, const double* b, int m, double* bt) {
        if (!pbc) { for (int d = 0; d < DIM; d++) bt[d] = b[d]; return; }
        double t4 = 4.0 * lam * m * tau;
        for (int d = 0; d < DIM; d++) {
            double x = mimg(b[d] - a[d]), w[2 * NIMG + 1], z = 0;
            for (int n = -NIMG; n <= NIMG; n++) { double y = x + n * Lbox; w[n + NIMG] = std::exp(-y * y / t4); z += w[n + NIMG]; }
            double u = uni() * z; int k = 0;
            while (k < 2 * NIMG && u >= w[k]) { u -= w[k]; k++; }
            bt[d] = a[d] + x + (k - NIMG) * Lbox;
        }
    }

    // ---- bead management
    int newBead(const double* r, int s) {
        int id;
        if (!freeList.empty()) { id = freeList.back(); freeList.pop_back(); }
        else { id = (int)B.size(); B.push_back(Bead()); }
        Bead& b = B[id];
        for (int d = 0; d < DIM; d++) b.r[d] = r[d];
        b.slice = s; b.prev = -1; b.next = -1;
        b.posS = (int)sl[s].size(); sl[s].push_back(id);
        b.posA = (int)act.size(); act.push_back(id);
        return id;
    }
    void killBead(int id) {
        int s = B[id].slice, p = B[id].posS;
        int last = sl[s].back(); sl[s][p] = last; B[last].posS = p; sl[s].pop_back();
        p = B[id].posA;
        last = act.back(); act[p] = last; B[last].posA = p; act.pop_back();
        freeList.push_back(id);
    }

    // ---- path sampling
    // Levy bridge from a to the (unwrapped) point b in m steps; out gets the
    // m-1 interior points, wrapped into the box with pbc
    void bridge(const double* a, const double* b, int m, std::vector<Vec>& out) {
        out.resize(m > 0 ? m - 1 : 0);
        Vec prev; for (int d = 0; d < DIM; d++) prev[d] = a[d];
        for (int j = 1; j <= m - 1; j++) {
            int rem = m - j + 1;
            double sd = std::sqrt(2.0 * lam * tau * (double)(m - j) / (double)rem);
            for (int d = 0; d < DIM; d++)
                out[j - 1][d] = prev[d] + (b[d] - prev[d]) / rem + sd * gauss();
            prev = out[j - 1];
        }
        for (auto& v : out) wrapr(v.data());
    }
    // free forward extension by n steps
    void extend(const double* a, int n, std::vector<Vec>& out) {
        out.resize(n);
        Vec prev; for (int d = 0; d < DIM; d++) prev[d] = a[d];
        double sd = std::sqrt(2.0 * lam * tau);
        for (int j = 0; j < n; j++) {
            for (int d = 0; d < DIM; d++) out[j][d] = prev[d] + sd * gauss();
            prev = out[j];
        }
        for (auto& v : out) wrapr(v.data());
    }

    bool accept(double logA) { return logA >= 0.0 || std::log(uni()) < logA; }
    int wrap(int s) const { s %= P; return s < 0 ? s + P : s; }

    // =====================================================================
    // OPEN (Z -> G)
    bool open_move() {
        if (worm) return false;
        int Nb = (int)act.size(); if (Nb == 0) return false;
        att[0]++;
        int b0 = act[rint(Nb)];
        int m = 1 + rint(Lmax);
        std::vector<int> ch(m + 1); ch[0] = b0;
        for (int j = 1; j <= m; j++) ch[j] = B[ch[j - 1]].next;
        double dV = 0;
        for (int j = 1; j <= m - 1; j++) dV += U(B[ch[j]].r, B[ch[j]].slice, ch[j]);
        double dU = 0;
        if (linkact()) for (int j = 0; j < m; j++) dU += linkUb(ch[j]);
        double logA = std::log(C) + std::log((double)Nb) + std::log((double)Lmax)
                    + tau * dV + dU - (m - 1) * mu * tau - logG0i(B[b0].r, B[ch[m]].r, m);
        if (!accept(logA)) return false;
        for (int j = 1; j <= m - 1; j++) killBead(ch[j]);
        B[b0].next = -1; B[ch[m]].prev = -1;
        head = b0; tail = ch[m]; worm = true;
        acc[0]++; return true;
    }
    // CLOSE (G -> Z)
    bool close_move() {
        if (!worm) return false;
        int sI = B[head].slice, sM = B[tail].slice;
        int g = wrap(sM - sI - 1) + 1;           // gap in [1,P]
        if (g > Lmax) return false;
        att[1]++;
        std::vector<Vec> np;
        double bt[DIM]; pickImage(B[head].r, B[tail].r, g, bt);
        bridge(B[head].r, bt, g, np);
        double dV = 0;
        for (int k = 1; k <= g - 1; k++) dV += U(np[k - 1].data(), wrap(sI + k), -1);
        double dU = 0;
        if (linkact())
            for (int k = 0; k < g; k++)
                dU += linkU(k == 0 ? B[head].r : np[k - 1].data(), k == g - 1 ? B[tail].r : np[k].data(), wrap(sI + k), -1);
        int NbAfter = (int)act.size() + (g - 1);
        double logA = logG0i(B[head].r, B[tail].r, g) + (g - 1) * mu * tau - tau * dV - dU
                    - std::log(C) - std::log((double)NbAfter) - std::log((double)Lmax);
        if (!accept(logA)) return false;
        int cur = head;
        for (int k = 1; k <= g - 1; k++) {
            int nb = newBead(np[k - 1].data(), wrap(sI + k));
            B[cur].next = nb; B[nb].prev = cur; cur = nb;
        }
        B[cur].next = tail; B[tail].prev = cur;
        head = tail = -1; worm = false;
        acc[1]++; return true;
    }
    // INSERT (Z -> G)
    bool insert_move() {
        if (worm) return false;
        att[2]++;
        int sM = rint(P), ell = rint(Lmax);       // ell in [0,Lmax-1] links
        Vec r0; for (int d = 0; d < DIM; d++) r0[d] = (uni() - 0.5) * Lbox;
        std::vector<Vec> np; extend(r0.data(), ell, np);
        double dV = U(r0.data(), sM, -1);
        for (int k = 1; k <= ell; k++) dV += U(np[k - 1].data(), wrap(sM + k), -1);
        if (linkact())
            for (int k = 0; k < ell; k++) dV += linkU(k == 0 ? r0.data() : np[k - 1].data(), np[k].data(), wrap(sM + k), -1) / tau;
        double logA = std::log(C) + std::log(Vbox) + std::log((double)P) + std::log((double)Lmax)
                    + (ell + 1) * mu * tau - tau * dV;
        if (!accept(logA)) return false;
        int cur = newBead(r0.data(), sM); tail = cur;
        for (int k = 1; k <= ell; k++) {
            int nb = newBead(np[k - 1].data(), wrap(sM + k));
            B[cur].next = nb; B[nb].prev = cur; cur = nb;
        }
        head = cur; worm = true;
        acc[2]++; return true;
    }
    // REMOVE (G -> Z)
    bool remove_move() {
        if (!worm) return false;
        int b = head, ell = 0;
        while (b != tail) {
            if (ell >= Lmax - 1) return false;       // worm too long to be removed
            b = B[b].prev; ell++;
        }
        for (int d = 0; d < DIM; d++) if (std::fabs(B[tail].r[d]) > 0.5 * Lbox) return false;
        att[3]++;
        double dV = 0; b = head;
        while (true) {
            dV += U(B[b].r, B[b].slice, b);
            if (linkact() && b != tail) dV += linkUb(B[b].prev) / tau;
            if (b == tail) break;
            b = B[b].prev;
        }
        double logA = -(std::log(C) + std::log(Vbox) + std::log((double)P) + std::log((double)Lmax)
                      + (ell + 1) * mu * tau - tau * dV);
        if (!accept(logA)) return false;
        b = head;
        while (true) { int p = B[b].prev; bool last = (b == tail); killBead(b); if (last) break; b = p; }
        head = tail = -1; worm = false;
        acc[3]++; return true;
    }
    // ADVANCE / RECEDE; dir=+1 head, dir=-1 tail
    bool advance_move(int dir) {
        if (!worm) return false;
        int idx = dir > 0 ? 4 : 6;
        att[idx]++;
        int m = 1 + rint(Lmax);
        int e = dir > 0 ? head : tail;
        std::vector<Vec> np; extend(B[e].r, m, np);
        double dV = 0; int s0 = B[e].slice;
        for (int k = 1; k <= m; k++) dV += U(np[k - 1].data(), wrap(s0 + dir * k), -1);
        if (linkact())
            for (int k = 0; k < m; k++) {
                const double* a = k == 0 ? B[e].r : np[k - 1].data();
                if (dir > 0) dV += linkU(a, np[k].data(), wrap(s0 + k), -1) / tau;
                else         dV += linkU(np[k].data(), a, wrap(s0 - k - 1), -1) / tau;
            }
        double logA = m * mu * tau - tau * dV;
        if (!accept(logA)) return false;
        int cur = e;
        for (int k = 1; k <= m; k++) {
            int nb = newBead(np[k - 1].data(), wrap(s0 + dir * k));
            if (dir > 0) { B[cur].next = nb; B[nb].prev = cur; }
            else         { B[cur].prev = nb; B[nb].next = cur; }
            cur = nb;
        }
        if (dir > 0) head = cur; else tail = cur;
        acc[idx]++; return true;
    }
    bool recede_move(int dir) {
        if (!worm) return false;
        int idx = dir > 0 ? 5 : 7;
        int m = 1 + rint(Lmax);
        int e = dir > 0 ? head : tail, other = dir > 0 ? tail : head;
        std::vector<int> rem; int b = e;
        for (int t = 0; t < m; t++) {
            if (b == other) return false;            // not enough links
            rem.push_back(b);
            b = dir > 0 ? B[b].prev : B[b].next;
        }
        att[idx]++;
        double dV = 0;
        for (int x : rem) dV += U(B[x].r, B[x].slice, x);
        if (linkact()) for (int x : rem) dV += linkUb(dir > 0 ? B[x].prev : x) / tau;
        double logA = -m * mu * tau + tau * dV;
        if (!accept(logA)) return false;
        for (int x : rem) killBead(x);
        if (dir > 0) { head = b; B[b].next = -1; } else { tail = b; B[b].prev = -1; }
        acc[idx]++; return true;
    }
    // SWAP (head)
    bool swap_move() {
        if (!worm) return false;
        int m = 1 + rint(Lmax);
        int h = head, s = B[h].slice, sp = wrap(s + m);
        std::vector<int> cand, candQ;
        for (int j : sl[sp]) {
            int p = j; bool ok = true;
            for (int t = 0; t < m; t++) { int pp = B[p].prev; if (pp < 0) { ok = false; break; } p = pp; }
            if (ok) { cand.push_back(j); candQ.push_back(p); }
        }
        if (cand.empty()) return false;
        att[8]++;
        int nc = (int)cand.size();
        std::vector<double> e1(nc);
        double mx = -1e300;
        for (int i = 0; i < nc; i++) { e1[i] = logG0i(B[h].r, B[cand[i]].r, m); mx = std::max(mx, e1[i]); }
        double sum1 = 0; for (int i = 0; i < nc; i++) sum1 += std::exp(e1[i] - mx);
        // choose candidate
        double u = uni() * sum1, cs = 0; int idx = nc - 1;
        for (int i = 0; i < nc; i++) { cs += std::exp(e1[i] - mx); if (u < cs) { idx = i; break; } }
        int j = cand[idx], q = candQ[idx];
        double logsum1 = mx + std::log(sum1);
        // reverse normalisation from q
        double mx2 = -1e300; std::vector<double> e2(nc);
        for (int i = 0; i < nc; i++) { e2[i] = logG0i(B[q].r, B[cand[i]].r, m); mx2 = std::max(mx2, e2[i]); }
        double sum2 = 0; for (int i = 0; i < nc; i++) sum2 += std::exp(e2[i] - mx2);
        double logsum2 = mx2 + std::log(sum2);
        // old interior chain (m-1 beads), oldc[t] at slice s+m-1-t
        std::vector<int> oldc(m > 1 ? m - 1 : 0);
        { int p = j; for (int t = 0; t < m - 1; t++) { p = B[p].prev; oldc[t] = p; } }
        double bt[DIM]; pickImage(B[h].r, B[j].r, m, bt);
        std::vector<Vec> np; bridge(B[h].r, bt, m, np);       // np[k] at slice s+1+k
        double dV = 0;
        for (int k = 0; k < m - 1; k++) {
            int ob = oldc[m - 2 - k]; int sk = wrap(s + 1 + k);
            dV += U(np[k].data(), sk, ob) - U(B[ob].r, sk, ob);
        }
        if (linkact())          // old links O_k -> O_k+1 (O_0 = q, O_m = j), new N_k -> N_k+1 (N_0 = h)
            for (int k = 0; k < m; k++) {
                int Ok = k == 0 ? q : oldc[m - 1 - k];
                const double* Nk = k == 0 ? B[h].r : np[k - 1].data();
                const double* Nk1 = k == m - 1 ? B[j].r : np[k].data();
                dV += (linkU(Nk, Nk1, wrap(s + k), Ok) - linkUb(Ok)) / tau;
            }
        double logA = (logsum1 - logsum2) - tau * dV;
        if (!accept(logA)) return false;
        for (int x : oldc) killBead(x);
        int cur = h;
        for (int k = 0; k < m - 1; k++) {
            int nb = newBead(np[k].data(), wrap(s + 1 + k));
            B[cur].next = nb; B[nb].prev = cur; cur = nb;
        }
        B[cur].next = j; B[j].prev = cur;
        B[q].next = -1; head = q;
        acc[8]++; return true;
    }
    // STAGING of a segment (any sector)
    bool stage_move() {
        int Nb = (int)act.size(); if (Nb == 0 || Lmax < 2) return false;
        int a = act[rint(Nb)], a0 = a;
        int m = 2 + rint(Lmax - 1);                 // 2..Lmax
        std::vector<int> ch(m + 1); ch[0] = a;
        for (int j = 1; j <= m; j++) { int n = B[ch[j - 1]].next; if (n < 0) return false; ch[j] = n; }
        att[9]++;
        // keep the segment's actual end-to-end displacement (winding preserved)
        double bt[DIM]; for (int d = 0; d < DIM; d++) bt[d] = B[a].r[d];
        for (int j = 1; j <= m; j++) for (int d = 0; d < DIM; d++) bt[d] += mimg(B[ch[j]].r[d] - B[ch[j - 1]].r[d]);
        std::vector<Vec> np; bridge(B[a].r, bt, m, np);
        double dV = 0;
        for (int j = 1; j <= m - 1; j++) {
            int x = ch[j]; int sk = B[x].slice;
            dV += U(np[j - 1].data(), sk, x) - U(B[x].r, sk, x);
        }
        if (linkact())
            for (int j = 1; j <= m; j++) {
                const double* a = j == 1 ? B[a0].r : np[j - 2].data();
                const double* b = j == m ? B[ch[m]].r : np[j - 1].data();
                dV += (linkU(a, b, B[ch[j - 1]].slice, ch[j - 1]) - linkUb(ch[j - 1])) / tau;
            }
        if (!accept(-tau * dV)) return false;
        for (int j = 1; j <= m - 1; j++) for (int d = 0; d < DIM; d++) B[ch[j]].r[d] = np[j - 1][d];
        acc[9]++; return true;
    }

    // ---- observables (Z sector)
    enum { O_N, O_V, O_E, O_D2, O_C, O_N2 = O_C + NCYC, O_W1, O_S2, O_SR = O_S2 + NSQ, O_SI = O_SR + NSQ,
           NOBS = O_SI + NSQ };
    void measure(double* o) const {
        double Nb = (double)act.size();
        double e = 0, evx = 0;
        for (int s = 0; s < P; s++) {
            const std::vector<int>& L = sl[s];
            for (size_t i = 0; i < L.size(); i++) {
                double vx = vext(B[L[i]].r); e += vx; evx += vx;
                if (pairs())
                    for (size_t j = i + 1; j < L.size(); j++) e += vpair(d2(B[L[i]].r, B[L[j]].r));
            }
        }
        double du = 0;          // contact: sum over pair links of du/dtau (energy) and du/dg (-> <V_int>)
        if (linkact()) {
            const double ht = 1e-5 * tau, hg = 1e-5 * std::max(gc, 1e-8);
            for (int s = 0; s < P; s++) {
                const std::vector<int>& L = sl[s];
                for (size_t i = 0; i < L.size(); i++) {
                    if (B[L[i]].next < 0) continue;
                    for (size_t j = i + 1; j < L.size(); j++) {
                        if (B[L[j]].next < 0) continue;
                        double r, rp; relpair(B[L[i]].r, B[B[L[i]].next].r, B[L[j]].r, B[B[L[j]].next].r, r, rp);
                        du += (ucontact(r, rp, tau + ht, gc) - ucontact(r, rp, tau - ht, gc)) / (2 * ht);
                        if (gc > 0) e += gc * (ucontact(r, rp, tau, gc + hg) - ucontact(r, rp, tau, gc - hg)) / (2 * hg) / tau;
                    }
                }
            }
        }
        double k2 = 0, D[DIM] = {0};
        for (int id : act) {
            const Bead& b = B[id];
            const Bead& c = B[b.next];
            for (int d = 0; d < DIM; d++) { double x = mimg(c.r[d] - b.r[d]); k2 += x * x; D[d] += x; }
        }
        double D2 = 0; for (int d = 0; d < DIM; d++) D2 += D[d] * D[d];
        // a closed loop of j particles has P*j beads and crosses slice 0 j times:
        // walking it from each of its slice-0 beads adds j to bin j
        for (int k = 0; k < NCYC; k++) o[O_C + k] = 0;
        for (int b0 : sl[0]) {
            int c = b0, len = 0;
            do { c = B[c].next; len++; } while (c != b0);
            o[O_C + std::min(len / P, NCYC) - 1] += 1;
        }
        o[O_N] = Nb / P;
        o[O_V] = e / P;
        // with the contact action e/P = g <sum delta> (Hellmann-Feynman); its energy term is du/P instead
        o[O_E] = (0.5 * DIM * Nb / tau - k2 / (4.0 * lam * tau * tau)) / P + (linkact() ? du / P + evx / P : e / P);
        o[O_D2] = D2;
        o[O_N2] = o[O_N] * o[O_N];
        o[O_W1] = D[0] / Lbox;      // winding number along x (integer with pbc)
        // static structure factor: rho_q(s) = sum_{b in s} z_b^m, z_b = exp(2 pi i x_b / Lbox)
        for (int m = 0; m < NSQ; m++) o[O_S2 + m] = o[O_SR + m] = o[O_SI + m] = 0;
        if (nsq > 0) {
            std::complex<double> rq[NSQ];
            for (int s = 0; s < P; s++) {
                for (int m = 0; m < nsq; m++) rq[m] = 0;
                for (int id : sl[s]) {
                    double ph = 2.0 * M_PI * B[id].r[0] / Lbox;
                    std::complex<double> z(std::cos(ph), std::sin(ph)), w = z;
                    for (int m = 0; m < nsq; m++) { rq[m] += w; w *= z; }
                }
                for (int m = 0; m < nsq; m++) {
                    o[O_S2 + m] += std::norm(rq[m]) / P; o[O_SR + m] += rq[m].real() / P; o[O_SI + m] += rq[m].imag() / P;
                }
            }
        }
    }

    void step() {
        int r = rint(12);
        switch (r) {
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

    // consistency check (debug)
    bool check() const {
        int nb = (int)act.size();
        long cnt = 0; for (auto& v : sl) cnt += v.size();
        if (cnt != nb) return false;
        int nh = 0, nt = 0;
        for (int id : act) {
            const Bead& b = B[id];
            if (b.next >= 0) { if (B[b.next].prev != id || B[b.next].slice != (b.slice + 1) % P) return false; }
            else nh++;
            if (b.prev >= 0) { if (B[b.prev].next != id) return false; }
            else nt++;
        }
        if (worm) { if (nh != 1 || nt != 1 || B[head].next != -1 || B[tail].prev != -1) return false; }
        else if (nh != 0 || nt != 0) return false;
        return true;
    }
};

static double getd(int argc, char** argv, const char* key, double def) {
    size_t n = strlen(key);
    double v = def;                       // the last occurrence wins
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], key, n) && argv[i][n] == '=') v = atof(argv[i] + n + 1);
    return v;
}

// disorder coefficients c_k = Vd sqrt(2/M) exp(i phi_k), phi_k uniform from splitmix64(dseed)
static void disorder(int M, double Vd, unsigned long long seed, std::vector<double>& c) {
    c.resize(2 * M);
    for (int k = 0; k < M; k++) {
        unsigned long long z = (seed += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31;
        double ph = 2.0 * M_PI * (double)(z >> 11) * 0x1p-53, a = Vd * std::sqrt(2.0 / M);
        c[2 * k] = a * std::cos(ph); c[2 * k + 1] = a * std::sin(ph);
    }
}

int main(int argc, char** argv) {
    Sim S;
    S.P = (int)getd(argc, argv, "P", 24);
    S.beta = getd(argc, argv, "beta", 1.0);
    S.mu = getd(argc, argv, "mu", 0.0);
    S.C = getd(argc, argv, "C", 1.0);
    S.Lmax = (int)getd(argc, argv, "Lmax", 8);
    S.Lbox = getd(argc, argv, "Lbox", 12.0);
    S.omega = getd(argc, argv, "omega", 1.0);
    S.kappa = getd(argc, argv, "kappa", 0.0);
    S.lam = getd(argc, argv, "lam", 0.5);
    S.pbc = (int)getd(argc, argv, "pbc", 0);
    S.pot = (int)getd(argc, argv, "pot", 0);
    S.eps = getd(argc, argv, "eps", 1.0);
    S.sig = getd(argc, argv, "sig", 1.0);
    S.rc = getd(argc, argv, "rc", S.pbc ? 0.5 * S.Lbox : 1e300);
    S.gc = getd(argc, argv, "g", 0.0);
    if (S.pot == 3 && DIM != 1) { fprintf(stderr, "pot=3 (contact) needs DIM=1\n"); return 1; }
    double Vd = getd(argc, argv, "Vd", 0.0), qc = getd(argc, argv, "qc", 2.0);
    unsigned long long dseed = (unsigned long long)getd(argc, argv, "dseed", 1);
    S.nsq = (int)getd(argc, argv, "nsq", 0);
    if (S.nsq > NSQ || (S.nsq > 0 && !S.pbc)) { fprintf(stderr, "need nsq <= NSQ = %d and pbc=1\n", NSQ); return 1; }
    if (Vd != 0.0) {
        S.ndis = (int)std::floor(qc * S.Lbox / (2.0 * M_PI));
        if (S.ndis < 1) { fprintf(stderr, "need qc >= 2 pi / Lbox\n"); return 1; }
        disorder(S.ndis, Vd, dseed, S.dc);
    }
    double nsteps = getd(argc, argv, "steps", 2e7);
    double ntherm = getd(argc, argv, "therm", 2e6);
    int meas = (int)getd(argc, argv, "meas", 20);
    int nblocks = (int)getd(argc, argv, "blocks", 20);
    int debug = (int)getd(argc, argv, "debug", 0);
    S.rng.seed((unsigned long long)getd(argc, argv, "seed", 12345));
    if (S.Lmax > S.P - 2) { fprintf(stderr, "need Lmax <= P-2\n"); return 1; }
    if (S.pbc && S.rc > 0.5 * S.Lbox * (1 + 1e-12)) { fprintf(stderr, "need rc <= Lbox/2 with pbc\n"); return 1; }
    S.init();
    // optional initial state: N0 straight closed worldlines on a simple lattice
    int N0 = (int)getd(argc, argv, "N0", 0);
    if (N0 > 0) {
        int ns = (int)std::ceil(std::pow((double)N0, 1.0 / DIM) - 1e-9);
        double a = S.Lbox / ns;
        for (int i = 0; i < N0; i++) {
            double x[DIM]; int c = i;
            for (int d = 0; d < DIM; d++) { x[d] = -0.5 * S.Lbox + (c % ns + 0.5) * a; c /= ns; }
            int first = S.newBead(x, 0), cur = first;
            for (int s = 1; s < S.P; s++) { int nb = S.newBead(x, s); S.B[cur].next = nb; S.B[nb].prev = cur; cur = nb; }
            S.B[cur].next = first; S.B[first].prev = cur;
        }
    }

    long total = (long)nsteps, th = (long)ntherm;
    long blockLen = total / nblocks;
    const int NO = Sim::NOBS;
    std::vector<std::array<double, Sim::NOBS>> bs(nblocks);
    for (auto& a : bs) a.fill(0.0);
    std::vector<long> bn(nblocks, 0);
    long nZ = 0, nAll = 0;
    std::map<long, long> histN, histW;      // Z-sector histograms of N and of the x winding
    auto t0 = std::chrono::steady_clock::now();
    double tth = 0;
    for (long it = 0; it < th + total; it++) {
        if (it == th) tth = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        S.step();
        if (debug && it % 1000 == 0 && !S.check()) { fprintf(stderr, "INCONSISTENT at %ld\n", it); return 2; }
        if (it >= th && ((it - th) % meas == 0)) {
            nAll++;
            if (!S.worm) {
                nZ++;
                double o[Sim::NOBS]; S.measure(o);
                int b = (int)std::min<long>((it - th) / blockLen, nblocks - 1);
                for (int k = 0; k < NO; k++) bs[b][k] += o[k];
                histN[std::lround(o[Sim::O_N])]++;
                if (S.pbc) histW[std::lround(o[Sim::O_W1])]++;
                bn[b]++;
            }
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // jackknife over blocks; derived: rho_s/rho = <D2>/(2 lam d beta <N>), <dN^2>,
    // <W^2> = <D2>/(d Lbox^2), K = pi sqrt(<W^2> <dN^2>)
    enum { Q_RS = NO, Q_DN2, Q_W2, Q_K, Q_S = Q_K + 1, Q_SC = Q_S + NSQ, NQ = Q_SC + NSQ };
    auto derive = [&](const double* m, double* q) {
        for (int k = 0; k < NO; k++) q[k] = m[k];
        q[Q_RS] = m[Sim::O_N] > 0 ? m[Sim::O_D2] / (2.0 * S.lam * DIM * S.beta * m[Sim::O_N]) : 0.0;
        q[Q_DN2] = m[Sim::O_N2] - m[Sim::O_N] * m[Sim::O_N];
        q[Q_W2] = m[Sim::O_D2] / (DIM * S.Lbox * S.Lbox);
        q[Q_K] = M_PI * std::sqrt(std::max(0.0, q[Q_W2] * q[Q_DN2]));
        for (int k = 0; k < NSQ; k++) {
            double n = m[Sim::O_N] > 0 ? m[Sim::O_N] : 1.0;
            q[Q_S + k] = m[Sim::O_S2 + k] / n;
            q[Q_SC + k] = (m[Sim::O_S2 + k] - m[Sim::O_SR + k] * m[Sim::O_SR + k] - m[Sim::O_SI + k] * m[Sim::O_SI + k]) / n;
        }
    };
    double tot[NO] = {0}; long totn = 0; int nb = 0;
    for (int b = 0; b < nblocks; b++) if (bn[b] > 0) { nb++; totn += bn[b]; for (int k = 0; k < NO; k++) tot[k] += bs[b][k]; }
    double m[NO], mean[NQ], err[NQ] = {0};
    for (int k = 0; k < NO; k++) m[k] = tot[k] / totn;
    derive(m, mean);
    for (int b = 0; b < nblocks; b++) if (bn[b] > 0) {
        double mj[NO], qj[NQ];
        for (int k = 0; k < NO; k++) mj[k] = (tot[k] - bs[b][k]) / (totn - bn[b]);
        derive(mj, qj);
        for (int k = 0; k < NQ; k++) err[k] += (qj[k] - mean[k]) * (qj[k] - mean[k]);
    }
    for (int k = 0; k < NQ; k++) err[k] = std::sqrt(err[k] * (nb - 1.0) / nb);
    const char* names[10] = {"open", "close", "insert", "remove", "advH", "recH", "advT", "recT", "swap", "stage"};
    printf("DIM=%d P=%d beta=%g mu=%g C=%g Lmax=%d Lbox=%g omega=%g kappa=%g lam=%g pbc=%d pot=%d eps=%g sig=%g rc=%g g=%g steps=%.3g\n",
           DIM, S.P, S.beta, S.mu, S.C, S.Lmax, S.Lbox, S.omega, S.kappa, S.lam, S.pbc, S.pot, S.eps, S.sig, S.rc, S.gc, nsteps);
    printf("RESULT N= %.5f +- %.5f   Vbar= %.5f +- %.5f   Zfrac= %.4f   time=%.1fs\n",
           mean[Sim::O_N], err[Sim::O_N], mean[Sim::O_V], err[Sim::O_V], nAll ? (double)nZ / nAll : 0.0, secs);
    printf("TIMING therm=%.2fs prod=%.2fs prod_rate=%.4g steps/s\n", tth, secs - tth, total / (secs - tth));
    printf("RESULT2 E= %.5f +- %.5f   D2= %.5f +- %.5f   rhos= %.5f +- %.5f\n",
           mean[Sim::O_E], err[Sim::O_E], mean[Sim::O_D2], err[Sim::O_D2], mean[Q_RS], err[Q_RS]);
    printf("RESULT3 dN2= %.5f +- %.5f   W2= %.5e +- %.2e   K= %.5f +- %.5f\n",
           mean[Q_DN2], err[Q_DN2], mean[Q_W2], err[Q_W2], mean[Q_K], err[Q_K]);
    if (S.ndis) printf("DISORDER Vd=%g qc=%g dseed=%llu modes=%d\n", Vd, qc, dseed, S.ndis);
    printf("CYCLES");
    for (int k = 0; k < NCYC; k++)
        printf(" n%d%s= %.7f +- %.7f", k + 1, k == NCYC - 1 ? "+" : "", mean[Sim::O_C + k], err[Sim::O_C + k]);
    printf("\n");
    for (int k = 0; k < S.nsq; k++)
        printf("SQ m= %d q= %.6f S= %.6f +- %.6f Sc= %.6f +- %.6f rho2= %.6f +- %.6f\n", k + 1, 2 * M_PI * (k + 1) / S.Lbox,
               mean[Q_S + k], err[Q_S + k], mean[Q_SC + k], err[Q_SC + k], mean[Sim::O_S2 + k], err[Sim::O_S2 + k]);
    // histograms for the discrete-Gaussian (theta-function) form of the Luttinger zero modes
    printf("HISTN"); for (auto& h : histN) printf(" %ld:%ld", h.first, h.second); printf("\n");
    if (S.pbc) { printf("HISTW"); for (auto& h : histW) printf(" %ld:%ld", h.first, h.second); printf("\n"); }
    printf("acceptance:");
    for (int i = 0; i < 10; i++) printf(" %s=%.3f(%ld)", names[i], S.att[i] ? (double)S.acc[i] / S.att[i] : 0.0, S.att[i]);
    printf("\n");
    return 0;
}
