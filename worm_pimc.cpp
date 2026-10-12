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
// Build:  g++ -O2 -std=c++17 -DDIM=1 worm_pimc.cpp -o worm_pimc_1d
//
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <array>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>

#ifndef DIM
#define DIM 1
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
    const double lam = 0.5;
    double Vbox = 1.0;

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
        sl.assign(P, {});
    }

    double uni() { return ud(rng); }
    int rint(int n) { int k = (int)(uni() * n); return k >= n ? n - 1 : k; }
    double gauss() { return nd(rng); }

    // ---- physics helpers
    inline double d2(const double* a, const double* b) const {
        double s = 0;
        for (int d = 0; d < DIM; d++) { double x = a[d] - b[d]; s += x * x; }
        return s;
    }
    inline double vext(const double* r) const {
        double s = 0;
        for (int d = 0; d < DIM; d++) s += r[d] * r[d];
        return 0.5 * omega * omega * s;
    }
    inline double vpair(double r2) const { return 0.5 * kappa * r2; }

    // potential felt by a bead at position r on slice s (others on that slice,
    // excluding bead 'excl'); pairs are attributed to the bead being added/removed
    double U(const double* r, int s, int excl) const {
        double e = vext(r);
        if (kappa != 0.0) {
            for (int b : sl[s]) if (b != excl) e += vpair(d2(r, B[b].r));
        }
        return e;
    }

    double logG0(double dist2, int m) const {
        double t = m * tau;
        return -0.5 * DIM * std::log(4.0 * M_PI * lam * t) - dist2 / (4.0 * lam * t);
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
    // Levy bridge from a to b in m steps; out gets the m-1 interior points
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
        double logA = std::log(C) + std::log((double)Nb) + std::log((double)Lmax)
                    + tau * dV - (m - 1) * mu * tau - logG0(d2(B[b0].r, B[ch[m]].r), m);
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
        bridge(B[head].r, B[tail].r, g, np);
        double dV = 0;
        for (int k = 1; k <= g - 1; k++) dV += U(np[k - 1].data(), wrap(sI + k), -1);
        int NbAfter = (int)act.size() + (g - 1);
        double logA = logG0(d2(B[head].r, B[tail].r), g) + (g - 1) * mu * tau - tau * dV
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
        double den = 4.0 * lam * m * tau;
        std::vector<double> e1(nc);
        double mx = -1e300;
        for (int i = 0; i < nc; i++) { e1[i] = -d2(B[h].r, B[cand[i]].r) / den; mx = std::max(mx, e1[i]); }
        double sum1 = 0; for (int i = 0; i < nc; i++) sum1 += std::exp(e1[i] - mx);
        // choose candidate
        double u = uni() * sum1, cs = 0; int idx = nc - 1;
        for (int i = 0; i < nc; i++) { cs += std::exp(e1[i] - mx); if (u < cs) { idx = i; break; } }
        int j = cand[idx], q = candQ[idx];
        double logsum1 = mx + std::log(sum1);
        // reverse normalisation from q
        double mx2 = -1e300; std::vector<double> e2(nc);
        for (int i = 0; i < nc; i++) { e2[i] = -d2(B[q].r, B[cand[i]].r) / den; mx2 = std::max(mx2, e2[i]); }
        double sum2 = 0; for (int i = 0; i < nc; i++) sum2 += std::exp(e2[i] - mx2);
        double logsum2 = mx2 + std::log(sum2);
        // old interior chain (m-1 beads), oldc[t] at slice s+m-1-t
        std::vector<int> oldc(m > 1 ? m - 1 : 0);
        { int p = j; for (int t = 0; t < m - 1; t++) { p = B[p].prev; oldc[t] = p; } }
        std::vector<Vec> np; bridge(B[h].r, B[j].r, m, np);   // np[k] at slice s+1+k
        double dV = 0;
        for (int k = 0; k < m - 1; k++) {
            int ob = oldc[m - 2 - k]; int sk = wrap(s + 1 + k);
            dV += U(np[k].data(), sk, ob) - U(B[ob].r, sk, ob);
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
        int a = act[rint(Nb)];
        int m = 2 + rint(Lmax - 1);                 // 2..Lmax
        std::vector<int> ch(m + 1); ch[0] = a;
        for (int j = 1; j <= m; j++) { int n = B[ch[j - 1]].next; if (n < 0) return false; ch[j] = n; }
        att[9]++;
        std::vector<Vec> np; bridge(B[a].r, B[ch[m]].r, m, np);
        double dV = 0;
        for (int j = 1; j <= m - 1; j++) {
            int x = ch[j]; int sk = B[x].slice;
            dV += U(np[j - 1].data(), sk, x) - U(B[x].r, sk, x);
        }
        if (!accept(-tau * dV)) return false;
        for (int j = 1; j <= m - 1; j++) for (int d = 0; d < DIM; d++) B[ch[j]].r[d] = np[j - 1][d];
        acc[9]++; return true;
    }

    // ---- observables (Z sector): N and mean-over-slices potential energy
    void measure(double& Nn, double& Vbar) const {
        Nn = act.size() / (double)P;
        double e = 0;
        for (int s = 0; s < P; s++) {
            const std::vector<int>& L = sl[s];
            for (size_t i = 0; i < L.size(); i++) {
                e += vext(B[L[i]].r);
                if (kappa != 0.0)
                    for (size_t j = i + 1; j < L.size(); j++) e += vpair(d2(B[L[i]].r, B[L[j]].r));
            }
        }
        Vbar = e / P;
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
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return atof(argv[i] + n + 1);
    return def;
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
    double nsteps = getd(argc, argv, "steps", 2e7);
    double ntherm = getd(argc, argv, "therm", 2e6);
    int meas = (int)getd(argc, argv, "meas", 20);
    int nblocks = (int)getd(argc, argv, "blocks", 20);
    int debug = (int)getd(argc, argv, "debug", 0);
    S.rng.seed((unsigned long long)getd(argc, argv, "seed", 12345));
    if (S.Lmax > S.P - 2) { fprintf(stderr, "need Lmax <= P-2\n"); return 1; }
    S.init();

    long total = (long)nsteps, th = (long)ntherm;
    long blockLen = total / nblocks;
    std::vector<double> bN(nblocks, 0), bV(nblocks, 0), bN2(nblocks, 0);
    std::vector<long> bn(nblocks, 0);
    long nZ = 0, nAll = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (long it = 0; it < th + total; it++) {
        S.step();
        if (debug && it % 1000 == 0 && !S.check()) { fprintf(stderr, "INCONSISTENT at %ld\n", it); return 2; }
        if (it >= th && ((it - th) % meas == 0)) {
            nAll++;
            if (!S.worm) {
                nZ++;
                double Nn, Vb; S.measure(Nn, Vb);
                int b = (int)std::min<long>((it - th) / blockLen, nblocks - 1);
                bN[b] += Nn; bV[b] += Vb; bN2[b] += Nn * Nn; bn[b]++;
            }
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // block statistics
    double mN = 0, mV = 0; int nb = 0;
    std::vector<double> aN, aV;
    for (int b = 0; b < nblocks; b++) if (bn[b] > 0) { aN.push_back(bN[b] / bn[b]); aV.push_back(bV[b] / bn[b]); }
    nb = (int)aN.size();
    for (int b = 0; b < nb; b++) { mN += aN[b]; mV += aV[b]; }
    mN /= nb; mV /= nb;
    double eN = 0, eV = 0;
    for (int b = 0; b < nb; b++) { eN += (aN[b] - mN) * (aN[b] - mN); eV += (aV[b] - mV) * (aV[b] - mV); }
    eN = std::sqrt(eN / (nb * (nb - 1.0))); eV = std::sqrt(eV / (nb * (nb - 1.0)));
    const char* names[10] = {"open", "close", "insert", "remove", "advH", "recH", "advT", "recT", "swap", "stage"};
    printf("DIM=%d P=%d beta=%g mu=%g C=%g Lmax=%d Lbox=%g omega=%g kappa=%g steps=%.3g\n",
           DIM, S.P, S.beta, S.mu, S.C, S.Lmax, S.Lbox, S.omega, S.kappa, nsteps);
    printf("RESULT N= %.5f +- %.5f   Vbar= %.5f +- %.5f   Zfrac= %.4f   time=%.1fs\n",
           mN, eN, mV, eV, nAll ? (double)nZ / nAll : 0.0, secs);
    printf("acceptance:");
    for (int i = 0; i < 10; i++) printf(" %s=%.3f(%ld)", names[i], S.att[i] ? (double)S.acc[i] / S.att[i] : 0.0, S.att[i]);
    printf("\n");
    return 0;
}
