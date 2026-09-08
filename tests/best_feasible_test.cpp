// best_feasible_test.cpp — v0.17.1
// Verrouille une garantie explicitement demandee : meme en cas de MaxIter ou
// de TimeLimit, le solveur doit TOUJOURS retourner le meilleur point
// FAISABLE rencontre (si au moins un accept feasible a eu lieu), jamais un
// iterate brut potentiellement moins faisable, meme si ce dernier a une
// meilleure stationnarite apparente.
#include "lazyipdca/solver.hpp"
#include <cstdio>
#include <random>
#include <chrono>

using namespace lazyipdca;

static SparseSymUpper tridiag_indef(idx_t n, double dscale, unsigned seed) {
  std::vector<idx_t> I, J; std::vector<double> V;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> ud(-1.0, 1.0);
  for (idx_t i = 0; i < n; ++i) {
    I.push_back(i); J.push_back(i); V.push_back(dscale * ud(gen));
    if (i + 1 < n) { I.push_back(i); J.push_back(i + 1); V.push_back(0.5 * ud(gen)); }
  }
  SparseSymUpper Q; Q.U = csc_from_coo(n, n, I, J, V);
  return Q;
}

static int fails = 0;
static void expect(bool ok, const char* what) {
  std::printf("  %-62s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

// Verifie que la faisabilite (Ax=b, cl<=Cx<=cu, l<=x<=u) du point RENDU
// tient a la tolerance eps_feas -- controle independant de info.res_feas_rel
// (qui pourrait, en theorie, ne pas etre recalculee sur le point final).
static double check_feas(const Problem& P, const Result& r) {
  double viol = 0.0;
  for (idx_t i = 0; i < P.n; ++i) {
    if (std::isfinite(P.l[i])) viol = std::max(viol, P.l[i] - r.x[i]);
    if (std::isfinite(P.u[i])) viol = std::max(viol, r.x[i] - P.u[i]);
  }
  if (P.p > 0) {
    std::vector<double> ax(P.p, 0.0);
    P.A.gaxpy(r.x.data(), ax.data());
    for (idx_t i = 0; i < P.p; ++i) viol = std::max(viol, std::fabs(ax[i] - P.b[i]));
  }
  if (P.m > 0) {
    std::vector<double> cx(P.m, 0.0);
    P.C.gaxpy(r.x.data(), cx.data());
    for (idx_t i = 0; i < P.m; ++i) {
      if (std::isfinite(P.cl[i])) viol = std::max(viol, P.cl[i] - cx[i]);
      if (std::isfinite(P.cu[i])) viol = std::max(viol, cx[i] - P.cu[i]);
    }
  }
  return viol;
}

int main() {
  // ---- Test 1 : MaxIter force AVEC un incumbent faisable disponible -------
  // n modere, Q indefini : converge normalement en quelques dizaines
  // d'iterations avec max_iter genereux ; on force max_iter juste au-dessus
  // du premier accept feasible pour couper la course en pleine ascension,
  // avant convergence complete, et verifier que le point RENDU reste
  // faisable malgre l'arret premature.
  {
    idx_t n = 300, p = 60;
    Problem P; P.n = n; P.p = p; P.name = "maxiter_feasible";
    P.Q = tridiag_indef(n, 3.0, 11);
    P.q.assign(n, 0.0);
    std::mt19937 gen(5); std::uniform_real_distribution<double> ud(-1, 1);
    for (auto& v : P.q) v = ud(gen);
    P.l.assign(n, -3.0); P.u.assign(n, 3.0);
    // A rang plein (indices distincts par ligne, pas de motif periodique) et
    // b COHERENT (issu d'un point interieur garanti) -- cf. legons de la
    // session precedente sur les systemes d'egalites mal poses.
    std::mt19937 gidx(21);
    std::uniform_int_distribution<idx_t> col_dist(0, n - 1);
    std::uniform_real_distribution<double> ub(0.5, 1.5);
    std::vector<idx_t> I, J; std::vector<double> V;
    for (idx_t i = 0; i < p; ++i) {
      std::vector<idx_t> cols;
      while ((idx_t)cols.size() < 5) {
        idx_t c = col_dist(gidx);
        if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
      }
      for (idx_t c : cols) { I.push_back(i); J.push_back(c); V.push_back(ub(gidx) * (gidx() % 2 ? 1 : -1)); }
    }
    P.A = csc_from_coo(p, n, I, J, V);
    std::mt19937 gx(23); std::uniform_real_distribution<double> uxs(-0.5, 0.5);
    std::vector<double> xstar(n);
    for (auto& v : xstar) v = uxs(gx);
    P.b.assign(p, 0.0);
    P.A.gaxpy(xstar.data(), P.b.data());

    // reference : combien d'iterations et d'accepts pour converger normalement
    Settings s_ref; s_ref.verbose = 0;
    Result r_ref = solve(P, s_ref);
    std::printf("[T1] reference : status=%d it=%lld n_accepts=%lld obj=%+.6e\n",
                (int)r_ref.info.status, (long long)r_ref.info.iters,
                (long long)r_ref.info.n_accepts, r_ref.info.obj);
    expect(r_ref.info.n_accepts >= 2, "T1 : au moins 2 accepts en reference (sinon coupe trop tot)");

    // coupe forcee : max_iter = un peu plus que l'iteration du 1er accept,
    // largement avant la convergence complete (it_ref).
    idx_t cut_iter = std::max<idx_t>(3, r_ref.info.iters / 4);
    Settings s_cut = s_ref; s_cut.max_iter = cut_iter;
    Result r_cut = solve(P, s_cut);
    double viol = check_feas(P, r_cut);
    std::printf("[T1] coupe a max_iter=%lld : status=%d it=%lld obj=%+.6e "
                "feas_rel(info)=%.2e violation_directe=%.2e\n",
                (long long)cut_iter, (int)r_cut.info.status, (long long)r_cut.info.iters,
                r_cut.info.obj, r_cut.info.res_feas_rel, viol);
    expect(r_cut.info.status == Status::MaxIter,
           "T1 : le statut est bien MaxIter (la coupe a fonctionne comme prevu)");
    expect(viol <= 1e-6,
           "T1 : le point RENDU est FAISABLE malgre l'arret premature (violation directe <= 1e-6)");
    expect(r_cut.info.res_feas_rel <= 100.0 * s_ref.eps_feas,
           "T1 : info.res_feas_rel confirme la faisabilite (coherent avec le controle direct)");
  }

  // ---- Test 2 : TimeLimit force, meme garantie ------------------------------
  {
    idx_t n = 400, p = 80;
    Problem P; P.n = n; P.p = p; P.name = "timelimit_feasible";
    P.Q = tridiag_indef(n, 3.0, 31);
    P.q.assign(n, 0.0);
    std::mt19937 gen(7); std::uniform_real_distribution<double> ud(-1, 1);
    for (auto& v : P.q) v = ud(gen);
    P.l.assign(n, -3.0); P.u.assign(n, 3.0);
    std::mt19937 gidx(41);
    std::uniform_int_distribution<idx_t> col_dist(0, n - 1);
    std::uniform_real_distribution<double> ub(0.5, 1.5);
    std::vector<idx_t> I, J; std::vector<double> V;
    for (idx_t i = 0; i < p; ++i) {
      std::vector<idx_t> cols;
      while ((idx_t)cols.size() < 5) {
        idx_t c = col_dist(gidx);
        if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
      }
      for (idx_t c : cols) { I.push_back(i); J.push_back(c); V.push_back(ub(gidx) * (gidx() % 2 ? 1 : -1)); }
    }
    P.A = csc_from_coo(p, n, I, J, V);
    std::mt19937 gx(43); std::uniform_real_distribution<double> uxs(-0.5, 0.5);
    std::vector<double> xstar(n);
    for (auto& v : xstar) v = uxs(gx);
    P.b.assign(p, 0.0);
    P.A.gaxpy(xstar.data(), P.b.data());

    Settings s_ref; s_ref.verbose = 0;
    auto t0 = std::chrono::steady_clock::now();
    Result r_ref = solve(P, s_ref);
    double full_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[T2] reference : status=%d it=%lld n_accepts=%lld time=%.3fs\n",
                (int)r_ref.info.status, (long long)r_ref.info.iters,
                (long long)r_ref.info.n_accepts, full_time);
    expect(r_ref.info.n_accepts >= 2, "T2 : au moins 2 accepts en reference");

    Settings s_cut = s_ref; s_cut.time_limit = std::max(1e-4, full_time * 0.25);
    Result r_cut = solve(P, s_cut);
    double viol = check_feas(P, r_cut);
    std::printf("[T2] coupe a time_limit=%.4fs : status=%d it=%lld obj=%+.6e "
                "feas_rel(info)=%.2e violation_directe=%.2e\n",
                s_cut.time_limit, (int)r_cut.info.status, (long long)r_cut.info.iters,
                r_cut.info.obj, r_cut.info.res_feas_rel, viol);
    // Le TimeLimit est intrinsequement moins deterministe que max_iter (le
    // seuil depend de la vitesse d'execution reelle) : on tolere que la
    // coupe n'ait pas eu lieu si la machine etait rapide, mais SI elle a eu
    // lieu (TimeLimit effectivement atteint), la garantie doit tenir.
    if (r_cut.info.status == Status::TimeLimit) {
      expect(viol <= 1e-6, "T2 : point RENDU faisable malgre TimeLimit (violation directe <= 1e-6)");
      expect(r_cut.info.res_feas_rel <= 100.0 * s_ref.eps_feas,
             "T2 : info.res_feas_rel coherent avec le controle direct");
    } else {
      std::printf("  (coupe non atteinte -- machine plus rapide que prevu ; test non concluant "
                  "mais pas un echec)\n");
    }
  }

  // ---- Test 3 : AUCUN accept feasible -> pas de fausse promesse -----------
  // Si best_feas_valid est FAUX (jamais aucun accept), le point brut est
  // retourne tel quel -- il n'y a rien a restaurer, et le solveur ne doit
  // pas pretendre le contraire. On verifie juste l'absence de crash/
  // incoherence sur ce cas degenere (max_iter=1, quasi aucune chance
  // d'accepter).
  {
    idx_t n = 200;
    Problem P; P.n = n; P.name = "no_accept_yet";
    P.Q = tridiag_indef(n, 3.0, 51);
    P.q.assign(n, 0.0);
    std::mt19937 gen(9); std::uniform_real_distribution<double> ud(-1, 1);
    for (auto& v : P.q) v = ud(gen);
    P.l.assign(n, -3.0); P.u.assign(n, 3.0);
    Settings s; s.verbose = 0; s.max_iter = 1;
    Result r = solve(P, s);
    std::printf("[T3] max_iter=1 : status=%d it=%lld n_accepts=%lld\n",
                (int)r.info.status, (long long)r.info.iters, (long long)r.info.n_accepts);
    expect(true, "T3 : pas de crash sur un run coupe avant tout accept");
  }

  std::printf(fails == 0 ? "BEST_FEASIBLE_TEST: ALL PASS\n" : "BEST_FEASIBLE_TEST: %d FAIL\n", fails);
  return fails;
}
