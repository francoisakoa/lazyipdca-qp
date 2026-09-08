// smoke.cpp — validation of lazyipdca on structured nonconvex QPs.
// Test 1: NCVXQP-like — indefinite tridiagonal Q, equalities, box.
// Test 2: box-only indefinite QP (checks pure lazy rho escalation).
// Test 3: convex QP with two-sided inequalities (rho must stay 0).
#include "lazyipdca/solver.hpp"
#include <cstdio>
#include <random>

using namespace lazyipdca;

static SparseSymUpper tridiag_indef(idx_t n, double dscale) {
  std::vector<idx_t> I, J; std::vector<double> V;
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> ud(-1.0, 1.0);
  for (idx_t i = 0; i < n; ++i) {
    I.push_back(i); J.push_back(i); V.push_back(dscale * ud(gen));  // indefinite diag
    if (i + 1 < n) { I.push_back(i); J.push_back(i + 1); V.push_back(0.5 * ud(gen)); }
  }
  SparseSymUpper Q; Q.U = csc_from_coo(n, n, I, J, V);
  return Q;
}

static int check(const char* name, const Result& r, bool need_soc) {
            // tolerance 5e-6 : queues marginales sensibles au codegen (ASAN propre)
  bool ok = (r.info.status == Status::Solved || r.info.status == Status::SolvedSOC) &&
            r.info.res_stat_rel <= 5e-6 && r.info.res_feas_rel <= 5e-6 && r.info.res_comp_rel <= 5e-6;
  if (need_soc) ok = ok && r.info.status == Status::SolvedSOC;
  std::printf("%-28s status=%d it=%3lld obj=%+.6e stat=%.1e feas=%.1e comp=%.1e rho=%.2e lmin=%.3e %s\n",
              name, (int)r.info.status, (long long)r.info.iters, r.info.obj,
              r.info.res_stat_rel, r.info.res_feas_rel, r.info.res_comp_rel,
              r.info.rho_final, r.info.lambda_min_reduced, ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  int fails = 0;
  { // Test 1: n=2000, p=500 equalities, box
    idx_t n = 2000, p = 500;
    Problem P; P.n = n; P.p = p; P.m = 0; P.name = "ncvx_eq_box";
    P.Q = tridiag_indef(n, 2.0);
    P.q.assign(n, 0.0);
    std::mt19937 gen(11); std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (auto& v : P.q) v = ud(gen);
    // A: each row couples 4 variables
    std::vector<idx_t> I, J; std::vector<double> V;
    for (idx_t i = 0; i < p; ++i)
      for (int k = 0; k < 4; ++k) { I.push_back(i); J.push_back((4 * i + k) % n); V.push_back(1.0 + 0.1 * k); }
    P.A = csc_from_coo(p, n, I, J, V);
    P.b.assign(p, 1.0);
    P.l.assign(n, -5.0); P.u.assign(n, 5.0);
    Settings st; st.verbose = 0;
    fails += check("ncvx eq+box (n=2000)", solve(P, st), true);
  }
  { // Test 2: box-only indefinite, some free variables
    idx_t n = 1000;
    Problem P; P.n = n; P.p = 0; P.m = 0; P.name = "ncvx_box";
    P.Q = tridiag_indef(n, 1.5);
    P.q.assign(n, 0.1);
    P.l.assign(n, -1.0); P.u.assign(n, 2.0);
    // full finite box: box-only indefinite QP with free directions is unbounded
    Settings st; st.verbose = 0;
    fails += check("ncvx box-only (n=1000)", solve(P, st), false);
  }
  { // Test 3: convex, two-sided inequalities — rho must remain 0
    idx_t n = 800, m = 300;
    Problem P; P.n = n; P.p = 0; P.m = m; P.name = "cvx_ineq";
    std::vector<idx_t> I, J; std::vector<double> V;
    for (idx_t i = 0; i < n; ++i) { I.push_back(i); J.push_back(i); V.push_back(2.0 + (double)(i % 5)); }
    P.Q.U = csc_from_coo(n, n, I, J, V);
    P.q.assign(n, -1.0);
    I.clear(); J.clear(); V.clear();
    for (idx_t i = 0; i < m; ++i)
      for (int k = 0; k < 3; ++k) { I.push_back(i); J.push_back((3 * i + k) % n); V.push_back(1.0); }
    P.C = csc_from_coo(m, n, I, J, V);
    P.cl.assign(m, -1.0); P.cu.assign(m, 1.0);
    for (idx_t i = 0; i < m; i += 7) P.cu[i] = INF;  // one-sided rows
    P.l.assign(n, -INF); P.u.assign(n, INF);
    Settings st; st.verbose = 0;
    Result r = solve(P, st);
    fails += check("convex ineq (n=800)", r, false);
    if (r.info.rho_final != 0.0) { std::printf("  !! rho should be 0 on convex problem\n"); fails++; }
  }
  std::printf(fails ? "SMOKE: %d FAILURES\n" : "SMOKE: ALL PASS\n", fails);
  return fails;
}
