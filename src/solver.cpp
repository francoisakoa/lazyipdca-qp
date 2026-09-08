// lazyipdca — solver.cpp
// Single-loop IPDCA (one Newton step of the DCA-convexified barrier problem
// per iteration, as validated in the paper: "making only one iteration in
// Algorithm 1, as in LOQO, improves the speed of the overall algorithm").
//
// Nesting (integrated Algorithm 1, three levels):
//   L1  barrier continuation        mu_{k+1} = r * gap / n_comp
//   L2  DCA linearization           h(x) = rho/2 ||x||^2 linearized at x_k
//                                   => Hessian model Q + rho I, gradient Qx+q
//   L3  Newton on the quasidefinite KKT system, factorized by MUMPS+METIS.
//
// Lazy convexification / regularization oracle:
//   * legacy/lazy preserve the historical low-rank-aware rho controller;
//   * CONST freezes the DC-model rho after spectral initialization;
//   * DYN keeps H_k=0 and changes model rho only through explicit full-space
//     curvature probes; KKT inertia is a separate numerical-stability gate;
//   * on failure escalate geometrically, CAPPED by
//         rho_cap = |min(0, lambda_min_hat)| * (1 + rel_margin)
//     where lambda_min_hat is the measured reduced curvature of Q on
//     ker(Aeq) (projected Lanczos), or a Gershgorin bound when the problem
//     is feasibility-dominated (p/n large) — per the validated policy;
//   * decay rho geometrically after a streak of clean first-try inertias.
#include "lazyipdca/atom_cluster_prod.hpp"  // ATOM_CLUSTER_PROD_V1
#include "lazyipdca/adaptive_marginal_seed.hpp"  // ADAPTIVE_MARGINAL_SEED_V2
#include "lazyipdca/adaptive_marginal_return_v23.hpp"  // ADAPTIVE_MARGINAL_RETURN_V23
#include "lazyipdca/solver.hpp"
#include "lazyipdca/inertial_complete_v24.hpp"  // INERTIA_COMPLETE_AMRS_V24
#include "lazyipdca/mumps_solver.hpp"
#include "lazyipdca/ldl_solver.hpp"
#include "lazyipdca/qdldl_solver.hpp"
#include "lazyipdca/lanczos.hpp"
#include <Eigen/SparseLU>
#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <deque>
#include <map>

namespace lazyipdca {
namespace {

struct Work {
  // iterates
  std::vector<double> x, s, y, z;          // primal x, slack s, duals y, z
  std::vector<double> vl, vu, wl, wu;      // bound duals (x low/up, s low/up)
  // finite-bound masks
  std::vector<char> hl, hu, hsl, hsu;
  // scratch
  std::vector<double> Qx, Aty, Ctz, Ax, Cx, rhs, dx, ds, dy, dz;
  std::vector<double> dvl, dvu, dwl, dwu, thx, ths, xt, st;
};

inline double clampd(double v, double lo, double hi) { return std::min(hi, std::max(lo, v)); }

// ---------------------------------------------------------------- KKT matrix
class KKT {
 public:
  KKT(const Problem& P, const Settings& st) : P_(P), st_(st) { build(nullptr); }

  // Rebuild the pattern with the atomic low-rank convexification appended in
  // AUGMENTED form:  [ K  U ; U' -B^{-1} ]  (Haynsworth: In = In(K+UBU') +
  // (0, natoms, 0)) — one sparse column per atom, target neg = p+m+natoms.
  void set_atoms(const AtomSet& at) { build(&at); }
  idx_t natoms() const { return natoms_; }

  // returns Fact::Ok iff factorization succeeded AND inertia is
  // (n, p+m+natoms, 0)
  enum class Fact { Ok, CurvatureDeficit, RankDeficit, Fail, Budget };
  Fact factorize(const std::vector<double>& thx, const std::vector<double>& ths,
                 double rho, double delta, Info& info) {
    const idx_t n = P_.n, p = P_.p, m = P_.m;
#pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n; ++i) vals_[off_xdiag_ + i] = thx[i] + rho;
    for (idx_t i = 0; i < p; ++i) vals_[off_ydiag_ + i] = -delta;
#pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < m; ++i) vals_[off_zdiag_ + i] = -(1.0 / ths[i] + delta);
    // KKT_FACTORIZATION_BUDGET_V4G: strict resource gate.  Test BEFORE increment/factorization.
    if (st_.max_kkt_factorizations > 0 &&
        info.n_factorizations >= st_.max_kkt_factorizations) {
      info.status = Status::FactorizationBudget;
      info.message = "KKT factorization budget reached";
      return Fact::Budget;
    }
    info.n_factorizations++;
    if (const char* dp = std::getenv("LAZYIPDCA_DUMP_KKT"); dp && !dumped_ &&
        ++dump_calls_ >= (std::getenv("LAZYIPDCA_DUMP_CALL") ? std::atoi(std::getenv("LAZYIPDCA_DUMP_CALL")) : 1)) {
      dumped_ = true;
      if (FILE* f = std::fopen(dp, "w")) {
        for (size_t k = 0; k < vals_.size(); ++k)
          std::fprintf(f, "%d %d %.17g\n", irn_dbg_[k], jcn_dbg_[k], vals_[k]);
        std::fclose(f);
        std::fprintf(stderr, "[dump] KKT %zu nnz, natoms=%lld -> %s\n", vals_.size(), (long long)natoms_, dp);
      }
    }
    // STAGE_PROFILE_KKT_V1
    const bool stage_prof = std::getenv("LAZYIPDCA_STAGE_PROFILE") != nullptr;
    const auto stage_fact_t0 = std::chrono::steady_clock::now();

    bool fact_ok = (qd_ ? qd_->factorize(vals_.data())
                        : ldl_ ? ldl_->factorize(vals_.data())
                               : mumps_->factorize(vals_.data()));

    if (stage_prof) {
      const auto stage_fact_t1 = std::chrono::steady_clock::now();
      const double sec =
          std::chrono::duration<double>(stage_fact_t1-stage_fact_t0).count();
      std::fprintf(stderr,
          "[STAGE-KKT] factorize #%lld sec=%.9f n=%lld m=%lld atoms=%lld ok=%d\n",
          (long long)info.n_factorizations, sec,
          (long long)n, (long long)m, (long long)natoms_, fact_ok ? 1 : 0);
    }
    if (!fact_ok) {
      // Backend mince : un LDL SANS PIVOTAGE n'existe que sur le cone
      // quasi-defini (Vanderbei). Or la certification d'inertie factorise
      // deliberement des matrices d'essai INDEFINIES (rho/2, sondes) : le
      // breakdown (pivot quasi-nul, croissance) y est un THEOREME, pas un
      // accident — MUMPS les traverse par ses pivots 2x2. Le breakdown est
      // donc un SIGNAL DE COURBURE : essai refuse / escalade rho_x,
      // exactement ce que le compte d'inertie MUMPS aurait conclu.
      if (slim_) return Fact::CurvatureDeficit;
      const int e = mumps_->info1();
      if (e == -10 || e == -6) return Fact::RankDeficit;
      // Workspace persistant (-8/-9/...) avec (1,1) non quasi-defini = pivots
      // retardes explosifs => signal de COURBURE : escalader rho le rend
      // quasi-defini et la factorisation redevient bon marche.
      if (e == -8 || e == -9 || e == -14 || e == -15 || e == -17 || e == -20)
        return Fact::CurvatureDeficit;
      return Fact::Fail;
    }
    const int neg = (qd_ ? qd_->neg_pivots() : ldl_ ? ldl_->neg_pivots() : mumps_->neg_pivots()), target = (int)(p + m + natoms_);
    if (!slim_ && mumps_->null_pivots() > 0) return Fact::RankDeficit;
    if (neg > target) return Fact::CurvatureDeficit;   // genuine curvature
    if (neg < target) return Fact::RankDeficit;        // conditioning/rank
    return Fact::Ok;
  }
  int neg_pivots() const { return qd_ ? qd_->neg_pivots() : ldl_ ? ldl_->neg_pivots() : mumps_->neg_pivots(); }
  int info1() const { return slim_ ? 0 : mumps_->info1(); }
  // rhs has size n+p+m; the atom block is padded with zeros internally.
  void solve(std::vector<double>& rhs) {
    const bool stage_prof = std::getenv("LAZYIPDCA_STAGE_PROFILE") != nullptr;
    const auto stage_solve_t0 = std::chrono::steady_clock::now();
    static thread_local long long stage_solve_call = 0;
    ++stage_solve_call;

    if (natoms_ == 0) {
      if (qd_) qd_->solve(rhs.data());
      else if (ldl_) ldl_->solve(rhs.data());
      else mumps_->solve(rhs);

      if (stage_prof) {
        const auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr,
            "[STAGE-KKT] solve #%lld sec=%.9f atoms=0\n",
            stage_solve_call,
            std::chrono::duration<double>(t1-stage_solve_t0).count());
      }
      return;
    }

    std::vector<double> full(P_.n + P_.p + P_.m + natoms_, 0.0);
    std::copy(rhs.begin(), rhs.end(), full.begin());
    if (qd_) qd_->solve(full.data());
    else if (ldl_) ldl_->solve(full.data());
    else mumps_->solve(full);
    std::copy(full.begin(), full.begin() + (idx_t)rhs.size(), rhs.begin());

    if (stage_prof) {
      const auto t1 = std::chrono::steady_clock::now();
      std::fprintf(stderr,
          "[STAGE-KKT] solve #%lld sec=%.9f atoms=%lld\n",
          stage_solve_call,
          std::chrono::duration<double>(t1-stage_solve_t0).count(),
          (long long)natoms_);
    }
  }

 private:
  void build(const AtomSet* at) {
    const idx_t n = P_.n, p = P_.p, m = P_.m;
    natoms_ = at ? at->size() : 0;
    const idx_t N = n + p + m + natoms_;
    const idx_t nq = P_.Q.U.nnz(), na = P_.A.nnz(), nc = P_.C.nnz();
    const idx_t nat = at ? at->nnz() : 0;
    std::vector<int> irn, jcn;
    const idx_t nnz = nq + n + na + nc + p + m + nat + natoms_;
    irn.reserve(nnz); jcn.reserve(nnz);
    vals_.assign(nnz, 0.0);
    // 1) upper triangle of Q (constant values)
    for (idx_t j = 0; j < n; ++j)
      for (idx_t pp = P_.Q.U.colptr[j]; pp < P_.Q.U.colptr[j + 1]; ++pp) {
        idx_t i = P_.Q.U.rowind[pp];
        irn.push_back((int)std::min(i, j) + 1);
        jcn.push_back((int)std::max(i, j) + 1);
        vals_[irn.size() - 1] = P_.Q.U.values[pp];
      }
    // 2) explicit x-block diagonal (theta_x + rho) — updated each iteration
    off_xdiag_ = (idx_t)irn.size();
    for (idx_t i = 0; i < n; ++i) { irn.push_back((int)i + 1); jcn.push_back((int)i + 1); }
    // 3) A^T block (constant)
    for (idx_t j = 0; j < P_.A.cols; ++j)
      for (idx_t pp = P_.A.colptr[j]; pp < P_.A.colptr[j + 1]; ++pp) {
        irn.push_back((int)j + 1);
        jcn.push_back((int)(n + P_.A.rowind[pp]) + 1);
        vals_[irn.size() - 1] = P_.A.values[pp];
      }
    // 4) C^T block (constant)
    for (idx_t j = 0; j < P_.C.cols; ++j)
      for (idx_t pp = P_.C.colptr[j]; pp < P_.C.colptr[j + 1]; ++pp) {
        irn.push_back((int)j + 1);
        jcn.push_back((int)(n + p + P_.C.rowind[pp]) + 1);
        vals_[irn.size() - 1] = P_.C.values[pp];
      }
    // 5) y-block diagonal: -delta — updated each iteration
    off_ydiag_ = (idx_t)irn.size();
    for (idx_t i = 0; i < p; ++i) { irn.push_back((int)(n + i) + 1); jcn.push_back((int)(n + i) + 1); }
    // 6) z-block diagonal: -(1/theta_s + delta) — updated each iteration
    off_zdiag_ = (idx_t)irn.size();
    for (idx_t i = 0; i < m; ++i) { irn.push_back((int)(n + p + i) + 1); jcn.push_back((int)(n + p + i) + 1); }
    // 7) atom columns, congruence-scaled by sqrt(beta):  [K  U S; S U' -I]
    //    with S = diag(sqrt(beta)) — same inertia as [K U; U' -B^{-1}] but the
    //    atom pivot is -1 (perfectly conditioned) instead of -1/beta ~ -1e-6
    //    against O(1) column entries (growth ~1e4 -> MUMPS sign miscount).
    if (at) {
      for (idx_t j = 0; j < natoms_; ++j) {
        const int col = (int)(n + p + m + j) + 1;
        const double sb = std::sqrt(at->beta[j]);
        for (size_t k = 0; k < at->idx[j].size(); ++k) {
          irn.push_back((int)at->idx[j][k] + 1);
          jcn.push_back(col);
          vals_[irn.size() - 1] = sb * at->val[j][k];
        }
        irn.push_back(col); jcn.push_back(col);
        vals_[irn.size() - 1] = -1.0;
      }
    }
    if (std::getenv("LAZYIPDCA_DUMP_KKT")) { irn_dbg_ = irn; jcn_dbg_ = jcn; }
    const int be = (st_.linear_backend != 0) ? st_.linear_backend
                   : (N <= st_.slim_backend_max_dim ? 3 : 1);
    slim_ = (be != 1);
    mumps_.reset(); ldl_.reset(); qd_.reset();
    if (be == 3) {
      qd_ = std::make_unique<QdldlSym>();
      qd_->set_matrix(N, std::move(irn), std::move(jcn), (idx_t)vals_.size(),
                      st_.ordering, st_.pivot_tol);
      qd_->analyze();
    } else if (be == 2) {
      ldl_ = std::make_unique<LdlSym>();
      ldl_->set_matrix(N, std::move(irn), std::move(jcn), (idx_t)vals_.size(),
                       st_.ordering, st_.pivot_tol);
      ldl_->analyze();
    } else {
      mumps_ = std::make_unique<MumpsSym>();
      mumps_->set_matrix(N, std::move(irn), std::move(jcn), (idx_t)vals_.size(),
                         st_.ordering, st_.pivot_tol);
      mumps_->analyze();
    }
  }

  const Problem& P_;
  const Settings& st_;
  std::unique_ptr<MumpsSym> mumps_;
  std::unique_ptr<LdlSym> ldl_;
  std::unique_ptr<QdldlSym> qd_;
  bool slim_ = false;
  std::vector<double> vals_;
  idx_t natoms_ = 0;
  std::vector<int> irn_dbg_, jcn_dbg_;
  bool dumped_ = false; int dump_calls_ = 0;
  idx_t off_xdiag_ = 0, off_ydiag_ = 0, off_zdiag_ = 0;
};

// ------------------------------------------------------- residuals (original)
struct Residuals {
  double stat_abs, stat_rel, eq_abs, ineq_abs, feas_rel, comp_abs, comp_rel;
  double gap, comp_dev; idx_t ncomp;
};

Residuals compute_residuals(const Problem& P, Work& w, double mu = 0.0) {
  const idx_t n = P.n, p = P.p, m = P.m;
  setzero(w.Qx); P.Q.gaxpy_sym(w.x.data(), w.Qx.data());
  setzero(w.Aty); if (p) P.A.gaxpy_t(w.y.data(), w.Aty.data());
  setzero(w.Ctz); if (m) P.C.gaxpy_t(w.z.data(), w.Ctz.data());
  double stat = 0.0, scale = 0.0;
  scale = std::max({norm_inf(w.Qx), norm_inf(P.q), norm_inf(w.Aty), norm_inf(w.Ctz)});
  for (idx_t i = 0; i < n; ++i) {
    double r = w.Qx[i] + P.q[i] - w.Aty[i] - w.Ctz[i] - w.vl[i] + w.vu[i];
    stat = std::max(stat, std::fabs(r));
  }
  double eq = 0.0;
  if (p) {
    setzero(w.Ax); P.A.gaxpy(w.x.data(), w.Ax.data());
    for (idx_t i = 0; i < p; ++i) eq = std::max(eq, std::fabs(w.Ax[i] - P.b[i]));
  }
  double ineq = 0.0;
  if (m) {
    setzero(w.Cx); P.C.gaxpy(w.x.data(), w.Cx.data());
    for (idx_t i = 0; i < m; ++i) {
      if (w.hsl[i]) ineq = std::max(ineq, P.cl[i] - w.Cx[i]);
      if (w.hsu[i]) ineq = std::max(ineq, w.Cx[i] - P.cu[i]);
    }
  }
  for (idx_t i = 0; i < n; ++i) {
    if (w.hl[i]) ineq = std::max(ineq, P.l[i] - w.x[i]);
    if (w.hu[i]) ineq = std::max(ineq, w.x[i] - P.u[i]);
  }
  ineq = std::max(0.0, ineq);
  double comp = 0.0, gap = 0.0, cdev = 0.0; idx_t ncomp = 0;
  auto acc = [&](double dual, double slack) {
    double pr = dual * slack; gap += pr; ncomp++; comp = std::max(comp, std::fabs(pr));
    cdev = std::max(cdev, std::fabs(pr - mu));
  };
  for (idx_t i = 0; i < n; ++i) {
    if (w.hl[i]) acc(w.vl[i], w.x[i] - P.l[i]);
    if (w.hu[i]) acc(w.vu[i], P.u[i] - w.x[i]);
  }
  for (idx_t i = 0; i < m; ++i) {
    if (w.hsl[i]) acc(w.wl[i], w.s[i] - P.cl[i]);
    if (w.hsu[i]) acc(w.wu[i], P.cu[i] - w.s[i]);
  }
  Residuals R;
  R.stat_abs = stat; R.stat_rel = stat / (1.0 + scale);
  R.eq_abs = eq; R.ineq_abs = ineq;
  double xsc = 1.0 + norm_inf(w.x);
  R.feas_rel = std::max(eq, ineq) / xsc;
  R.comp_abs = comp; R.comp_rel = comp / xsc;
  R.gap = gap; R.comp_dev = cdev; R.ncomp = std::max<idx_t>(1, ncomp);
  return R;
}

double objective(const Problem& P, Work& w) {
  setzero(w.Qx); P.Q.gaxpy_sym(w.x.data(), w.Qx.data());
  return 0.5 * dot(w.x, w.Qx) + dot(P.q, w.x) + P.c0;
}

// merit of the DCA-convexified barrier model + exact l1 penalty, evaluated at
// trial point (xt, st); linearization point is the current iterate x (so the
// proximal term is rho/2 ||xt - x||^2).
double merit(const Problem& P, Work& w, const std::vector<double>& xt,
             const std::vector<double>& st, double mu, double sigma, double rho,
             bool& interior, const AtomSet* at, const std::vector<double>& xref,
             const std::vector<double>* inertia_force = nullptr) {
  const idx_t n = P.n, p = P.p, m = P.m;
  interior = true;
  setzero(w.Qx); P.Q.gaxpy_sym(xt.data(), w.Qx.data());
  double f = 0.0;
  for (idx_t i = 0; i < n; ++i) f += (0.5 * w.Qx[i] + P.q[i]) * xt[i];
  double prox = 0.0;
  std::vector<double> dxa(n);
  for (idx_t i = 0; i < n; ++i) {
    dxa[i] = xt[i] - xref[i];
    prox += dxa[i] * dxa[i];
  }
  f += 0.5 * rho * prox;
  if (at && at->size()) f += 0.5 * at->quad(dxa.data());
  // Inertial DC linearization: replacing the concave component's
  // linearization point by an extrapolated/multi-centre point contributes
  // the linear term -p_k'(x-x_ref), with p_k = J_k v_k.
  if (inertia_force && inertia_force->size() == (size_t)n)
    f -= dot(*inertia_force, dxa);
  double bar = 0.0;
  for (idx_t i = 0; i < n; ++i) {
    if (w.hl[i]) { double d = xt[i] - P.l[i]; if (d <= 0) { interior = false; return INF; } bar -= std::log(d); }
    if (w.hu[i]) { double d = P.u[i] - xt[i]; if (d <= 0) { interior = false; return INF; } bar -= std::log(d); }
  }
  for (idx_t i = 0; i < m; ++i) {
    if (w.hsl[i]) { double d = st[i] - P.cl[i]; if (d <= 0) { interior = false; return INF; } bar -= std::log(d); }
    if (w.hsu[i]) { double d = P.cu[i] - st[i]; if (d <= 0) { interior = false; return INF; } bar -= std::log(d); }
  }
  double pen = 0.0;
  if (p) {
    setzero(w.Ax); P.A.gaxpy(xt.data(), w.Ax.data());
    for (idx_t i = 0; i < p; ++i) pen += std::fabs(w.Ax[i] - P.b[i]);
  }
  if (m) {
    setzero(w.Cx); P.C.gaxpy(xt.data(), w.Cx.data());
    for (idx_t i = 0; i < m; ++i) pen += std::fabs(w.Cx[i] - st[i]);
  }
  return f + mu * bar + sigma * pen;
}

}  // namespace

// E14_COMMON_FINALIZER_V1
//
// Common exact-LCP postsolve finalizer.
//
// IMPORTANT:
//   * no spectral routing decision is made here;
//   * no atomic/scalar portfolio decision is made here;
//   * generic QPs never call this function;
//   * IPM KKT counts are not incremented here;
//   * a rejected Newton polish is transactional: the selected child is kept.
//
Result finalize_exact_lcp_e14(
    const Problem& P_in,
    const Settings& st,
    Result X,
    double struct_res,
    double solve_time_now) {

  const idx_t nn=P_in.n;

  X.info.auto_lcp_structure=true;
  X.info.auto_lcp_structure_residual=struct_res;

  struct LcpAuditE14 {
              std::vector<double> w;
              double feas=INF;
              double comp=INF;
              double gap=INF;
              double natural=INF;
              double stat_abs=INF;
              double stat_rel=INF;
              double dual_feas=INF;
            };

            auto audit_e14 =
                [&](const std::vector<double>& xv) -> LcpAuditE14 {

              LcpAuditE14 A;

              if ((idx_t)xv.size()!=nn)
                return A;

              A.w.assign((size_t)nn,0.0);
              P_in.C.gaxpy(xv.data(),A.w.data());

              A.feas=0.0;
              A.comp=0.0;
              A.gap=0.0;
              A.natural=0.0;

              for (idx_t k=0;k<nn;++k) {
                A.w[k]-=P_in.cl[k];

                A.feas=std::max(
                    A.feas,
                    std::max(0.0,-xv[k]));

                A.feas=std::max(
                    A.feas,
                    std::max(0.0,-A.w[k]));

                A.comp=std::max(
                    A.comp,
                    std::fabs(xv[k]*A.w[k]));

                A.gap += xv[k]*A.w[k];

                A.natural=std::max(
                    A.natural,
                    std::fabs(std::min(xv[k],A.w[k])));
              }

              // For the proved structure:
              //
              // Q=C+C', q=-cl.
              //
              // Exact dual reconstruction:
              // z=x, zl=w, zu=0.
              std::vector<double> Qx((size_t)nn,0.0);
              std::vector<double> Ctx((size_t)nn,0.0);

              P_in.Q.gaxpy_sym(xv.data(),Qx.data());
              P_in.C.gaxpy_t(xv.data(),Ctx.data());

              A.stat_abs=0.0;
              A.dual_feas=0.0;
              double scale=1.0;

              for (idx_t k=0;k<nn;++k) {
                const double rr=
                    Qx[k] + P_in.q[k]
                    - Ctx[k]
                    - A.w[k];

                A.stat_abs=std::max(
                    A.stat_abs,
                    std::fabs(rr));

                scale=std::max(
                    {scale,
                     std::fabs(Qx[k]),
                     std::fabs(P_in.q[k]),
                     std::fabs(Ctx[k]),
                     std::fabs(A.w[k])});

                A.dual_feas=std::max(
                    A.dual_feas,
                    std::max(0.0,-xv[k]));

                A.dual_feas=std::max(
                    A.dual_feas,
                    std::max(0.0,-A.w[k]));
              }

              A.stat_rel=A.stat_abs/std::max(1.0,scale);

              return A;
            };

            const double cert_tol=
                std::max(0.0,st.auto_lcp_admissible_tol);

            const double gap_tol=
                std::max(0.0,st.auto_lcp_gap_target);

            auto lcp_cert =
                [&](const LcpAuditE14& A) -> bool {
                  return
                      std::isfinite(A.feas) &&
                      std::isfinite(A.comp) &&
                      std::isfinite(A.gap) &&
                      A.feas<=cert_tol &&
                      A.comp<=cert_tol &&
                      std::fabs(A.gap)<=gap_tol;
                };

            LcpAuditE14 pre=audit_e14(X.x);

            X.info.auto_lcp_pre_gap=pre.gap;
            X.info.auto_lcp_post_gap=pre.gap;
            X.info.auto_lcp_pre_comp=pre.comp;
            X.info.auto_lcp_post_comp=pre.comp;
            X.info.auto_lcp_pre_natural=pre.natural;
            X.info.auto_lcp_post_natural=pre.natural;
            X.info.auto_lcp_pre_feas=pre.feas;
            X.info.auto_lcp_post_feas=pre.feas;
            X.info.auto_lcp_post_kkt_stat=pre.stat_rel;

            // ------------------------------------------------------
            // FAST PATH: production pays ZERO Newton if already good.
            // ------------------------------------------------------
            if (lcp_cert(pre) ||
                !st.auto_lcp_polish_enable) {

              if (st.verbose>=1)
                std::printf(
                  "[LazyDCA-AUTO-v4hE1.4] exact LCP fast path: "
                  "gap=%.3e comp=%.3e feas=%.3e nat=%.3e "
                  "(polish=%s)\n",
                  pre.gap,pre.comp,pre.feas,pre.natural,
                  st.auto_lcp_polish_enable
                    ? "not-needed"
                    : "disabled");

              return X;
            }

            X.info.auto_lcp_polish_used=true;

            bool singular=false;
            std::vector<double> xp=X.x;
            LcpAuditE14 cur=pre;

            const idx_t max_newton=
                std::max<idx_t>(
                    0,
                    st.auto_lcp_polish_max_iters);

            const idx_t max_bt=
                std::max<idx_t>(
                    0,
                    st.auto_lcp_polish_max_backtracks);

            const double armijo=
                std::min(
                    0.5,
                    std::max(
                        0.0,
                        st.auto_lcp_polish_armijo));

            const double nat_tol=
                std::max(
                    0.0,
                    st.auto_lcp_polish_natural_tol);

            using E14SpMat=
                Eigen::SparseMatrix<
                    double,
                    Eigen::ColMajor,
                    int>;

            if (nn >
                static_cast<idx_t>(
                  std::numeric_limits<int>::max())) {
              singular=true;
            }

            // Important: no termination from natural residual alone.
            // LCP10 demonstrated that natural residual can be tiny while
            // aggregate x'w still misses the requested historical gap.
            for (idx_t nit=0;
                 !singular && nit<max_newton;
                 ++nit) {

              cur=audit_e14(xp);

              if (lcp_cert(cur))
                break;

              std::vector<char> use_C(
                  (size_t)nn,
                  0);

              for (idx_t k=0;k<nn;++k)
                use_C[k]=(cur.w[k] < xp[k]) ? 1 : 0;

              std::vector<
                  Eigen::Triplet<double,int>> trip;

              trip.reserve(
                  P_in.C.values.size()
                  + static_cast<size_t>(nn));

              // Rows taking derivative of w=Cx-cl.
              for (idx_t col=0;
                   col<P_in.C.cols;
                   ++col) {

                for (idx_t pp=P_in.C.colptr[col];
                     pp<P_in.C.colptr[col+1];
                     ++pp) {

                  const idx_t row=
                      P_in.C.rowind[pp];

                  if (use_C[row])
                    trip.emplace_back(
                        static_cast<int>(row),
                        static_cast<int>(col),
                        P_in.C.values[pp]);
                }
              }

              // Rows taking derivative of x.
              for (idx_t k=0;k<nn;++k) {
                if (!use_C[k])
                  trip.emplace_back(
                      static_cast<int>(k),
                      static_cast<int>(k),
                      1.0);
              }

              E14SpMat J(
                  static_cast<int>(nn),
                  static_cast<int>(nn));

              J.setFromTriplets(
                  trip.begin(),
                  trip.end());

              J.makeCompressed();

              Eigen::VectorXd rhs(
                  static_cast<Eigen::Index>(nn));

              for (idx_t k=0;k<nn;++k)
                rhs[static_cast<Eigen::Index>(k)] =
                    -std::min(xp[k],cur.w[k]);

              Eigen::SparseLU<
                  E14SpMat,
                  Eigen::COLAMDOrdering<int>> lu;

              lu.analyzePattern(J);
              lu.factorize(J);

              X.info.auto_lcp_polish_factorizations++;

              if (lu.info()!=Eigen::Success) {
                singular=true;
                break;
              }

              Eigen::VectorXd dx=lu.solve(rhs);

              if (lu.info()!=Eigen::Success ||
                  !dx.allFinite()) {
                singular=true;
                break;
              }

              bool accepted=false;
              double alpha=1.0;

              for (idx_t bt=0;bt<=max_bt;++bt) {
                std::vector<double> xt=xp;

                for (idx_t k=0;k<nn;++k)
                  xt[k] +=
                      alpha *
                      dx[static_cast<Eigen::Index>(k)];

                const LcpAuditE14 trial=
                    audit_e14(xt);

                const double rhs_nat=
                    (1.0-armijo*alpha)*cur.natural
                    + std::max(
                        1e-15,
                        0.01*nat_tol);

                if (std::isfinite(trial.natural) &&
                    trial.natural<=rhs_nat) {

                  xp.swap(xt);
                  cur=trial;
                  accepted=true;

                  X.info.auto_lcp_polish_iterations++;

                  if (bt>0)
                    X.info.auto_lcp_polish_backtracks += bt;

                  break;
                }

                alpha*=0.5;
              }

              if (!accepted)
                break;
            }

            X.info.auto_lcp_polish_singular=singular;

            const LcpAuditE14 post=
                audit_e14(xp);

            X.info.auto_lcp_post_gap=post.gap;
            X.info.auto_lcp_post_comp=post.comp;
            X.info.auto_lcp_post_natural=post.natural;
            X.info.auto_lcp_post_feas=post.feas;
            X.info.auto_lcp_post_kkt_stat=post.stat_rel;

            const double kkt_tol=
                std::max(
                    0.0,
                    st.auto_lcp_polish_kkt_tol);

            const bool full_cert=
                lcp_cert(post) &&
                post.dual_feas<=kkt_tol &&
                post.stat_rel<=kkt_tol &&
                post.natural<=
                    pre.natural +
                    std::max(
                        1e-14,
                        0.01*nat_tol);

            X.info.auto_lcp_polish_kkt_certified=
                full_cert;

            if (full_cert) {
              X.x=xp;

              // Exact reconstructed QP duals.
              X.z=xp;
              X.zl=post.w;
              X.zu.assign((size_t)nn,0.0);
              X.y.clear();

              // On the algebraically certified LCP structure,
              //
              //   0.5*x'Q*x + q'x = x'(Cx-cl) = x'w.
              //
              // Use the structural evaluation directly: it avoids catastrophic
              // cancellation near complementarity and is mathematically identical
              // under the detector invariants Q=C+C', q=-cl, c0=0.
              const double objp=post.gap;
              X.info.obj=objp;

              if (!std::isfinite(X.info.best_feasible_obj) ||
                  objp<X.info.best_feasible_obj)
                X.info.best_feasible_obj=objp;

              X.info.res_stat_abs=post.stat_abs;
              X.info.res_stat_rel=post.stat_rel;

              X.info.res_eq_abs=0.0;
              X.info.res_ineq_abs=post.feas;
              X.info.res_feas_rel=post.feas;

              X.info.res_comp_abs=post.comp;
              X.info.res_comp_rel=post.comp;

              // First-order certificate only.
              X.info.status=Status::Solved;
              X.info.soc_certified=false;
              X.info.message=
                  "exact LCP semismooth polish + reconstructed QP KKT certified";

              X.info.auto_lcp_polish_accepted=true;
              X.info.solve_time=solve_time_now;

              if (st.verbose>=1)
                std::printf(
                  "[LazyDCA-AUTO-v4hE1.4] polish ACCEPTED: "
                  "gap %.3e -> %.3e, "
                  "comp %.3e -> %.3e, "
                  "nat %.3e -> %.3e, "
                  "stat=%.3e, NewtonFact=%lld, "
                  "steps=%lld, backtracks=%lld\n",
                  pre.gap,post.gap,
                  pre.comp,post.comp,
                  pre.natural,post.natural,
                  post.stat_rel,
                  (long long)
                    X.info.auto_lcp_polish_factorizations,
                  (long long)
                    X.info.auto_lcp_polish_iterations,
                  (long long)
                    X.info.auto_lcp_polish_backtracks);
            } else {
              // Transactional rejection:
              // X.x and all child duals remain untouched.
              if (st.verbose>=1)
                std::printf(
                  "[LazyDCA-AUTO-v4hE1.4] polish REJECTED: "
                  "gap %.3e -> %.3e, "
                  "comp %.3e -> %.3e, "
                  "nat %.3e -> %.3e, "
                  "stat=%.3e, singular=%d\n",
                  pre.gap,post.gap,
                  pre.comp,post.comp,
                  pre.natural,post.natural,
                  post.stat_rel,
                  singular ? 1 : 0);
            }

            return X;

}

// End E14_COMMON_FINALIZER_V1

// ============================================================== main solver
Result solve(const Problem& P_in, const Settings& st) {
  const bool reg_const = st.regularization_mode == DcRegularizationMode::Const;
  const bool reg_dyn   = st.regularization_mode == DcRegularizationMode::Dyn;
  const bool reg_lazy  = st.regularization_mode == DcRegularizationMode::Lazy;
  const bool reg_controlled = reg_const || reg_dyn || reg_lazy;
  // Controlled OMS modes: CONST/DYN impose H_k=0. LAZY uses the low-rank
  // correction by default, but honors atoms_enable=false for the isolated
  // LAZY-no-atoms A/B test. In that ablation H_k=0 while all LAZY update/
  // acceptance logic remains active. A one-time spectral scalar shift is then
  // required to keep the initial DCA subproblem convex.
  bool atoms_enabled = (reg_const || reg_dyn) ? false : st.atoms_enable;
  // CONST always freezes the DC-model rho.  rho_model_lock is an isolated
  // controlled-LAZY A/B switch: rho_k is then kept EXACTLY at rho_model_min,
  // while rho_x remains free to stabilize the numerical factorization.
  const bool rho_model_fixed = reg_const || (reg_lazy && st.rho_model_lock);
  const bool rho_spectral_init = reg_const || reg_dyn ||
                                 (reg_lazy && (!atoms_enabled || st.atom_oracle_enable));
  // A/B-only scalar floor for the LAZY DC-model shift. It does NOT act on
  // rho_x and is ignored by Legacy/CONST/DYN.  With rho_model_lock=true this
  // value becomes the exact model shift, not merely a lower bound.
  const double rho_model_floor = reg_lazy
      ? std::clamp(st.rho_model_min, 0.0, st.rho_abs_max)
      : 0.0;
  // ADAPTIVE_ORACLE_HYBRID_V4C
  // Mathematical state: changed only after an atom COMMIT.
  double rho_oracle_residual_floor = 0.0;

  auto apply_rho_model_floor = [&](double candidate) {
    if (reg_lazy && st.rho_model_lock) return rho_model_floor;
    // ADAPTIVE_ORACLE_HYBRID_V4D:
    // Transient atomic-COMMIT guard only. It must not permanently
    // constrain the ordinary LAZY rho controller.
    return reg_lazy ? std::max(candidate, rho_model_floor) : candidate;
  };

  // Plafond STRUCTUREL : au-dela de n atomes, ils sont NECESSAIREMENT
  // lineairement dependants (l'espace des directions dans R^n n'a que n
  // dimensions) — un atome supplementaire n'apporte AUCUNE courbure
  // nouvelle, seulement une colonne quasi-redondante dans le systeme KKT qui
  // degrade le conditionnement (observe : n=385, --max-atoms 1000 -> 1000
  // atomes commis, dont 615 necessairement redondants, cycles d'inertie).
  const idx_t max_atoms_eff = atoms_enabled ? std::min(st.max_atoms, P_in.n) : 0;
  // Initial seed has its own work/rank budget.  Default <=0 deliberately
  // aliases the historical max_atoms budget, so old commands are bit-for-bit
  // semantically unchanged at this decision point.
  const idx_t max_seed_atoms_eff = atoms_enabled
      ? ((st.atoms_seed_max_atoms > 0)
            ? std::min(st.atoms_seed_max_atoms, max_atoms_eff)
            : max_atoms_eff)
      : 0;
  if (atoms_enabled && max_atoms_eff < st.max_atoms && st.verbose >= 0)
    std::fprintf(stderr, "[LazyDCA] ATTENTION : --max-atoms %lld > n=%lld ; plafonne a %lld "
                         "(au-dela de n, les atomes sont necessairement lineairement dependants "
                         "et n'apportent aucune courbure nouvelle).\n",
                (long long)st.max_atoms, (long long)P_in.n, (long long)max_atoms_eff);
  // ---- IPOPT gradient-based scaling (statique, au point initial x0=0-ish) --
  Problem Ps;
  double s_f = 1.0;
  std::vector<double> s_eq, s_in;
  const bool do_scale = st.scaling_enable;
  if (do_scale) {
    Ps = P_in;  // copie profonde (structures CSC copiees)
    // gradient de f a l'origine des variables (QP: q) — IPOPT évalue au x0
    double gmax = 0.0;
    for (idx_t i = 0; i < P_in.n; ++i) gmax = std::max(gmax, std::fabs(P_in.q[i]));
    s_f = std::min(1.0, st.scaling_max_grad / std::max(gmax, 1e-8));
    for (double& v : Ps.Q.U.values) v *= s_f;
    for (idx_t i = 0; i < P_in.n; ++i) Ps.q[i] *= s_f;
    Ps.c0 *= s_f;
    s_eq.assign(P_in.p, 1.0); s_in.assign(P_in.m, 1.0);
    { std::vector<double> rn(P_in.p, 0.0);
      for (idx_t j = 0; j < P_in.A.cols; ++j)
        for (idx_t k = P_in.A.colptr[j]; k < P_in.A.colptr[j + 1]; ++k)
          rn[P_in.A.rowind[k]] = std::max(rn[P_in.A.rowind[k]], std::fabs(P_in.A.values[k]));
      for (idx_t i = 0; i < P_in.p; ++i) s_eq[i] = std::min(1.0, st.scaling_max_grad / std::max(rn[i], 1e-8));
      for (idx_t j = 0; j < Ps.A.cols; ++j)
        for (idx_t k = Ps.A.colptr[j]; k < Ps.A.colptr[j + 1]; ++k)
          Ps.A.values[k] *= s_eq[Ps.A.rowind[k]];
      for (idx_t i = 0; i < P_in.p; ++i) Ps.b[i] *= s_eq[i]; }
    { std::vector<double> rn(P_in.m, 0.0);
      for (idx_t j = 0; j < P_in.C.cols; ++j)
        for (idx_t k = P_in.C.colptr[j]; k < P_in.C.colptr[j + 1]; ++k)
          rn[P_in.C.rowind[k]] = std::max(rn[P_in.C.rowind[k]], std::fabs(P_in.C.values[k]));
      for (idx_t i = 0; i < P_in.m; ++i) s_in[i] = std::min(1.0, st.scaling_max_grad / std::max(rn[i], 1e-8));
      for (idx_t j = 0; j < Ps.C.cols; ++j)
        for (idx_t k = Ps.C.colptr[j]; k < Ps.C.colptr[j + 1]; ++k)
          Ps.C.values[k] *= s_in[Ps.C.rowind[k]];
      for (idx_t i = 0; i < P_in.m; ++i) { Ps.cl[i] *= s_in[i]; Ps.cu[i] *= s_in[i]; } }
  }
  // Variables FIXEES (l == u, cas QPNSTAIR) : une boite sans interieur rend
  // la barriere infinie (log 0) — toute factorisation echoue quel que soit
  // rho (escalade vaine observee jusqu'a 2.3e12), puis merites NaN. Le
  // traitement IPM standard : RELAXATION EPSILON de la boite (l-e, u+e) —
  // l'interieur existe, la barriere est finie, x reste epingle a e pres.
  bool has_fixed = false;
  {
    const idx_t nfx = P_in.n;
    for (idx_t i = 0; i < nfx; ++i) {
      const bool hl = std::isfinite(P_in.l.empty() ? -1e300 : P_in.l[i]);
      const bool hu = std::isfinite(P_in.u.empty() ?  1e300 : P_in.u[i]);
      if (hl && hu && P_in.u[i] - P_in.l[i] <
          2e-8 * std::max(1.0, std::fabs(P_in.l[i]))) { has_fixed = true; break; }
    }
  }
  if (has_fixed && !do_scale) Ps = P_in;   // copie mutable requise
  if (has_fixed) {
    for (idx_t i = 0; i < Ps.n; ++i) {
      const bool hl = std::isfinite(Ps.l.empty() ? -1e300 : Ps.l[i]);
      const bool hu = std::isfinite(Ps.u.empty() ?  1e300 : Ps.u[i]);
      if (hl && hu) {
        const double eps_box = 1e-8 * std::max(1.0, std::fabs(Ps.l[i]));
        if (Ps.u[i] - Ps.l[i] < 2.0 * eps_box) {
          Ps.l[i] -= eps_box; Ps.u[i] += eps_box;
        }
      }
    }
  }
  const Problem& P = (do_scale || has_fixed) ? Ps : P_in;
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

  Result res;
  Info& info = res.info;
  const idx_t n = P.n, p = P.p, m = P.m;

  Work w;
  w.x.assign(n, 0.0); w.s.assign(m, 0.0); w.y.assign(p, 0.0); w.z.assign(m, 0.0);
  w.vl.assign(n, 0.0); w.vu.assign(n, 0.0); w.wl.assign(m, 0.0); w.wu.assign(m, 0.0);
  w.hl.assign(n, 0); w.hu.assign(n, 0); w.hsl.assign(m, 0); w.hsu.assign(m, 0);
  for (idx_t i = 0; i < n; ++i) { w.hl[i] = std::isfinite(P.l[i]); w.hu[i] = std::isfinite(P.u[i]); }
  for (idx_t i = 0; i < m; ++i) { w.hsl[i] = std::isfinite(P.cl[i]); w.hsu[i] = std::isfinite(P.cu[i]); }
  w.Qx.assign(n, 0.0); w.Aty.assign(n, 0.0); w.Ctz.assign(n, 0.0);
  w.Ax.assign(p, 0.0); w.Cx.assign(m, 0.0);
  w.rhs.assign(n + p + m, 0.0);
  w.dx.assign(n, 0.0); w.ds.assign(m, 0.0); w.dy.assign(p, 0.0); w.dz.assign(m, 0.0);
  w.dvl.assign(n, 0.0); w.dvu.assign(n, 0.0); w.dwl.assign(m, 0.0); w.dwu.assign(m, 0.0);
  w.thx.assign(n, 0.0); w.ths.assign(m, 0.0);
  w.xt.assign(n, 0.0); w.st.assign(m, 0.0);

  // ---------------- initialization: interior x0 + feasible anchor ----------
  for (idx_t i = 0; i < n; ++i) {
    double lo = P.l[i], hi = P.u[i], v = 0.0;
    if (w.hl[i] && w.hu[i]) { double k = 0.1 * std::min(1.0, hi - lo); v = clampd(v, lo + k, hi - k); if (hi <= lo) v = 0.5 * (lo + hi); }
    else if (w.hl[i]) v = std::max(v, lo + 1.0);
    else if (w.hu[i]) v = std::min(v, hi - 1.0);
    w.x[i] = v;
  }
  // -------- article §4.1 trust-region start (paper Algorithm 3) ------------
  // min over the ball S = {||v|| <= sqrt(n)} of the box-mapped quadratic:
  //   x = Dinv v + mid,  Dinv = diag((u-l)/2),  mid = (l+u)/2,
  //   f(x(v)) = 1/2 v'Mv + g0'v + const,  M = Dinv Q Dinv,  g0 = Dinv(Q mid + q).
  // Projection-DCA (global for the trust-region subproblem):
  //   w_k = (rho I - M) v_k - g0 ;  v_{k+1} = w_k/rho if ||w_k|| <= rho*sqrt(n)
  //                                 else sqrt(n) * w_k / ||w_k||.
  if (st.init_strategy >= 1) {
    bool bounded = true;
    for (idx_t i = 0; i < n && bounded; ++i)
      bounded = w.hl[i] && w.hu[i] && (P.u[i] - P.l[i] > 1e-12);
    if (!bounded) {
      if (st.verbose >= 2) std::printf("[LazyDCA-init] TR start requested but box unbounded -> anchor fallback\n");
    } else {
      std::vector<double> dinv(n), mid(n), g0(n, 0.0), v(n, 0.0), vw(n), t(n), xv(n);
      for (idx_t i = 0; i < n; ++i) { dinv[i] = 0.5 * (P.u[i] - P.l[i]); mid[i] = 0.5 * (P.l[i] + P.u[i]); }
      setzero(t); P.Q.gaxpy_sym(mid.data(), t.data());
      for (idx_t i = 0; i < n; ++i) g0[i] = dinv[i] * (t[i] + P.q[i]);
      // rho >= lambda_max(M) via Gershgorin on the scaled matrix
      std::vector<double> diag(n, 0.0), rad(n, 0.0);
      for (idx_t j = 0; j < n; ++j)
        for (idx_t pp = P.Q.U.colptr[j]; pp < P.Q.U.colptr[j + 1]; ++pp) {
          idx_t i = P.Q.U.rowind[pp];
          double a = P.Q.U.values[pp] * dinv[i] * dinv[j];
          if (i == j) diag[j] += a;
          else { rad[i] += std::fabs(a); rad[j] += std::fabs(a); }
        }
      double lmax_ub = -INF;
      for (idx_t j = 0; j < n; ++j) lmax_ub = std::max(lmax_ub, diag[j] + rad[j]);
      const double rho_tr = std::max(lmax_ub, 1e-8);
      const double rad_S = std::sqrt((double)n);
      double err = 1.0; idx_t it_tr = 0;
      for (; it_tr < st.tr_dca_maxit && err > st.tr_dca_tol; ++it_tr) {
        // vw = (rho I - M) v - g0 = rho v - Dinv Q Dinv v - g0
        for (idx_t i = 0; i < n; ++i) t[i] = dinv[i] * v[i];
        setzero(xv); P.Q.gaxpy_sym(t.data(), xv.data());
        double nw = 0.0;
        for (idx_t i = 0; i < n; ++i) { vw[i] = rho_tr * v[i] - dinv[i] * xv[i] - g0[i]; nw += vw[i] * vw[i]; }
        nw = std::sqrt(nw);
        double sc = (nw <= rho_tr * rad_S) ? (1.0 / rho_tr) : (rad_S / std::max(nw, 1e-300));
        double e2 = 0.0, nv2 = 0.0;
        for (idx_t i = 0; i < n; ++i) {
          double vn = sc * vw[i], d = vn - v[i];
          e2 += d * d; nv2 += v[i] * v[i]; v[i] = vn;
        }
        err = std::sqrt(e2) / (1.0 + std::sqrt(nv2));
      }
      // back-transform + strict interior clip (our barrier needs l < x < u)
      for (idx_t i = 0; i < n; ++i) {
        double xi = dinv[i] * v[i] + mid[i];
        double k = st.tr_interior_margin * std::max(1e-8, P.u[i] - P.l[i]);
        w.x[i] = clampd(xi, P.l[i] + k, P.u[i] - k);
      }
      if (st.verbose >= 2) {
        setzero(w.Qx); P.Q.gaxpy_sym(w.x.data(), w.Qx.data());
        double f0 = 0.0;
        for (idx_t i = 0; i < n; ++i) f0 += (0.5 * w.Qx[i] + P.q[i]) * w.x[i];
        std::printf("[LazyDCA-init] TR start (Alg.3): it=%lld err=%.2e rho_tr=%.3e f(x0)=%+.8e\n",
                    (long long)it_tr, err, rho_tr, f0 + P.c0);
      }
    }
  }
  // Kernel projector, built once (reused by the anchor, Lanczos, certificate)
  std::unique_ptr<KerProjector> proj;
  const bool feas_dominated = (n > 0) && ((double)p / (double)n > st.eq_density_skip);
  if (p > 0 && (st.feasible_anchor || (!feas_dominated))) {
    try { proj = std::make_unique<KerProjector>(n, P.A, st.ordering); }
    catch (...) { proj.reset(); }
  }
  const bool do_anchor = st.feasible_anchor && (st.init_strategy != 1);
  if (p > 0 && do_anchor && proj && proj->ok()) {
    // x_anchor = argmin 1/2||x - x0||^2 s.t. Ax = b   (unanchored-proximal fix)
    // solve [I A'; A -dI][xa; ya] = [x0; b - A x0] then xa = x0 + correction:
    // directly: [I A'; A -dI][xa; ya] = [x0; b]  => xa = P x0 + A'(AA')^+ b.
    std::vector<double> rhs(n + p, 0.0);
    for (idx_t i = 0; i < n; ++i) rhs[i] = w.x[i];
    for (idx_t i = 0; i < p; ++i) rhs[n + i] = P.b[i];
    // KerProjector::apply solves with zero second block; do a raw solve here:
    // (small extension: apply() handles zero-b; reuse by shifting)
    // xa = P x0 + xb where xb solves [I A'; A -dI][xb; y] = [0; b].
    std::vector<double> px(n, 0.0);
    proj->apply(w.x.data(), px.data());
    std::vector<double> xb(n, 0.0);
    {
      std::vector<double> e(n, 0.0);
      // trick: apply() on v solves rhs=[v;0]; we need [0;b].  Solve via
      // superposition using a dedicated method would be cleaner; emulate:
      // [I A'; A -dI][xb;y]=[0;b]  <=>  xb = A'(AA'+d)^{-1} b.  Compute by
      // one extra projector built on-the-fly is wasteful; instead use CG on
      // (AA'+d) which is cheap for moderate p, or reuse: apply on A' seed.
      // Practical route: xb = x_ls where x_ls = argmin ||Ax-b||: run a few
      // CG iterations on normal equations (matrix-free).
      std::vector<double> r(p), pr(p), Ap(p), xls(n, 0.0), Atr(n, 0.0), Apx(p, 0.0);
      // r = b - A*0 = b
      r = P.b; pr = r;
      double rs = dot(r, r);
      for (int it = 0; it < 200 && std::sqrt(rs) > 1e-10 * (1.0 + norm_inf(P.b)); ++it) {
        setzero(Atr); P.A.gaxpy_t(pr.data(), Atr.data());       // A' p
        setzero(Apx); P.A.gaxpy(Atr.data(), Apx.data());        // A A' p
        double alpha = rs / std::max(1e-300, dot(pr, Apx));
        for (idx_t i = 0; i < n; ++i) xls[i] += alpha * Atr[i];
        for (idx_t i = 0; i < p; ++i) r[i] -= alpha * Apx[i];
        double rs2 = dot(r, r);
        for (idx_t i = 0; i < p; ++i) pr[i] = r[i] + (rs2 / std::max(1e-300, rs)) * pr[i];
        rs = rs2;
      }
      xb = xls;
    }
    for (idx_t i = 0; i < n; ++i) w.x[i] = px[i] + xb[i];
    // projections alternees : la re-interiorisation boite casse Ax=b ; on
    // alterne (projection egalite par CG) <-> (clip interieur) pour retomber
    // faisable AUX DEUX (restauration d'ancrage du zip).
    for (int rounds = 0; rounds < 3; ++rounds) {
      for (idx_t i = 0; i < n; ++i) {
        if (w.hl[i] && w.hu[i]) { double k = 1e-3 * std::max(1e-8, P.u[i] - P.l[i]); w.x[i] = clampd(w.x[i], P.l[i] + k, P.u[i] - k); }
        else if (w.hl[i]) w.x[i] = std::max(w.x[i], P.l[i] + 1e-3);
        else if (w.hu[i]) w.x[i] = std::min(w.x[i], P.u[i] - 1e-3);
      }
      // correction egalite : x += A'(AA')^{-1}(b - Ax) par CG
      std::vector<double> resid(p, 0.0);
      { std::vector<double> ax(p, 0.0); P.A.gaxpy(w.x.data(), ax.data());
        for (idx_t i = 0; i < p; ++i) resid[i] = P.b[i] - ax[i]; }
      if (norm_inf(resid) <= 1e-10) break;
      std::vector<double> rr(resid), pr(resid), Ap(p), Atr(n), lam(p, 0.0);
      double rs = dot(rr, rr);
      for (int it2 = 0; it2 < 200 && std::sqrt(rs) > 1e-12 * (1.0 + norm_inf(P.b)); ++it2) {
        setzero(Atr); P.A.gaxpy_t(pr.data(), Atr.data());
        setzero(Ap);  P.A.gaxpy(Atr.data(), Ap.data());
        double alpha = rs / std::max(1e-300, dot(pr, Ap));
        for (idx_t i = 0; i < p; ++i) lam[i] += alpha * pr[i];
        for (idx_t i = 0; i < p; ++i) rr[i] -= alpha * Ap[i];
        double rs2 = dot(rr, rr);
        for (idx_t i = 0; i < p; ++i) pr[i] = rr[i] + (rs2 / std::max(1e-300, rs)) * pr[i];
        rs = rs2;
      }
      setzero(Atr); P.A.gaxpy_t(lam.data(), Atr.data());
      for (idx_t i = 0; i < n; ++i) w.x[i] += Atr[i];
    }
    // clip interieur FINAL : la correction d'egalite peut repousser x hors
    // boite ; la barriere exige l'interieur strict (slacks > 0), sinon
    // Theta_x devient negatif geant et aucune escalade de rho ne le couvre.
    for (idx_t i = 0; i < n; ++i) {
      if (w.hl[i] && w.hu[i]) { double k = 1e-4 * std::max(1e-8, P.u[i] - P.l[i]); w.x[i] = clampd(w.x[i], P.l[i] + k, P.u[i] - k); }
      else if (w.hl[i]) w.x[i] = std::max(w.x[i], P.l[i] + 1e-4);
      else if (w.hu[i]) w.x[i] = std::min(w.x[i], P.u[i] - 1e-4);
    }
    if (st.verbose >= 2) std::printf("[LazyDCA-anchor] feasible anchor applied, ||Ax-b||_inf after = ");
    if (st.verbose >= 2) { setzero(w.Ax); P.A.gaxpy(w.x.data(), w.Ax.data()); double e = 0; for (idx_t i = 0; i < p; ++i) e = std::max(e, std::fabs(w.Ax[i] - P.b[i])); std::printf("%.2e\n", e); }
  }
  // ---- bound_push d'IPOPT (kappa_1 = bound_push, kappa_2 = bound_frac) ----
  // Le point initial est repousse a l'INTERIEUR : p_L = min(k1*max(1,|l|),
  // k2*(u-l)). Un x0 colle aux bornes seme des slacks ~eps -> Theta enormes,
  // duals mu/sl a 1e3+ des la naissance -> la derive degeneree est semee a
  // l'initialisation (NCVXQP7). IPOPT part a >= 1% des bornes.
  for (idx_t i = 0; i < n; ++i) {
    if (w.hl[i] && w.hu[i]) {
      const double r = P.u[i] - P.l[i];
      const double pL = std::min(st.bound_push * std::max(1.0, std::fabs(P.l[i])), st.bound_frac * r);
      const double pU = std::min(st.bound_push * std::max(1.0, std::fabs(P.u[i])), st.bound_frac * r);
      if (P.l[i] + pL >= P.u[i] - pU) w.x[i] = 0.5 * (P.l[i] + P.u[i]);
      else w.x[i] = clampd(w.x[i], P.l[i] + pL, P.u[i] - pU);
    } else if (w.hl[i]) {
      w.x[i] = std::max(w.x[i], P.l[i] + st.bound_push * std::max(1.0, std::fabs(P.l[i])));
    } else if (w.hu[i]) {
      w.x[i] = std::min(w.x[i], P.u[i] - st.bound_push * std::max(1.0, std::fabs(P.u[i])));
    }
  }

  // slacks strictly interior (meme regle bound_push/bound_frac)
  if (m) { setzero(w.Cx); P.C.gaxpy(w.x.data(), w.Cx.data()); }
  for (idx_t i = 0; i < m; ++i) {
    double v = w.Cx[i];
    if (w.hsl[i] && w.hsu[i]) {
      const double r = P.cu[i] - P.cl[i];
      const double pL = std::min(st.bound_push * std::max(1.0, std::fabs(P.cl[i])), st.bound_frac * r);
      const double pU = std::min(st.bound_push * std::max(1.0, std::fabs(P.cu[i])), st.bound_frac * r);
      if (P.cl[i] + pL >= P.cu[i] - pU) v = 0.5 * (P.cl[i] + P.cu[i]);
      else v = clampd(v, P.cl[i] + pL, P.cu[i] - pU);
    }
    else if (w.hsl[i]) v = std::max(v, P.cl[i] + st.bound_push * std::max(1.0, std::fabs(P.cl[i])));
    else if (w.hsu[i]) v = std::min(v, P.cu[i] - st.bound_push * std::max(1.0, std::fabs(P.cu[i])));
    w.s[i] = v;
  }
  // duals from mu0
  double mu = (st.mu0 > 0) ? st.mu0 : 1.0;
  if (st.bound_mult_init_val > 0.0) {
    // Init a la IPOPT (bound_mult_init_val) : multiplicateurs de bornes
    // UNIFORMES z = valeur, bornes par construction, indifferents aux
    // slacks — la ou l'init centree mu/sl fait naitre un dual a 1e4 des
    // qu'un slack vaut 1e-4 (semence de la derive degeneree). La premiere
    // iteration recentre.
    for (idx_t i = 0; i < n; ++i) {
      if (w.hl[i]) w.vl[i] = st.bound_mult_init_val;
      if (w.hu[i]) w.vu[i] = st.bound_mult_init_val;
    }
    for (idx_t i = 0; i < m; ++i) {
      if (w.hsl[i]) w.wl[i] = st.bound_mult_init_val;
      if (w.hsu[i]) w.wu[i] = st.bound_mult_init_val;
      w.z[i] = w.wl[i] - w.wu[i];
    }
  } else {
    // init centree historique : produit vl*sl = mu exact (sur le chemin)
    for (idx_t i = 0; i < n; ++i) {
      const double eps_sl = 1e-12 * std::max(1.0, std::fabs(w.x[i]));
      if (w.hl[i]) w.vl[i] = mu / std::max(w.x[i] - P.l[i], eps_sl);
      if (w.hu[i]) w.vu[i] = mu / std::max(P.u[i] - w.x[i], eps_sl);
    }
    for (idx_t i = 0; i < m; ++i) {
      // v0.16.0 : meme garde eps que les bornes de x — sans elle, un s
      // atterrissant EXACTEMENT sur cl/cu (bound_push=0, ou ligne cl=cu
      // servie au midpoint) donne wl=mu/0=inf, et pour cl=cu z=inf-inf=NaN
      // des l'initialisation. Bit-identique partout ou le slack est sain.
      const double eps_ss = 1e-12 * std::max(1.0, std::fabs(w.s[i]));
      if (w.hsl[i]) w.wl[i] = mu / std::max(w.s[i] - P.cl[i], eps_ss);
      if (w.hsu[i]) w.wu[i] = mu / std::max(P.cu[i] - w.s[i], eps_ss);
      w.z[i] = w.wl[i] - w.wu[i];
    }
  }

  // ---------------- adaptive bound push (v0.16.0) ---------------------------
  // Etat : budget d'evenements reactifs + kappa courant (hysteresis : ne
  // redescend jamais dans un run — pas de re-detection oscillante).
  idx_t  bp_events = 0;
  double bp_kappa = st.bp_kappa_init;
  // Amplitudes de push IPOPT : p = min(kappa*max(1,|borne|), kappa*(u-l)),
  // kappa jouant les deux roles (push et frac).
  auto bp_pL = [&](double kappa, double lo, double r) {
    return std::min(kappa * std::max(1.0, std::fabs(lo)), kappa * r);
  };
  // "Poussable a kappa" : le clamp dans [lo+pL, hi-pU] deplacerait STRICTEMENT
  // la valeur. Les variables fixes (l=u) et les boites epsilon (QPNSTAIR) ne
  // le sont jamais : leur slack nul est STRUCTUREL, aucun push ne les bouge —
  // elles ne doivent ni declencher la signature ni consommer le budget.
  auto bp_movable = [&](double v, double lo, double hi, bool hL, bool hU,
                        double kappa) -> bool {
    if (hL && hU) {
      const double r = hi - lo;
      if (!(r > 0.0)) return false;
      const double pL = bp_pL(kappa, lo, r), pU = bp_pL(kappa, hi, r);
      if (lo + pL >= hi - pU) return false;          // boite trop etroite
      return (v < lo + pL) || (v > hi - pU);
    }
    if (hL) return v < lo + kappa * std::max(1.0, std::fabs(lo));
    if (hU) return v > hi - kappa * std::max(1.0, std::fabs(hi));
    return false;
  };
  // SIGNATURE "slack pathologique" : existe une composante (x ou s) dont le
  // slack RELATIF est sous bp_slack_tol ET qu'un push a kappa_max peut
  // effectivement deplacer. C'est la preuve de causalite exigee avant tout
  // push reactif — signature absente, le push ne servirait a rien.
  auto bp_signature = [&]() -> bool {
    for (idx_t i = 0; i < n; ++i) {
      double sr = INF;
      if (w.hl[i]) sr = std::min(sr, (w.x[i] - P.l[i]) / (1.0 + std::fabs(P.l[i])));
      if (w.hu[i]) sr = std::min(sr, (P.u[i] - w.x[i]) / (1.0 + std::fabs(P.u[i])));
      if (sr < st.bp_slack_tol &&
          bp_movable(w.x[i], P.l[i], P.u[i], w.hl[i], w.hu[i], st.bp_kappa_max))
        return true;
    }
    for (idx_t i = 0; i < m; ++i) {
      double sr = INF;
      if (w.hsl[i]) sr = std::min(sr, (w.s[i] - P.cl[i]) / (1.0 + std::fabs(P.cl[i])));
      if (w.hsu[i]) sr = std::min(sr, (P.cu[i] - w.s[i]) / (1.0 + std::fabs(P.cu[i])));
      if (sr < st.bp_slack_tol &&
          bp_movable(w.s[i], P.cl[i], P.cu[i], w.hsl[i], w.hsu[i], st.bp_kappa_max))
        return true;
    }
    return false;
  };
  // Push SELECTIF a kappa donne : seules les composantes pathologiques ET
  // deplacables sont poussees, et LEURS duaux sont re-inities coherents
  // (mu/slack, ou bound_mult_init_val si actif) — sinon la trust region duale
  // se bat contre des duaux corrompus herites du slack nul. Retourne le
  // nombre de composantes REELLEMENT deplacees.
  auto bp_push = [&](double kappa, double mu_now, const char* cause) -> idx_t {
    idx_t np = 0;
    double min_before = INF;
    auto zinit = [&](double sl) {
      return (st.bound_mult_init_val > 0.0) ? st.bound_mult_init_val
                                            : mu_now / std::max(sl, 1e-300);
    };
    auto push1 = [&](double& v, double lo, double hi, bool hL, bool hU) -> bool {
      double sr = INF;
      if (hL) sr = std::min(sr, (v - lo) / (1.0 + std::fabs(lo)));
      if (hU) sr = std::min(sr, (hi - v) / (1.0 + std::fabs(hi)));
      min_before = std::min(min_before, sr);
      if (sr >= st.bp_slack_tol || !bp_movable(v, lo, hi, hL, hU, kappa))
        return false;
      if (hL && hU) {
        const double r = hi - lo;
        v = clampd(v, lo + bp_pL(kappa, lo, r), hi - bp_pL(kappa, hi, r));
      } else if (hL) v = lo + kappa * std::max(1.0, std::fabs(lo));
      else if (hU)   v = hi - kappa * std::max(1.0, std::fabs(hi));
      return true;
    };
    for (idx_t i = 0; i < n; ++i) {
      if (push1(w.x[i], P.l[i], P.u[i], w.hl[i], w.hu[i])) {
        np++;
        const double eps_sl = 1e-12 * std::max(1.0, std::fabs(w.x[i]));
        if (w.hl[i]) w.vl[i] = zinit(std::max(w.x[i] - P.l[i], eps_sl));
        if (w.hu[i]) w.vu[i] = zinit(std::max(P.u[i] - w.x[i], eps_sl));
      }
    }
    for (idx_t i = 0; i < m; ++i) {
      if (push1(w.s[i], P.cl[i], P.cu[i], w.hsl[i], w.hsu[i])) {
        np++;
        if (w.hsl[i]) w.wl[i] = zinit(std::max(w.s[i] - P.cl[i], 1e-300));
        if (w.hsu[i]) w.wu[i] = zinit(std::max(P.cu[i] - w.s[i], 1e-300));
        w.z[i] = w.wl[i] - w.wu[i];
      }
    }
    if (np > 0) {
      info.n_bp_pushed += np;
      if (st.verbose >= 1)
        std::printf("[BP] event=%lld n_pushed=%lld kappa=%.1e min_slack_before=%.3e cause=%s\n",
                    (long long)bp_events, (long long)np, kappa, min_before, cause);
    }
    return np;
  };
  // Push avec escalade de kappa jusqu'au premier deplacement effectif : un
  // kappa trop petit pour depasser le slack courant est impuissant (le clamp
  // ne bouge rien) ; la signature garantit qu'un kappa <= kappa_max suffit.
  auto bp_push_escalate = [&](double mu_now, const char* cause) -> idx_t {
    idx_t np = bp_push(bp_kappa, mu_now, cause);
    while (np == 0 && bp_kappa < st.bp_kappa_max) {
      bp_kappa = std::min(bp_kappa * st.bp_kappa_mult, st.bp_kappa_max);
      np = bp_push(bp_kappa, mu_now, cause);
    }
    if (np > 0)  // hysteresis : le prochain evenement demarre un cran plus haut
      bp_kappa = std::min(bp_kappa * st.bp_kappa_mult, st.bp_kappa_max);
    return np;
  };
  // Recalcul des diagonales barriere Theta apres un push mid-run (les
  // valeurs de l'iteration courante sont perimees des que x/s bougent).
  auto bp_recompute_theta = [&]() {
    for (idx_t i = 0; i < n; ++i) {
      double t = 0.0;
      const double eps_sl = 1e-12 * std::max(1.0, std::fabs(w.x[i]));
      if (w.hl[i]) t += w.vl[i] / std::max(w.x[i] - P.l[i], eps_sl);
      if (w.hu[i]) t += w.vu[i] / std::max(P.u[i] - w.x[i], eps_sl);
      w.thx[i] = t;
    }
    for (idx_t i = 0; i < m; ++i) {
      double t = 0.0;
      // v0.16.0 : garde eps (miroir de thx) — slack exactement nul =>
      // Theta_s=inf => NaN en aval (fausse convergence a obj NaN observee).
      const double eps_ss = 1e-12 * std::max(1.0, std::fabs(w.s[i]));
      if (w.hsl[i]) t += w.wl[i] / std::max(w.s[i] - P.cl[i], eps_ss);
      if (w.hsu[i]) t += w.wu[i] / std::max(P.cu[i] - w.s[i], eps_ss);
      w.ths[i] = std::max(t, 1e-14);
    }
  };
  // ---- Restauration d'INTERIORITE (INCONDITIONNELLE, v0.16.0) --------------
  // Un slack <= 0 rend la barriere log infinie : merite INF, arithmetique
  // INF/NaN en aval, et une FAUSSE convergence observee (status=Solved,
  // obj=NaN, x=0, residus 0 exactement — qui compterait comme un succes en
  // benchmark). Un point non interieur n'est pas un etat valide d'une
  // methode barriere : la restauration ne depend PAS de bp_adaptive.
  // Seules les composantes DEPLACABLES sont restaurables ; les slacks
  // structurellement nuls (l=u, cl=cu) exigent l'epsilon-relaxation en
  // amont (probleme), hors juridiction du solveur.
  {
    bool noninterior = false;
    for (idx_t i = 0; i < n && !noninterior; ++i) {
      const bool z0 = (w.hl[i] && w.x[i] - P.l[i] <= 0.0) ||
                      (w.hu[i] && P.u[i] - w.x[i] <= 0.0);
      if (z0 && bp_movable(w.x[i], P.l[i], P.u[i], w.hl[i], w.hu[i], st.bp_kappa_max))
        noninterior = true;
    }
    for (idx_t i = 0; i < m && !noninterior; ++i) {
      const bool z0 = (w.hsl[i] && w.s[i] - P.cl[i] <= 0.0) ||
                      (w.hsu[i] && P.cu[i] - w.s[i] <= 0.0);
      if (z0 && bp_movable(w.s[i], P.cl[i], P.cu[i], w.hsl[i], w.hsu[i], st.bp_kappa_max))
        noninterior = true;
    }
    if (noninterior) bp_push_escalate(mu, "interiority");
  }
  // ---- Examen cible par variable, une seule passe (v0.17.0) ----------------
  // Remplace l'ancienne heuristique de signature (seuil 1e-10, escalade
  // reactive) par un examen DETERMINISTE, en un seul passage, recalibre sur
  // l'echelle observee empiriquement (bp_margin_tol/bp_push_kappa, defaut
  // 1e-2/5e-2 -- huit ordres de grandeur au-dessus de l'ancien seuil de
  // pathologie). Chaque composante (x_i, s_i) est examinee INDEPENDAMMENT :
  // si son slack relatif est sous bp_margin_tol, elle est poussee a
  // bp_push_kappa (formule bound_push/bound_frac standard, appliquee
  // SELECTIVEMENT) ; sinon rien. Pas d'escalade, pas de budget d'evenements,
  // pas de re-resolution -- une decision par composante, prise une fois,
  // avant la boucle principale. La boucle est parallelisable : chaque i
  // n'ecrit que ses PROPRES entrees (w.x[i]/w.vl[i]/w.vu[i], ou w.s[i]/
  // w.wl[i]/w.wu[i]/w.z[i]), aucune dependance entre iterations.
  if (st.bp_adaptive) {
    const double kappa = st.bp_push_kappa;
    const double tol = st.bp_margin_tol;
    idx_t n_pushed_x = 0, n_pushed_s = 0;
    auto zinit = [&](double sl) {
      return (st.bound_mult_init_val > 0.0) ? st.bound_mult_init_val
                                            : mu / std::max(sl, 1e-300);
    };
    #pragma omp parallel for schedule(static) reduction(+:n_pushed_x)
    for (idx_t i = 0; i < n; ++i) {
      double sr = INF;
      if (w.hl[i]) sr = std::min(sr, (w.x[i] - P.l[i]) / (1.0 + std::fabs(P.l[i])));
      if (w.hu[i]) sr = std::min(sr, (P.u[i] - w.x[i]) / (1.0 + std::fabs(P.u[i])));
      if (sr >= tol || !bp_movable(w.x[i], P.l[i], P.u[i], w.hl[i], w.hu[i], kappa)) continue;
      if (w.hl[i] && w.hu[i]) {
        const double r = P.u[i] - P.l[i];
        w.x[i] = clampd(w.x[i], P.l[i] + bp_pL(kappa, P.l[i], r), P.u[i] - bp_pL(kappa, P.u[i], r));
      } else if (w.hl[i]) w.x[i] = P.l[i] + kappa * std::max(1.0, std::fabs(P.l[i]));
      else if (w.hu[i])   w.x[i] = P.u[i] - kappa * std::max(1.0, std::fabs(P.u[i]));
      const double eps_sl = 1e-12 * std::max(1.0, std::fabs(w.x[i]));
      if (w.hl[i]) w.vl[i] = zinit(std::max(w.x[i] - P.l[i], eps_sl));
      if (w.hu[i]) w.vu[i] = zinit(std::max(P.u[i] - w.x[i], eps_sl));
      n_pushed_x++;
    }
    #pragma omp parallel for schedule(static) reduction(+:n_pushed_s)
    for (idx_t i = 0; i < m; ++i) {
      double sr = INF;
      if (w.hsl[i]) sr = std::min(sr, (w.s[i] - P.cl[i]) / (1.0 + std::fabs(P.cl[i])));
      if (w.hsu[i]) sr = std::min(sr, (P.cu[i] - w.s[i]) / (1.0 + std::fabs(P.cu[i])));
      if (sr >= tol || !bp_movable(w.s[i], P.cl[i], P.cu[i], w.hsl[i], w.hsu[i], kappa)) continue;
      if (w.hsl[i] && w.hsu[i]) {
        const double r = P.cu[i] - P.cl[i];
        w.s[i] = clampd(w.s[i], P.cl[i] + bp_pL(kappa, P.cl[i], r), P.cu[i] - bp_pL(kappa, P.cu[i], r));
      } else if (w.hsl[i]) w.s[i] = P.cl[i] + kappa * std::max(1.0, std::fabs(P.cl[i]));
      else if (w.hsu[i])   w.s[i] = P.cu[i] - kappa * std::max(1.0, std::fabs(P.cu[i]));
      if (w.hsl[i]) w.wl[i] = zinit(std::max(w.s[i] - P.cl[i], 1e-300));
      if (w.hsu[i]) w.wu[i] = zinit(std::max(P.cu[i] - w.s[i], 1e-300));
      w.z[i] = w.wl[i] - w.wu[i];
      n_pushed_s++;
    }
    if (n_pushed_x + n_pushed_s > 0) {
      info.n_bp_pushed += n_pushed_x + n_pushed_s;
      if (st.verbose >= 1)
        std::printf("[BP] examen cible : %lld composantes poussees a kappa=%.1e "
                    "(x:%lld, s:%lld, seuil=%.1e)\n",
                    (long long)(n_pushed_x + n_pushed_s), kappa,
                    (long long)n_pushed_x, (long long)n_pushed_s, tol);
      // Correction d'egalite (v0.17.0) : pousser les x_i INDEPENDAMMENT --
      // condition necessaire pour que l'examen par variable soit
      // parallelisable -- casse Ax=b des que p>0 (mesure : feas 2.4e-14 ->
      // 4.0e-3 sur un cas avec egalites). Un SEUL solve global (CG sur les
      // equations normales, comme la projection d'ancre) restaure la
      // faisabilite apres coup ; ce n'est plus per-variable donc pas
      // parallelise de la meme facon, mais c'est UNE correction, pas n.
      if (p > 0 && n_pushed_x > 0) {
        std::vector<double> resid(p, 0.0);
        { std::vector<double> ax(p, 0.0); P.A.gaxpy(w.x.data(), ax.data());
          for (idx_t i = 0; i < p; ++i) resid[i] = P.b[i] - ax[i]; }
        if (norm_inf(resid) > 1e-10) {
          std::vector<double> rr(resid), pr(resid), Ap(p), Atr(n), lam(p, 0.0);
          double rs = dot(rr, rr);
          for (int it2 = 0; it2 < 200 && std::sqrt(rs) > 1e-12 * (1.0 + norm_inf(P.b)); ++it2) {
            setzero(Atr); P.A.gaxpy_t(pr.data(), Atr.data());
            setzero(Ap);  P.A.gaxpy(Atr.data(), Ap.data());
            double alpha = rs / std::max(1e-300, dot(pr, Ap));
            for (idx_t i = 0; i < p; ++i) lam[i] += alpha * pr[i];
            for (idx_t i = 0; i < p; ++i) rr[i] -= alpha * Ap[i];
            double rs2 = dot(rr, rr);
            for (idx_t i = 0; i < p; ++i) pr[i] = rr[i] + (rs2 / std::max(1e-300, rs)) * pr[i];
            rs = rs2;
          }
          setzero(Atr); P.A.gaxpy_t(lam.data(), Atr.data());
          for (idx_t i = 0; i < n; ++i) w.x[i] += Atr[i];
          // la correction peut ressortir de la boite : re-clip interieur
          // (meme marge que le clip final de l'ancre, 1e-4 relatif).
          for (idx_t i = 0; i < n; ++i) {
            if (w.hl[i] && w.hu[i]) { double k = 1e-4 * std::max(1e-8, P.u[i] - P.l[i]); w.x[i] = clampd(w.x[i], P.l[i] + k, P.u[i] - k); }
            else if (w.hl[i]) w.x[i] = std::max(w.x[i], P.l[i] + 1e-4);
            else if (w.hu[i]) w.x[i] = std::min(w.x[i], P.u[i] - 1e-4);
          }
          if (m) { setzero(w.Cx); P.C.gaxpy(w.x.data(), w.Cx.data()); }
          if (st.verbose >= 2) {
            setzero(w.Ax); P.A.gaxpy(w.x.data(), w.Ax.data());
            double e = 0; for (idx_t i = 0; i < p; ++i) e = std::max(e, std::fabs(w.Ax[i] - P.b[i]));
            std::printf("[BP] correction d'egalite post-push : ||Ax-b||_inf apres = %.2e\n", e);
          }
        }
      }
    }
  }

  // ---------------- lazy convexification state ------------------------------
  double rho = apply_rho_model_floor(st.rho0);    // rho du MODELE DCA : fixe entre deux acceptations
  double rho_x = 0.0;      // regularisation primale de STABILITE (hors modele)
  double rho_cap = -1.0;  // computed lazily on first inertia failure
  idx_t clean_streak = 0;
  double rho_decay_floor = 0.0;  // do not decay below the last level that failed
  double floor_mu = INF;         // mu at which the floor was set
  double term_floor = 0.0;       // terminal-regime rho floor (relaxed each iter)
  idx_t last_rho_change_iter = 0;
  // convdesc: reduced-curvature certificate cache (valid while rho unchanged)
  double convdesc_lam = std::numeric_limits<double>::quiet_NaN();
  double convdesc_lam_rho = -1.0;
  AtomSet atoms;
  // ATOM_CLUSTER_PROD_V1 — zero semantic effect unless explicitly enabled.
  using AtomCluster = ProductionAtomClusterer<idx_t>;
  AtomCluster::Params atom_cluster_p;
  atom_cluster_p.w_direction = st.atom_cluster_w_direction;
  atom_cluster_p.w_support = st.atom_cluster_w_support;
  atom_cluster_p.w_curvature = st.atom_cluster_w_curvature;
  atom_cluster_p.tau_in = st.atom_cluster_tau_in;
  atom_cluster_p.tau_out = st.atom_cluster_tau_out;
  atom_cluster_p.adaptive_thresholds = st.atom_cluster_adaptive_thresholds;
  atom_cluster_p.adapt_min_samples = (std::size_t)std::max<idx_t>(1, st.atom_cluster_adapt_min_samples);
  atom_cluster_p.adapt_window = (std::size_t)std::max<idx_t>(1, st.atom_cluster_adapt_window);
  atom_cluster_p.adapt_strength = st.atom_cluster_adapt_strength;
  atom_cluster_p.max_clusters = (std::size_t)std::max<idx_t>(1, st.atom_cluster_max_reps);
  atom_cluster_p.replace_rel_quality = st.atom_cluster_replace_rel_quality;
  atom_cluster_p.replace_min_similarity = st.atom_cluster_replace_min_similarity;
  AtomCluster atom_clusters(atom_cluster_p);

  ProductionAtomSeedStopper::Params atom_cluster_stop_p;
  atom_cluster_stop_p.window = (std::size_t)std::max<idx_t>(1, st.atom_cluster_stop_window);
  atom_cluster_stop_p.min_observations = (std::size_t)std::max<idx_t>(1, st.atom_cluster_stop_min_observations);
  atom_cluster_stop_p.min_atoms = (std::size_t)std::max<idx_t>(1, st.atom_cluster_stop_min_atoms);
  atom_cluster_stop_p.max_novelty_rate = st.atom_cluster_stop_max_novelty_rate;
  atom_cluster_stop_p.max_coverage_gain = st.atom_cluster_stop_max_coverage_gain;
  atom_cluster_stop_p.max_marginal_lift_rel = st.atom_cluster_stop_max_lift_rel;
  atom_cluster_stop_p.confirmation_rounds =
      (std::size_t)std::max<idx_t>(1, st.atom_cluster_stop_confirmation_rounds);
  ProductionAtomSeedStopper atom_cluster_seed_stop(atom_cluster_stop_p);

  // ADAPTIVE_MARGINAL_SEED_V2 — no candidate ranking; commit-only O(1) controller.
  AdaptiveMarginalSeedController::Params amrs_p;
  amrs_p.warmup = (std::size_t)std::max<idx_t>(1, st.adaptive_seed_return_warmup);
  amrs_p.window = (std::size_t)std::max<idx_t>(1, st.adaptive_seed_return_window);
  amrs_p.confirmations = (std::size_t)std::max<idx_t>(1, st.adaptive_seed_return_confirmations);
  amrs_p.alpha_coverage = st.adaptive_seed_return_alpha_coverage;
  amrs_p.alpha_lift = st.adaptive_seed_return_alpha_lift;
  amrs_p.coverage_min = st.adaptive_seed_return_coverage_min;
  amrs_p.coverage_max = st.adaptive_seed_return_coverage_max;
  amrs_p.lift_min = st.adaptive_seed_return_lift_min;
  amrs_p.lift_max = st.adaptive_seed_return_lift_max;
  AdaptiveMarginalSeedController adaptive_seed_return(amrs_p);

  // ADAPTIVE_MARGINAL_RETURN_V21 — global commit-time controller.
  AdaptiveMarginalReturnController::Params amr21_p;
  amr21_p.warmup = (std::size_t)std::max<idx_t>(1, st.adaptive_return_warmup);
  amr21_p.window = (std::size_t)std::max<idx_t>(1, st.adaptive_return_window);
  amr21_p.confirmations = (std::size_t)std::max<idx_t>(1, st.adaptive_return_confirmations);
  amr21_p.alpha_lift = st.adaptive_return_alpha_lift;
  amr21_p.alpha_effective = st.adaptive_return_alpha_effective;
  amr21_p.lift_min = st.adaptive_return_lift_min;
  amr21_p.lift_max = st.adaptive_return_lift_max;
  amr21_p.effective_min = st.adaptive_return_effective_min;
  amr21_p.effective_max = st.adaptive_return_effective_max;
  amr21_p.cooldown_factor = st.adaptive_return_cooldown_factor;
  amr21_p.cooldown_max_factor = st.adaptive_return_cooldown_max_factor;
  amr21_p.strong_factor = st.adaptive_return_strong_factor;
  // ADAPTIVE_MARGINAL_RETURN_V22 seed decision parameters.
  amr21_p.seed_alpha_coverage = st.adaptive_return_seed_alpha_coverage;
  amr21_p.seed_alpha_lift = st.adaptive_return_seed_alpha_lift;
  amr21_p.seed_coverage_min = st.adaptive_return_seed_coverage_min;
  amr21_p.seed_coverage_max = st.adaptive_return_seed_coverage_max;
  amr21_p.seed_lift_min = st.adaptive_return_seed_lift_min;
  amr21_p.seed_lift_max = st.adaptive_return_seed_lift_max;
  amr21_p.seed_absolute_enable = st.adaptive_return_seed_absolute_enable;
  amr21_p.seed_absolute_window = (std::size_t)std::max<idx_t>(1, st.adaptive_return_seed_absolute_window);
  amr21_p.seed_absolute_confirmations = (std::size_t)std::max<idx_t>(1, st.adaptive_return_seed_absolute_confirmations);
  amr21_p.seed_absolute_coverage = st.adaptive_return_seed_absolute_coverage;
  amr21_p.seed_absolute_lift = st.adaptive_return_seed_absolute_lift;
  // ADAPTIVE_MARGINAL_RETURN_V23 plateau parameters.
  amr21_p.seed_plateau_enable = st.adaptive_return_seed_plateau_enable;
  amr21_p.seed_plateau_horizon = (std::size_t)std::max<idx_t>(1, st.adaptive_return_seed_plateau_horizon);
  amr21_p.seed_plateau_min_commits = (std::size_t)std::max<idx_t>(1, st.adaptive_return_seed_plateau_min_commits);
  amr21_p.seed_plateau_confirmations = (std::size_t)std::max<idx_t>(1, st.adaptive_return_seed_plateau_confirmations);
  amr21_p.seed_plateau_coverage_per_commit = st.adaptive_return_seed_plateau_coverage_per_commit;
  amr21_p.seed_plateau_lift = st.adaptive_return_seed_plateau_lift;


  AdaptiveMarginalReturnController adaptive_return(amr21_p);
  bool amr21_seed_stop_requested = false;
  auto sync_adaptive_return_info = [&]() {
    const auto &as = adaptive_return.stats();
    info.n_adaptive_return_commits = (idx_t)as.commits_total;
    info.n_adaptive_return_scored = (idx_t)as.scored_observations;
    info.n_adaptive_return_seed_commits = (idx_t)as.seed_commits;
    info.n_adaptive_return_event_oracle_commits = (idx_t)as.event_oracle_commits;
    info.n_adaptive_return_event_direct_commits = (idx_t)as.event_direct_commits;
    info.n_adaptive_return_update_commits = (idx_t)as.update_commits;
    info.n_adaptive_return_low_rounds = (idx_t)as.low_return_rounds;
    info.n_adaptive_return_seed_stop_requests = (idx_t)as.seed_stop_requests;
    info.n_adaptive_return_cooldowns = (idx_t)as.cooldowns;
    info.n_adaptive_return_reactivations = (idx_t)as.reactivations;
    info.n_adaptive_return_model_resets = (idx_t)as.model_resets;
    info.adaptive_return_last_source = (idx_t)as.last_source_code;
    info.adaptive_return_lift_ref = as.lift_ref;
    info.adaptive_return_effective_ref = as.effective_ref;
    info.adaptive_return_lift_threshold = as.lift_threshold;
    info.adaptive_return_effective_threshold = as.effective_threshold;
    info.adaptive_return_last_lift_rel = as.last_lift_rel;
    info.adaptive_return_last_effective_rel = as.last_effective_rel;
    info.adaptive_return_recent_lift_median = as.recent_lift_median;
    info.adaptive_return_recent_effective_median = as.recent_effective_median;
    info.adaptive_return_cooldown_multiplier = as.cooldown_multiplier;
    // ADAPTIVE_MARGINAL_RETURN_V22 seed telemetry sync.
    info.n_adaptive_return_seed_scored_rounds = (idx_t)as.seed_scored_rounds;
    info.n_adaptive_return_seed_unreliable_rounds = (idx_t)as.seed_unreliable_rounds;
    info.n_adaptive_return_seed_low_relative_rounds = (idx_t)as.seed_low_relative_rounds;
    info.n_adaptive_return_seed_low_absolute_rounds = (idx_t)as.seed_low_absolute_rounds;
    info.adaptive_return_seed_relative_confirmations = (idx_t)as.seed_relative_confirmations;
    info.adaptive_return_seed_absolute_confirmations = (idx_t)as.seed_absolute_confirmations;
    info.adaptive_return_seed_stop_reason = (idx_t)as.seed_last_stop_reason;
    info.adaptive_return_seed_references_ready = as.seed_references_ready;
    info.adaptive_return_seed_last_coverage = as.seed_last_coverage;
    info.adaptive_return_seed_last_coverage_gain = as.seed_last_coverage_gain;
    info.adaptive_return_seed_last_lift_rel = as.seed_last_lift_rel;
    info.adaptive_return_seed_coverage_ref = as.seed_coverage_ref;
    info.adaptive_return_seed_lift_ref = as.seed_lift_ref;
    info.adaptive_return_seed_coverage_threshold = as.seed_coverage_threshold;
    info.adaptive_return_seed_lift_threshold = as.seed_lift_threshold;
    info.adaptive_return_seed_recent_coverage_median = as.seed_recent_coverage_median;
    info.adaptive_return_seed_recent_lift_median = as.seed_recent_lift_median;
    info.adaptive_return_seed_absolute_recent_coverage_median = as.seed_absolute_recent_coverage_median;
    info.adaptive_return_seed_absolute_recent_lift_median = as.seed_absolute_recent_lift_median;
    info.adaptive_return_seed_stop_coverage = as.seed_stop_coverage;
    info.adaptive_return_seed_stop_coverage_gain = as.seed_stop_coverage_gain;
    info.adaptive_return_seed_stop_lift_rel = as.seed_stop_lift_rel;
    // ADAPTIVE_MARGINAL_RETURN_V23 plateau telemetry sync.
    info.n_adaptive_return_seed_progress_rounds = (idx_t)as.seed_progress_rounds;
    info.n_adaptive_return_seed_plateau_evaluations = (idx_t)as.seed_plateau_evaluations;
    info.n_adaptive_return_seed_low_plateau_rounds = (idx_t)as.seed_low_plateau_rounds;
    info.adaptive_return_seed_plateau_confirmations = (idx_t)as.seed_plateau_confirmations;
    info.adaptive_return_seed_plateau_ready = as.seed_plateau_ready;
    info.adaptive_return_seed_last_reliable_coverage = as.seed_last_reliable_coverage;
    info.adaptive_return_seed_plateau_horizon_gain = as.seed_plateau_horizon_gain;
    info.adaptive_return_seed_plateau_gain_per_commit = as.seed_plateau_gain_per_commit;
    info.adaptive_return_seed_plateau_lift_median = as.seed_plateau_lift_median;
    info.adaptive_return_seed_stop_plateau_gain_per_commit = as.seed_stop_plateau_gain_per_commit;
    info.adaptive_return_seed_stop_plateau_lift_median = as.seed_stop_plateau_lift_median;


    info.adaptive_return_event_spacing_current = (idx_t)adaptive_return.event_spacing(
        (std::size_t)std::max<idx_t>(1, st.atom_event_every));
  };



  double probe_hist[3] = {0, 0, 0}; idx_t probe_pos = 0;
  idx_t last_atom_iter = -1000000;
  double rho_pending = -1.0;   // candidat de decroissance du rho modele
  bool seeding_phase = false;
  bool seed_coverage_reached = false;
  bool atom_growth_frozen = false;
  idx_t last_event_atom_iter = -1000000;
  idx_t last_coherence_iter = -1000000;
  idx_t dca_banner_rows = 0, dca_stall_streak = 0;
  idx_t iters_since_last_accept = 0;
  bool dca_converged_flag = false;
  std::deque<double> dca_delta_hist;
  idx_t ipopt_acc_streak = 0;
  std::deque<double> ipopt_E_hist;
  auto Ra_ok_for_anchor = [&]() {
    Residuals Rq = compute_residuals(P, w, mu);
    return Rq.feas_rel <= 100.0 * st.eps_feas;   // ancre seulement si faisable
  };
  std::vector<idx_t> atom_brake;
  // Evenement en ATTENTE : detection n'importe quand, insertion differee au
  // prochain accept (x = x_ref => terme atome NUL au point courant, merite
  // continu — la lecon du zip) ; timeout 15 iterations pour les deadlocks.
  bool evt_pending_probe = false; char evt_pending_tag = 0;
  idx_t evt_pending_since = 0;
  std::vector<idx_t> evt_ui; std::vector<double> evt_uv; double evt_beta = 0.0;
  bool evt_has_direction = false;
  idx_t inertia_corr_streak = 0; idx_t prev_inertia_corr = 0;

  idx_t rho_bisect_left = 0;
  double rho_cut = 2.0;                // cut adaptatif des propositions
  idx_t rho_spacing_cur = st.rho_trial_every;
  idx_t dca_inner = 0; double dca_model_start = std::numeric_limits<double>::infinity();
  bool terminal_polish_active = false;
  // GUARDED_PROGRESSIVE_FALLBACK_V269
  bool guarded_progressive_active = false;
  // PROGRESSIVE_TAKEOVER_V270
  // Once event-driven discovery has retained no new correction for a full
  // event horizon at low rank, progressive discovery takes over exclusively.
  // last_event_atom_iter is not suitable for this decision because an event
  // probe updates it before a correction is necessarily retained.
  bool guarded_progressive_takeover = false;
  idx_t guarded_progressive_start_iter = -1;
  idx_t guarded_progressive_start_rank = 0;
  idx_t guarded_last_rank_seen = -1;
  idx_t guarded_last_rank_change_iter = 0;
  constexpr idx_t guarded_progressive_rank_gate = 5;
  // EARLY_LOWRANK_TAKEOVER_V271
  // If the post-seed model is still low-rank, let the progressive
  // channel choose subsequent corrections before the first event
  // correction can alter that low-rank trajectory.
  constexpr idx_t guarded_progressive_max_new_atoms = 7;
  // TERMINAL_QUALIFICATION_V261: DCA-stall is an operational trigger, not
  // a first-order certificate. Only an explicit recovery episode may defer
  // the historical DCA stop, and that episode is transactionally bounded.
  constexpr double terminal_recovery_comp_trigger = 1.0e-2;
  constexpr idx_t terminal_recovery_budget = 50;
  bool terminal_recovery_active = false;
  bool terminal_recovery_attempted = false;
  bool terminal_recovery_entry_valid = false;
  idx_t terminal_recovery_start = -1;
  idx_t terminal_recovery_deadline = -1;
  Work terminal_recovery_entry_work;
  double terminal_recovery_entry_mu = 0.0;
  double terminal_recovery_entry_rho = 0.0;
  double terminal_recovery_entry_rho_x = 0.0;
  double terminal_recovery_entry_rho_pending = -1.0;
  double terminal_recovery_entry_rho_cut = 2.0;
  bool event_feas_ok = false;   // maj a chaque iteration (residus)
  auto event_budget_ok = [&]() {
    // quasi-faisabilite requise : en phase 1 (iterate loin, merite domine
    // par la faisabilite, face provisoire), un atome mesure la cible de la
    // courbure sur une face sans pertinence — la lecon MPC10 en atomique.
    return st.atom_event_enable && !guarded_progressive_takeover && atoms_enabled && !atom_growth_frozen && event_feas_ok &&
           atoms.size() < max_atoms_eff && !terminal_polish_active &&
           info.iters >= last_event_atom_iter + st.atom_event_every;
  };
  auto amr21_atom_event_budget_ok = [&]() {
    if (!event_budget_ok()) return false;
    if (!st.adaptive_marginal_return_enable) return true;
    return info.iters >= last_event_atom_iter +
        (idx_t)adaptive_return.event_spacing(
            (std::size_t)std::max<idx_t>(1, st.atom_event_every));
  };

  Work terminal_best_work; double terminal_best_stat = std::numeric_limits<double>::infinity();
  bool terminal_best_valid = false; idx_t terminal_reanchors = 0;
  idx_t polish_cooldown_until = 0;
  bool best_feas_valid = false;
  std::vector<double> mcl_(n, 0.0), mcu_(n, 0.0), msl_(m, 0.0), msu_(m, 0.0);
  idx_t mu_last_change_iter = 0;
  bool mehrotra_corr_used = false;
  double frozen_sig = std::numeric_limits<double>::quiet_NaN(); idx_t frozen_count = 0;
  idx_t escape_count = 0;
  // pre-alloues AVANT la boucle : des allocations pendant le run alterent
  // (mecanisme non elucide, ASAN propre) les residus finaux des tests 2-3.
  std::vector<double> bx_(n), bs_(m), by_(p), bz_(m), bvl_(n), bvu_(n), bwl_(m), bwu_(m);
  // Gershgorin plein-espace de Q : borne inferieure GARANTIE de lambda_min ;
  // cible de saut sur echec de courbure persistant (zip : bisection bornee
  // par la borne de Gershgorin "guaranteed sufficient").
  double gersh_lb = 0.0;
  {
    std::vector<double> dg(n, 0.0), radg(n, 0.0);
    for (idx_t j = 0; j < n; ++j)
      for (idx_t pp2 = P.Q.U.colptr[j]; pp2 < P.Q.U.colptr[j + 1]; ++pp2) {
        idx_t i2 = P.Q.U.rowind[pp2];
        double a = P.Q.U.values[pp2];
        if (i2 == j) dg[j] += a;
        else { radg[i2] += std::fabs(a); radg[j] += std::fabs(a); }
      }
    for (idx_t j = 0; j < n; ++j) gersh_lb = std::min(gersh_lb, dg[j] - radg[j]);
  }

  // CONST/DYN initialization from the article.  The shift is computed once
  // before the main iteration from a full-space estimate of lambda_min(Q).
  // In CONST it is thereafter immutable as a DCA-model parameter; rho_x may
  // still vary because it is only a numerical linear-algebra safeguard.
  double rho_init_lmin = std::numeric_limits<double>::quiet_NaN();
  const char* rho_init_source = "legacy";
  if (rho_spectral_init) {
    RhoInitMethod method = st.rho_init_method;
    if (method == RhoInitMethod::Hint && !std::isfinite(st.lambda_min_hint))
      method = RhoInitMethod::Auto;

    if (method == RhoInitMethod::User) {
      rho = std::clamp(st.rho0, 0.0, st.rho_abs_max);
      rho_init_source = "user";
    } else {
      if ((method == RhoInitMethod::Auto || method == RhoInitMethod::Hint) &&
          std::isfinite(st.lambda_min_hint)) {
        rho_init_lmin = st.lambda_min_hint * s_f;
        rho_init_source = "hint";
      } else if (method == RhoInitMethod::Gershgorin) {
        rho_init_lmin = gersh_lb;
        rho_init_source = "gershgorin";
      } else {
        info.n_lanczos_calls++;
        auto rep_init = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit,
                                               st.lanczos_tol, 37, nullptr);
        if (std::isfinite(rep_init.lmin)) {
          rho_init_lmin = rep_init.lmin;
          rho_init_source = rep_init.converged ? "lanczos" : "lanczos-nonconverged";
        } else {
          rho_init_lmin = gersh_lb;
          rho_init_source = "gershgorin-fallback";
        }
      }
      rho = spectral_shift_from_lmin(rho_init_lmin, st.rho_init_rel_margin,
                                     st.rho_init_abs_margin);
      rho = apply_rho_model_floor(std::clamp(rho, 0.0, st.rho_abs_max));
      info.lambda_min_reduced = rho_init_lmin;
    }
    if (st.verbose >= 1)
      std::printf("[LazyDCA-rho] mode=%s init=%s lambda_hat=%.6e -> rho0=%.6e "
                  "(rel_margin=%.1e abs_margin=%.1e)\n",
                  regularization_mode_name(st.regularization_mode), rho_init_source,
                  rho_init_lmin, rho, st.rho_init_rel_margin, st.rho_init_abs_margin);
  } else if (st.verbose >= 2 && reg_controlled) {
    std::printf("[LazyDCA-rho] mode=%s : initialisation lazy/low-rank historique\n",
                regularization_mode_name(st.regularization_mode));
  }

  std::vector<double> x_ref;   // reference DCA (mode integre du zip)
  double ref_obj = INF, best_feas_obj = INF;
  idx_t accepts = 0, rejects = 0;

  // INERTIA_COMPLETE_AMRS_V24 — dual-path state
  // INERTIA_COMPLETE_AMRS_V24_PATCHER_FIX_V245: legacy and complete.
  std::vector<double> momentum_v(n, 0.0), momentum_force(n, 0.0);
  std::vector<double> momentum_metric_beta;
  double momentum_metric_rho = std::numeric_limits<double>::quiet_NaN();
  double momentum_theta_cur = std::clamp(st.dca_momentum_theta,
      std::max(0.0, st.dca_momentum_theta_min),
      std::max(std::max(0.0, st.dca_momentum_theta_min), st.dca_momentum_theta_max));
  double momentum_theta_global_complete = 0.0;
  double momentum_theta_rho_complete = 0.0;
  std::vector<double> momentum_theta_atom_complete;
  std::vector<char> momentum_reset_atom_complete;
  bool momentum_have_step = false;
  bool momentum_suppressed_epoch = false;
  idx_t momentum_counted_iter = -1;
  std::deque<double> dca_accept_obj_hist;

  auto complete_headline_theta = [&]() {
    if (!st.dca_momentum_complete) return momentum_theta_cur;
    if (st.dca_momentum == DcaMomentumMode::InertialAtomic) {
      double z=momentum_theta_rho_complete;
      for(double t:momentum_theta_atom_complete) z=std::max(z,t);
      return z;
    }
    return momentum_theta_global_complete;
  };
  auto snapshot_momentum_metric = [&]() {
    momentum_metric_rho = rho; momentum_metric_beta = atoms.beta;
    if (st.dca_momentum_complete && momentum_theta_atom_complete.size() < atoms.beta.size())
      momentum_theta_atom_complete.resize(atoms.beta.size(),0.0);
    if (st.dca_momentum_complete && momentum_reset_atom_complete.size() < atoms.beta.size())
      momentum_reset_atom_complete.resize(atoms.beta.size(),0);
  };
  auto full_reset_complete = [&]() {
    momentum_theta_global_complete=0.0; momentum_theta_rho_complete=0.0;
    momentum_theta_atom_complete.assign(atoms.beta.size(),0.0);
    momentum_reset_atom_complete.assign(atoms.beta.size(),0);
  };
  auto reset_momentum_epoch = [&](const char* why) {
    if (st.dca_momentum == DcaMomentumMode::Original) return;
    std::fill(momentum_force.begin(), momentum_force.end(), 0.0);
    std::fill(momentum_v.begin(), momentum_v.end(), 0.0);
    momentum_have_step=false; momentum_suppressed_epoch=true;
    if (st.dca_momentum_complete) full_reset_complete();
    info.n_momentum_resets++;
    if (st.verbose>=2) std::printf("[LazyDCA-momentum] reset d'epoque (%s)\n",why);
  };
  auto sync_complete_metric = [&]() {
    if (!st.dca_momentum_complete || st.dca_momentum==DcaMomentumMode::Original) {
      if (st.dca_momentum_reset_on_metric_change) {
        bool ch=!std::isfinite(momentum_metric_rho) || momentum_metric_rho!=rho ||
                momentum_metric_beta.size()!=atoms.beta.size();
        if(!ch) for(size_t j=0;j<atoms.beta.size();++j) if(momentum_metric_beta[j]!=atoms.beta[j]){ch=true;break;}
        if(ch){ reset_momentum_epoch("changement de M_k"); snapshot_momentum_metric(); }
      }
      return;
    }
    if (!std::isfinite(momentum_metric_rho)) { snapshot_momentum_metric(); return; }
    const double oldrho=momentum_metric_rho; const bool rhochg=oldrho!=rho;
    const size_t oldn=momentum_metric_beta.size(), newn=atoms.beta.size();
    bool atomchg=oldn!=newn, betachg=false; std::vector<size_t> changed;
    const size_t q=std::min(oldn,newn);
    for(size_t j=0;j<q;++j) if(momentum_metric_beta[j]!=atoms.beta[j]){betachg=true;changed.push_back(j);}
    if (!(rhochg||atomchg||betachg)) return;
    const double cap=complete_theta_cap(st);
    if (rhochg) {
      if (st.dca_momentum_rho_policy==DcaMomentumRhoPolicy::Restart) {
        if (st.dca_momentum==DcaMomentumMode::InertialAtomic && st.dca_momentum_selective_restart)
          momentum_theta_rho_complete=0.0;
        else full_reset_complete();
        info.n_momentum_restarts_rho++;
      } else if (st.dca_momentum_rho_policy==DcaMomentumRhoPolicy::Transport) {
        bool cp=false;
        if (st.dca_momentum==DcaMomentumMode::InertialAtomic)
          momentum_theta_rho_complete=complete_transport_theta(momentum_theta_rho_complete,oldrho,rho,st,&cp);
        else momentum_theta_global_complete=complete_transport_theta(momentum_theta_global_complete,oldrho,rho,st,&cp);
        if(cp) info.n_theta_capped_by_lyapunov++; info.n_momentum_rho_transports++;
      } else { info.n_momentum_rho_keeps++; }
    }
    if (atomchg || betachg) {
      if (st.dca_momentum==DcaMomentumMode::InertialGlobal ||
          (st.dca_momentum==DcaMomentumMode::InertialEuclidean &&
           st.dca_momentum_gamma_mode==DcaEuclideanGammaMode::MetricRayleigh) ||
          !st.dca_momentum_selective_restart) {
        full_reset_complete(); info.n_momentum_restarts_atom++;
      } else if (st.dca_momentum==DcaMomentumMode::InertialAtomic) {
        momentum_theta_atom_complete.resize(newn,0.0); momentum_reset_atom_complete.resize(newn,0);
        if(newn>oldn) for(size_t j=oldn;j<newn;++j) momentum_reset_atom_complete[j]=1;
        for(size_t j:changed) if(j<momentum_theta_atom_complete.size()){momentum_theta_atom_complete[j]=0.0;momentum_reset_atom_complete[j]=1;}
        if(atomchg) info.n_momentum_restarts_atom++; if(betachg) info.n_momentum_restarts_beta++;
      }
    }
    momentum_theta_global_complete=std::min(cap,momentum_theta_global_complete);
    momentum_theta_rho_complete=std::min(cap,momentum_theta_rho_complete);
    for(double& t:momentum_theta_atom_complete) t=std::min(cap,t);
    snapshot_momentum_metric();
  };
  auto shrink_complete = [&](const char* why) {
    if(!st.dca_momentum_complete) return false;
    const double a=(st.dca_momentum_backtrack>0.0&&st.dca_momentum_backtrack<1.0)?st.dca_momentum_backtrack:0.5;
    bool ch=false; auto sh=[&](double& t){double o=t;t*=a;if(t<st.dca_momentum_theta_min)t=0.0;ch|=t<o;};
    sh(momentum_theta_global_complete); sh(momentum_theta_rho_complete);
    for(double& t:momentum_theta_atom_complete) sh(t);
    if(ch){info.n_momentum_backtracks++; if(st.verbose>=2) std::printf("[LazyDCA-momentum] complete shrink (%s)\n",why);}
    return ch;
  };
  auto refresh_momentum_force = [&]() {
    sync_complete_metric();
    std::fill(momentum_force.begin(),momentum_force.end(),0.0);
    const bool enabled=st.dca_momentum!=DcaMomentumMode::Original && momentum_have_step &&
                       !momentum_suppressed_epoch && accepts>=st.dca_momentum_start;
    if(!enabled){info.momentum_force_inf_final=0.0;return;}
    if(!st.dca_momentum_complete){
      double theta_eff=std::clamp(momentum_theta_cur,0.0,std::max(0.0,st.dca_momentum_theta_max));
      if(st.dca_acceptance==DcaAcceptanceMode::Lyapunov){const double tb=st.dca_lyapunov_theta_bar>0.0?st.dca_lyapunov_theta_bar:st.dca_momentum_theta;theta_eff=std::min(theta_eff,std::max(0.0,tb));}
      double gamma_eff=std::max(0.0,st.dca_momentum_gamma);
      if(st.dca_momentum==DcaMomentumMode::InertialEuclidean&&st.dca_momentum_gamma_mode==DcaEuclideanGammaMode::MetricRayleigh){const double vv=dot(momentum_v,momentum_v);if(vv>1e-300)gamma_eff=dca_metric_quad(momentum_v,rho,atoms)/vv;}
      gamma_eff=std::clamp(gamma_eff,std::max(0.0,st.dca_momentum_gamma_min),std::max(std::max(0.0,st.dca_momentum_gamma_min),st.dca_momentum_gamma_max));
      build_dca_momentum_force(st.dca_momentum,momentum_v,rho,atoms,theta_eff,gamma_eff,st.dca_momentum_rho_factor,st.dca_momentum_atom_align_floor,st.dca_momentum_atom_align_power,momentum_force);
      info.momentum_theta_final=theta_eff;info.momentum_gamma_final=gamma_eff;
    } else {
      const double cap=complete_theta_cap(st);
      if(cap<st.dca_momentum_theta_max) info.n_theta_capped_by_lyapunov++;
      if(st.dca_momentum==DcaMomentumMode::InertialEuclidean){
        const double th=std::min(cap,momentum_theta_global_complete);
        double gamma=std::max(0.0,st.dca_momentum_gamma);
        if(st.dca_momentum_gamma_mode==DcaEuclideanGammaMode::MetricRayleigh){const double vv=dot(momentum_v,momentum_v);if(vv>1e-300)gamma=dca_metric_quad(momentum_v,rho,atoms)/vv;}
        gamma=std::clamp(gamma,std::max(0.0,st.dca_momentum_gamma_min),std::max(std::max(0.0,st.dca_momentum_gamma_min),st.dca_momentum_gamma_max));
        bool gc=false;gamma=complete_euclidean_gamma_cap(gamma,th,rho,st,&gc);if(gc)info.n_euclidean_gamma_capped++;
        for(size_t i=0;i<momentum_force.size();++i)momentum_force[i]=th*gamma*momentum_v[i];info.momentum_gamma_final=gamma;
      } else if(st.dca_momentum==DcaMomentumMode::InertialGlobal){
        const double th=std::min(cap,momentum_theta_global_complete);
        for(size_t i=0;i<momentum_force.size();++i)momentum_force[i]=th*rho*momentum_v[i];
        std::vector<double> tmp(momentum_force.size(),0.0);atoms.apply(momentum_v.data(),tmp.data());for(size_t i=0;i<tmp.size();++i)momentum_force[i]+=th*tmp[i];
      } else {
        momentum_theta_atom_complete.resize(atoms.beta.size(),0.0);
        complete_atomic_force(momentum_v,atoms,rho,std::min(cap,momentum_theta_rho_complete),momentum_theta_atom_complete,momentum_force);
      }
      bool fc=false;info.momentum_force_inf_final=complete_cap_force(momentum_force,momentum_v,atoms,rho,st.dca_momentum_force_rel_max,&fc);if(fc)info.n_momentum_force_capped++;
      const double base=complete_metric_force_inf(momentum_v,atoms,rho);info.momentum_force_rel=base>0.0?info.momentum_force_inf_final/base:0.0;
      info.momentum_theta_final=complete_headline_theta();info.momentum_theta_rho_final=momentum_theta_rho_complete;
      double mn=momentum_theta_atom_complete.empty()?0.0:momentum_theta_atom_complete[0],mx=0.0,sm=0.0;for(double t:momentum_theta_atom_complete){mn=std::min(mn,t);mx=std::max(mx,t);sm+=t;}
      info.momentum_theta_atom_min=mn;info.momentum_theta_atom_max=mx;info.momentum_theta_atom_mean=momentum_theta_atom_complete.empty()?0.0:sm/momentum_theta_atom_complete.size();
      info.n_momentum_complete_builds++;
    }
    info.momentum_force_inf_final=norm_inf(momentum_force);
    if(info.momentum_force_inf_final>0.0&&momentum_counted_iter!=info.iters){info.n_momentum_steps++;momentum_counted_iter=info.iters;}
  };
  double stat_model_rel_iter = INF;
  // Residus du probleme ORIGINAL (affichage, arret, acceptation) : les duals
  // scales se convertissent par y = s_eq.*y_s/s_f, z = s_in.*z_s/s_f,
  // vl/vu = vl_s/s_f (stationnarite scalee divisee par s_f).
  std::vector<double> y_o(p), z_o(m), vl_o(n), vu_o(n);
  auto residuals_original = [&](double mu_cur) -> Residuals {
    if (!do_scale) return compute_residuals(P_in, w, mu_cur);
    for (idx_t i = 0; i < p; ++i) y_o[i] = s_eq[i] * w.y[i] / s_f;
    for (idx_t i = 0; i < m; ++i) z_o[i] = s_in[i] * w.z[i] / s_f;
    for (idx_t i = 0; i < n; ++i) { vl_o[i] = w.vl[i] / s_f; vu_o[i] = w.vu[i] / s_f; }
    std::swap(w.y, y_o); std::swap(w.z, z_o); std::swap(w.vl, vl_o); std::swap(w.vu, vu_o);
    Residuals R2 = compute_residuals(P_in, w, mu_cur);
    std::swap(w.y, y_o); std::swap(w.z, z_o); std::swap(w.vl, vl_o); std::swap(w.vu, vu_o);
    return R2;
  };
  double last_lambda_est = std::numeric_limits<double>::quiet_NaN();  // zip: m_lazydca_last_lambda_est
  bool hint_valid = true;      // le hint lambda_min ne vaut que pour H_k = 0
  idx_t accepts_since_atoms = 0;
  bool atoms_dirty = false;    // pattern KKT a reconstruire
  // Last accepted DCA displacement used by the adaptive atom oracle.  The
  // oracle falls back to the objective gradient during initial seeding.
  std::vector<double> atom_oracle_last_step(n, 0.0);
  bool atom_oracle_last_step_valid = false;

  // ---- Classification budget epuise (v0.16.2) -------------------------------
  // Facteur commun a TOUS les sites ou une ressource d'escalade (rho, sigma)
  // atteint son plafond. Ne pas confondre "budget epuise" et "echec reel" :
  // si le point courant est FAISABLE, l'algorithme n'a pas rate le probleme,
  // il a simplement echoue a CERTIFIER davantage un point deja essentiellement
  // resolu (LICQ probablement en defaut, continuum dual non borne -- meme
  // diagnostic que le chemin sigma>1e14 pre-existant, cf. NCVXQP7). Ce n'est
  // PAS un succes complet (stat/comp peuvent rester mauvais) : le statut
  // DualDegenerate le dit explicitement, distinct de Solved. La restauration
  // du meilleur point faisable (bloc finish:) s'applique ensuite normalement
  // sur les deux issues.
  auto classify_budget_exhausted = [&](const char* budget_name, const std::string& raw_msg) {
    Residuals Rex = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
    if (Rex.feas_rel <= 100.0 * st.eps_feas) {
      info.status = Status::DualDegenerate;
      info.message = std::string("primal converged (feasible, stable objective) but multipliers "
                     "unbounded (LICQ likely deficient) -- ") + budget_name +
                     " exhausted, best feasible point returned";
      if (st.verbose >= 1)
        std::printf("[LazyDCA-budget] %s epuise, point FAISABLE (feas_rel=%.2e) -- "
                    "reclassifie DualDegenerate (pas un echec numerique)\n", budget_name, Rex.feas_rel);
    } else {
      info.status = Status::NumericalError;
      info.message = raw_msg;
    }
  };

  // Rafraichissement moindres-carres des multiplicateurs d'egalite :
  //   y* = argmin_y || g - A'y ||,  g = Qx + q - C'z - vl + vu
  // CG sur les equations normales (A A') — casse les deadlocks ou l'erreur de
  // stationnarite est entierement duale (x fige au minimum du merite).
  auto ls_multiplier_refresh = [&]() {
    if (p == 0) return;
    // INVARIANT : un refresh qui AGGRAVE le residu dual n'est jamais commis.
    // A mu minuscule, vl/vu pres des bornes quasi-actives empoisonnent g et
    // le CG renvoie un y ~ 3e6 : le "rafraichissement" multipliait inf_du
    // par 1e7 (cycle limite periode ~18 photographie sur NCVXQP2/7 :
    // streak Armijo 2 -> refresh -> explosion -> re-centrage mu -> rebelote).
    std::vector<double> y_save = w.y;
    const double stat_before = compute_residuals(P, w, mu).stat_abs;
    std::vector<double> g(n, 0.0);
    setzero(w.Qx); P.Q.gaxpy_sym(w.x.data(), w.Qx.data());
    setzero(w.Ctz); if (m) P.C.gaxpy_t(w.z.data(), w.Ctz.data());
    for (idx_t i = 0; i < n; ++i) g[i] = w.Qx[i] + P.q[i] - w.Ctz[i] - w.vl[i] + w.vu[i];
    std::vector<double> r(p, 0.0), pr(p), Ap(p), Atr(n), yn(p, 0.0);
    // r = A g - (AA') y0, avec y0 = y courant
    P.A.gaxpy(g.data(), r.data());
    { std::vector<double> tmp(n, 0.0), tmp2(p, 0.0);
      P.A.gaxpy_t(w.y.data(), tmp.data());
      P.A.gaxpy(tmp.data(), tmp2.data());
      for (idx_t i = 0; i < p; ++i) r[i] -= tmp2[i]; }
    yn = w.y; pr = r;
    double rs = dot(r, r);
    for (int it = 0; it < 100 && std::sqrt(rs) > 1e-12 * (1.0 + norm_inf(g)); ++it) {
      setzero(Atr); P.A.gaxpy_t(pr.data(), Atr.data());
      setzero(Ap);  P.A.gaxpy(Atr.data(), Ap.data());
      double alpha = rs / std::max(1e-300, dot(pr, Ap));
      for (idx_t i = 0; i < p; ++i) yn[i] += alpha * pr[i];
      for (idx_t i = 0; i < p; ++i) r[i] -= alpha * Ap[i];
      double rs2 = dot(r, r);
      for (idx_t i = 0; i < p; ++i) pr[i] = r[i] + (rs2 / std::max(1e-300, rs)) * pr[i];
      rs = rs2;
    }
    w.y = yn;
    if (st.verbose >= 2) std::printf("[LazyDCA-stall] rafraichissement LS des multiplicateurs y\n");

    const double stat_after = compute_residuals(P, w, mu).stat_abs;
    if (!(stat_after <= 2.0 * stat_before)) {
      w.y = y_save;   // refresh nuisible : retour arriere
      if (st.verbose >= 2)
        std::printf("[LazyDCA-refresh] REVERT : stat %.3e -> %.3e (degradation)\n",
                    stat_before, stat_after);
    } else if (st.verbose >= 2 && stat_after < 0.5 * stat_before)
      std::printf("[LazyDCA-refresh] commit : stat %.3e -> %.3e\n", stat_before, stat_after);
  };

  // Fabrique d'atomes : un run de Lanczos sur le modele courant Q+H_k, jusqu'a
  // `count` paires de Ritz negatives -> sparsification -> Rayleigh -> ortho.
  // Retourne le nombre d'atomes ajoutes ; applique le gate spectral.
  // v4c scope-fix-v2: visible from try_add_atoms and main loop
  idx_t reuse_streak = 0;
  double fact_mu = -1.0;  // mu of cached KKT factorization

  // ADAPTIVE_ORACLE_DELAYED_START_V4F
  bool atom_oracle_start_announced = false;

  auto try_add_atoms = [&](idx_t count) -> idx_t {
    // Delayed activation applies only to the adaptive oracle.
    // During initial seeding info.iters==0, so start_iter>0 gives H_0=0.
    if (st.atom_oracle_enable &&
        info.iters < st.atom_oracle_start_iter)
      return 0;

    if (st.atom_oracle_enable &&
        st.atom_oracle_start_iter > 0 &&
        !atom_oracle_start_announced) {
      atom_oracle_start_announced = true;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-atomOracle-delay] activation k=%lld (requested %lld)\n",
                    (long long)info.iters,
                    (long long)st.atom_oracle_start_iter);
    }

    if (!atoms_enabled || atom_growth_frozen || count <= 0 || atoms.size() >= max_atoms_eff) return 0;
    (void)0;
    double last_lam_after = std::numeric_limits<double>::quiet_NaN();
    count = std::min(count, max_atoms_eff - atoms.size());
    info.n_lanczos_calls++;
    const bool atom_choice_active = st.atom_selection_enable || st.atom_oracle_enable;
    const idx_t ritz_pool = st.atom_oracle_enable
        ? std::max<idx_t>(count, std::max<idx_t>(1, st.atom_oracle_pool_size))
        : (st.atom_selection_enable ? std::max<idx_t>(count, 5) : count);
    std::vector<std::vector<double>> ritz;
    auto rep = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit, st.lanczos_tol,
                                      42 + (unsigned)atoms.size(), &atoms, &ritz, ritz_pool);
    // ADAPTIVE_ORACLE_TRANSACTIONAL_PROBE_v3
    // Candidate evaluation must not mutate persistent solver state.
    // Legacy deliberately keeps the historical eager behaviour.
    if (!st.atom_oracle_enable) {
      last_lambda_est = rep.lmin;
      // zip 19e/19ak: rho_eff couvre la courbure mesuree du modele courant,
      // monotone apres insertion d'atomes (jamais re-hausse par une sonde ;
      // la chaine de certification par inertie reste le filet de securite).
      {
        double target = apply_rho_model_floor(
            std::fabs(std::min(0.0, rep.lmin)) * (1.0 + st.rho_rel_margin));
        // zip 19ak + bisection d'inertie : le niveau etabli par l'inertie est un
        // PLANCHER que la sonde ne peut pas percer ("acts as a floor: never
        // lowers rho") — sinon oscillation sonde/Gershgorin.
        // Les estimees de Krylov MAJORENT lambda_min (quotients de Rayleigh) :
        // chaque cible sous-estime donc le rho requis. Fenetre glissante :
        // rho ne descend qu'au MAX des 3 dernieres cibles (anti-ratchet).
        probe_hist[probe_pos % 3] = target; probe_pos++;
        double newrho = target;
        if (probe_pos >= 3) newrho = std::max({probe_hist[0], probe_hist[1], probe_hist[2]});
        if (atoms.size() == 0) {
          rho = apply_rho_model_floor(std::max(target, 0.0));  // seed : modele convexe initial
        } else if (!st.rho_monotone_after_atoms ||
                   (newrho < rho && probe_pos >= 3)) {
          // proposition : la decroissance ne sera COMMISE que si la
          // factorisation d'essai a (rho_candidat, rho_x=0) certifie l'inertie
          // (bisection certifiee du zip ; les sondes Krylov majorent lambda_min
          // donc sous-estiment systematiquement le rho requis).
          rho_pending = apply_rho_model_floor(std::max(newrho, 0.0));
          if (st.verbose >= 2)
            std::printf("[LazyDCA-rho] sonde lambda=%.3e : candidat rho modele %.3e (a certifier)\n",
                        rep.lmin, rho_pending);
        }
        rho_cap = std::max(rho + rho_x, std::max(target, 1e-6));
      }
    }
    idx_t added = 0;
    std::vector<std::vector<idx_t>> cand_ui; std::vector<std::vector<double>> cand_uv;
    std::vector<double> cand_beta, cand_energy, cand_ray, cand_activity;
    std::vector<double> cand_support_frac, cand_coupling;

    // DCA-relevant directions for the oracle.  Gradient uses the current DCA
    // anchor (or the current iterate before x_ref is initialised) and is
    // projected onto the current free face.  The last accepted DCA step is
    // already a face-respecting displacement.
    std::vector<double> oracle_grad(n, 0.0), oracle_step(n, 0.0);
    bool oracle_grad_valid = false, oracle_step_valid = false;
    if (st.atom_oracle_enable && st.atom_oracle_activity_mode != 0) {
      const std::vector<double>& xa = ((idx_t)x_ref.size() == n) ? x_ref : w.x;
      P.Q.gaxpy_sym(xa.data(), oracle_grad.data());
      for (idx_t i2 = 0; i2 < n; ++i2) {
        oracle_grad[i2] += P.q[i2];
        if ((w.hl[i2] && std::fabs(xa[i2] - P.l[i2]) <= st.atom_bound_active_tol) ||
            (w.hu[i2] && std::fabs(P.u[i2] - xa[i2]) <= st.atom_bound_active_tol))
          oracle_grad[i2] = 0.0;
      }
      double ng = std::sqrt(std::max(0.0, dot(oracle_grad, oracle_grad)));
      if (ng > 1e-14) { for (double& v : oracle_grad) v /= ng; oracle_grad_valid = true; }
      if (atom_oracle_last_step_valid) {
        oracle_step = atom_oracle_last_step;
        double ns = std::sqrt(std::max(0.0, dot(oracle_step, oracle_step)));
        if (ns > 1e-14) { for (double& v : oracle_step) v /= ns; oracle_step_valid = true; }
      }
    }
    // v4c: dimension effective de la face libre.
    idx_t oracle_nfree = n;
    if (st.atom_oracle_enable && st.atom_oracle_activity_mode != 0) {
      const std::vector<double>& xa_nf =
          ((idx_t)x_ref.size() == n) ? x_ref : w.x;

      oracle_nfree = 0;
      for (idx_t i2 = 0; i2 < n; ++i2) {
        const bool active_nf =
            (w.hl[i2] &&
             std::fabs(xa_nf[i2] - P.l[i2]) <= st.atom_bound_active_tol) ||
            (w.hu[i2] &&
             std::fabs(P.u[i2] - xa_nf[i2]) <= st.atom_bound_active_tol);

        if (!active_nf) oracle_nfree++;
      }
    }

    auto oracle_activity = [&](const std::vector<idx_t>& ui0, const std::vector<double>& uv0) {
      auto ad = [&](const std::vector<double>& d) {
        double z = 0.0; for (size_t kk = 0; kk < ui0.size(); ++kk) z += uv0[kk] * d[ui0[kk]];
        return std::fabs(z);
      };
      const double ag = oracle_grad_valid ? ad(oracle_grad) : 0.0;
      const double as = oracle_step_valid ? ad(oracle_step) : 0.0;
      if (st.atom_oracle_activity_mode == 1) return ag;
      if (st.atom_oracle_activity_mode == 2) return as;
      if (st.atom_oracle_activity_mode == 3) return std::max(ag, as);
      return 0.0;
    };
    for (idx_t c = 0; c < (idx_t)ritz.size() && added < count; ++c) {
      if (c < (idx_t)rep.ritz_low.size() && rep.ritz_low[c] >= -1e-10) break;  // plus de negatif
      // projection de face (zip legacy) : zeroter les coordonnees ou x_ref
      // est sur une borne active — l'atome vit sur la face libre.
      std::vector<double> zc = ritz[c];
      for (idx_t i2 = 0; i2 < n; ++i2) {
        if ((w.hl[i2] && std::fabs(x_ref[i2] - P.l[i2]) <= st.atom_bound_active_tol) ||
            (w.hu[i2] && std::fabs(P.u[i2] - x_ref[i2]) <= st.atom_bound_active_tol))
          zc[i2] = 0.0;
      }
      { double nz = 0.0; for (double v2 : zc) nz += v2 * v2;
        if (!(nz > st.atom_min_norm * st.atom_min_norm)) continue; }
      // Energy retained by Top-k BEFORE renormalisation. This distinguishes a
      // numerically dense Ritz vector from a genuinely delocalised mode.
      double zc_n2 = 0.0;
      for (double v2 : zc) zc_n2 += v2 * v2;
      std::vector<idx_t> ui; std::vector<double> uv;
      double energy_kept = 0.0;
      const bool sparse_ok = st.atom_energy_sparsify
          ? AtomSet::sparsify_energy(zc, st.atom_max_nnz, st.atom_energy_target,
                                     st.atom_min_nnz, ui, uv, &energy_kept)
          : AtomSet::sparsify(zc, st.atom_max_nnz, ui, uv);
      if (!sparse_ok) continue;
      if (!st.atom_energy_sparsify) {
        double kept_n2 = 0.0;
        for (idx_t ii : ui) kept_n2 += zc[ii] * zc[ii];
        energy_kept = (zc_n2 > 0.0) ? kept_n2 / zc_n2 : 0.0;
      }

      std::vector<double> ud(n, 0.0), t2(n, 0.0);
      for (size_t k2 = 0; k2 < ui.size(); ++k2) ud[ui[k2]] = uv[k2];
      P.Q.gaxpy_sym(ud.data(), t2.data());
      atoms.apply(ud.data(), t2.data());
      double ray = 0.0;
      for (idx_t i2 = 0; i2 < n; ++i2) ray += ud[i2] * t2[i2];
      bool ortho_ok = true;
      for (idx_t j2 = 0; j2 < atoms.size() && ortho_ok; ++j2)
        ortho_ok = atoms.abs_cos(j2, ud.data()) <= st.atom_ortho_tol;
      // zip legacy : PAS de test Rayleigh post-sparsification ni de filtre
      // d'orthogonalite (ingredients de la SELECTION, optionnelle) ;
      // beta = |lambda_pre| + margin (additif).
      if (atom_choice_active && (ray >= -st.atom_neg_tol || !ortho_ok)) continue;
      const double lam_pre = (c < (idx_t)rep.ritz_low.size()) ? rep.ritz_low[c] : rep.lmin;
      const double beta_legacy = -lam_pre + st.atom_margin;
      double beta_c = beta_legacy;
      if (atom_choice_active) beta_c = -ray + st.atom_margin;   // selection/oracle: beta from sparse Rayleigh
      // ADAPTIVE_ORACLE_BETA_SCALE_V4E
      // Oracle chooses both direction and strength. Residual negative
      // curvature remains covered by scalar rho.
      if (st.atom_oracle_enable)
        beta_c *= clampd(st.atom_oracle_beta_scale, 0.0, 1.0);

      // A/B SPARSITY RECERTIFICATION. Build the UNSPARSIFIED legacy atom from
      // this same face-projected Ritz direction, measure its post-update
      // lambda_min, then tune beta of the Top-k atom until it matches that
      // spectral lift. The support is the intended experimental variable.
      if (st.atom_sparse_recertify_beta && !atom_choice_active && beta_legacy > 0.0) {
        std::vector<idx_t> dui; std::vector<double> duv;
        if (AtomSet::sparsify(zc, n, dui, duv)) {
          const unsigned recert_seed = 7001u + (unsigned)atoms.size();
          auto eval_lmin = [&](const std::vector<idx_t>& ai,
                               const std::vector<double>& av,
                               double b) -> double {
            AtomSet tmp = atoms;
            tmp.idx.push_back(ai); tmp.val.push_back(av); tmp.beta.push_back(b);
            auto rr = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit,
                                             st.lanczos_tol, recert_seed, &tmp);
            info.n_lanczos_calls++;
            return (rr.converged && std::isfinite(rr.lmin))
                       ? rr.lmin : -std::numeric_limits<double>::infinity();
          };

          const double lam_dense = eval_lmin(dui, duv, beta_legacy);
          const double tol_match = st.atom_sparse_recertify_tol_rel *
              std::max(1.0, std::fabs(lam_pre));
          bool attainable = std::isfinite(lam_dense);
          double b_lo = 0.0, b_hi = std::max(beta_legacy, 1e-14);
          // Full support is the control: preserve the legacy beta exactly.
          // This makes k=n a genuine reproduction of the dense freeze run.
          const bool full_support = (ui.size() == dui.size());
          double lam_hi = full_support ? lam_dense
              : (attainable ? eval_lmin(ui, uv, b_hi)
                            : -std::numeric_limits<double>::infinity());
          idx_t neval = full_support ? 1 : 2;

          while (!full_support && attainable && lam_hi < lam_dense - tol_match &&
                 b_hi < st.atom_sparse_beta_max) {
            b_lo = b_hi;
            b_hi = std::min(st.atom_sparse_beta_max, 4.0 * b_hi);
            lam_hi = eval_lmin(ui, uv, b_hi);
            neval++;
            if (!std::isfinite(lam_hi)) { attainable = false; break; }
            if (b_hi >= st.atom_sparse_beta_max) break;
          }
          if (attainable && lam_hi < lam_dense - tol_match) attainable = false;

          double lam_sparse = lam_hi;
          if (attainable && !full_support) {
            for (idx_t bi = 0; bi < st.atom_sparse_recertify_maxit; ++bi) {
              const double b_mid = 0.5 * (b_lo + b_hi);
              const double lam_mid = eval_lmin(ui, uv, b_mid);
              neval++;
              if (!std::isfinite(lam_mid)) { attainable = false; break; }
              if (std::fabs(lam_mid - lam_dense) <= tol_match) {
                b_lo = b_hi = b_mid; lam_sparse = lam_mid; break;
              }
              if (lam_mid < lam_dense) b_lo = b_mid;
              else { b_hi = b_mid; lam_sparse = lam_mid; }
            }
            beta_c = b_hi;
            if (std::isfinite(lam_sparse) && std::fabs(lam_sparse - lam_dense) > tol_match) {
              lam_sparse = eval_lmin(ui, uv, beta_c); neval++;
            }
          } else {
            // Diagnostic only: a MATCH=NO run is not a matched-spectral A/B.
            beta_c = std::min(b_hi, st.atom_sparse_beta_max);
          }

          if (st.verbose >= 1)
            std::printf("[LazyDCA-sparseAB] atom=%lld nnz=%zu/%lld energy=%.9f "
                        "beta_dense=%.6e lambda_dense=%.6e beta_sparse=%.6e "
                        "lambda_sparse=%.6e match=%s evals=%lld\n",
                        (long long)(atoms.size() + 1), ui.size(), (long long)n,
                        energy_kept, beta_legacy, lam_dense, beta_c, lam_sparse,
                        attainable ? "YES" : "NO", (long long)neval);
        }
      }
      if (beta_c <= 0.0) continue;                          // zip: beta <= 0 -> stop
      if (st.atom_oracle_enable ||
          (st.atom_selection_enable && st.atom_selection_criterion >= 1)) {
        // Collect a candidate pool.  The adaptive oracle will evaluate each
        // candidate against the scalar no-atom alternative before committing.
        cand_ui.push_back(ui); cand_uv.push_back(uv); cand_beta.push_back(beta_c);
        cand_energy.push_back(energy_kept); cand_ray.push_back(ray);
        cand_activity.push_back(oracle_activity(ui, uv));
        cand_support_frac.push_back((double)ui.size() / std::max<double>(1.0, (double)n));
        double l1 = 0.0; for (double vv : uv) l1 += std::fabs(vv);
        const double kden = std::max<double>(1.0, (double)ui.size() - 1.0);
        cand_coupling.push_back(std::max(0.0, l1 * l1 - 1.0) / kden);
        info.n_atom_candidates++;
        continue;
      }
      atoms.idx.push_back(std::move(ui));
      atoms.val.push_back(std::move(uv));
      atoms.beta.push_back(beta_c);
      added++;
      if (st.verbose >= 2)
        std::printf("[LazyDCA-atom] +atom #%lld nnz=%zu rayleigh=%.3e beta=%.3e\n",
                    (long long)atoms.size(), atoms.idx.back().size(), ray, atoms.beta.back());
    }
    while (!cand_ui.empty() && added < count) {
      idx_t best = -1; double best_lam = -std::numeric_limits<double>::infinity();
      double best_key = -std::numeric_limits<double>::infinity();
      double best_activity = 0.0, best_lift_rel = 0.0, best_score = 0.0;
      double best_activity_scaled = 0.0;
      double best_directional_ratio = 0.0;
      double best_directional_ratio_after = 0.0;
      double best_rho_scalar = 0.0, best_rho_spectral = 0.0;
      double best_rho_residual = 0.0, best_rho_hybrid = 0.0;
      // ATOM_CLUSTER_PROD_V1: pool-local classifications are pure.
      std::vector<AtomCluster::Result> cand_cluster_result;
      if (st.atom_cluster_enable && st.atom_oracle_enable)
        cand_cluster_result.resize(cand_ui.size());
      for (idx_t c2 = 0; c2 < (idx_t)cand_ui.size(); ++c2) {
        atoms.idx.push_back(cand_ui[c2]); atoms.val.push_back(cand_uv[c2]);
        atoms.beta.push_back(cand_beta[c2]);
        auto rep_after = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit,
                                                st.lanczos_tol, 42, &atoms);
        atoms.idx.pop_back(); atoms.val.pop_back(); atoms.beta.pop_back();
        info.n_lanczos_calls++;
        const double lam_after = (rep_after.converged && std::isfinite(rep_after.lmin))
                           ? rep_after.lmin : -std::numeric_limits<double>::infinity();
        const double lam_base = std::isfinite(last_lam_after) ? last_lam_after : rep.lmin;
        double key = lam_after;
        bool admissible = std::isfinite(lam_after);
        double activity = (c2 < (idx_t)cand_activity.size()) ? cand_activity[c2] : 0.0;
        double lift_rel = 0.0, directional_ratio = 1.0, score = key;
        double activity_scaled = activity;
        double directional_ratio_after = 1.0;
        double rho_scalar_eval = 0.0, rho_spectral_eval = 0.0;
        double rho_residual_eval = 0.0, rho_hybrid_eval = 0.0;

        if (st.atom_oracle_enable && admissible) {
          const double denom_lam =
              std::max(st.atom_neg_tol,
                       std::fabs(std::min(lam_base, -st.atom_neg_tol)));

          const double lift = std::max(0.0, lam_after - lam_base);
          lift_rel = lift / denom_lam;

          rho_scalar_eval =
              std::fabs(std::min(0.0, lam_base)) *
              (1.0 + st.rho_rel_margin);

          rho_spectral_eval =
              std::fabs(std::min(0.0, lam_after)) *
              (1.0 + st.rho_rel_margin);

          // Diagnostic historique avant correction hybride.
          directional_ratio = rho_scalar_eval > 1e-14
              ? (rho_spectral_eval +
                 cand_beta[c2] * activity * activity) / rho_scalar_eval
              : 1.0;

          activity_scaled =
              activity *
              std::sqrt((double)std::max<idx_t>(1, oracle_nfree));

          const double activity_gate =
              st.atom_oracle_activity_sqrt_nfree
                  ? activity_scaled
                  : activity;

          const double min_activity_gate =
              st.atom_oracle_activity_sqrt_nfree
                  ? st.atom_oracle_min_scaled_activity
                  : st.atom_oracle_min_activity;

          // En politique residual, theta n'est plus un hard gate.
          // Il définit la quantité isotrope minimale à conserver.
          rho_residual_eval = 0.0;

          if (st.atom_oracle_directional_residual &&
              st.atom_oracle_scalar_fallback &&
              st.atom_oracle_min_directional_ratio > 0.0)
            rho_residual_eval =
                st.atom_oracle_min_directional_ratio * rho_scalar_eval;

          rho_hybrid_eval =
              std::max(rho_spectral_eval, rho_residual_eval);

          directional_ratio_after = rho_scalar_eval > 1e-14
              ? (rho_hybrid_eval +
                 cand_beta[c2] * activity * activity) / rho_scalar_eval
              : 1.0;

          const double sf =
              (c2 < (idx_t)cand_support_frac.size())
                  ? cand_support_frac[c2]
                  : 1.0;

          const double density = sf * sf;

          const double coupling =
              (c2 < (idx_t)cand_coupling.size())
                  ? cand_coupling[c2]
                  : 1.0;

          const double beta_rel = cand_beta[c2] / denom_lam;

          // Dimensionless activity participates in ranking but is capped.
          const double activity_rank =
              std::min(1.0, std::max(0.0, activity_gate));

          score =
              lift_rel *
                  (1.0 + st.atom_oracle_activity_weight * activity_rank)
              + st.atom_oracle_directional_weight *
                  std::min(1.0, std::max(0.0, directional_ratio))
              - st.atom_oracle_density_weight * density
              - st.atom_oracle_coupling_weight * coupling
              - st.atom_oracle_beta_weight * beta_rel;

          admissible =
              lift_rel >= st.atom_oracle_min_lift_rel &&
              (st.atom_oracle_activity_mode == 0 ||
               activity_gate >= min_activity_gate) &&
              score >= st.atom_oracle_min_score;

          // Reproduction exacte de l'ancien hard gate pour A/B.
          if (!st.atom_oracle_directional_residual &&
              st.atom_oracle_scalar_fallback &&
              st.atom_oracle_min_directional_ratio > 0.0 &&
              directional_ratio < st.atom_oracle_min_directional_ratio)
            admissible = false;

          if (st.verbose >= 2)
            std::printf(
                "[LazyDCA-atomOracle-v4] cand=%lld "
                "actRaw=%.3e actScaled=%.3e nfree=%lld "
                "dirBefore=%.3e dirAfter=%.3e "
                "rhoScalar=%.3e rhoSpectral=%.3e "
                "rhoResidual=%.3e rhoHybrid=%.3e\n",
                (long long)c2,
                activity, activity_scaled,
                (long long)oracle_nfree,
                directional_ratio, directional_ratio_after,
                rho_scalar_eval, rho_spectral_eval,
                rho_residual_eval, rho_hybrid_eval);

          if (!admissible) {
            info.n_atom_oracle_rejects++;
            if (st.verbose >= 2)
              std::printf("[LazyDCA-atomOracle] reject cand=%lld lift=%.3e act=%.3e "
                          "dirRatio=%.3e score=%.3e nnz=%zu energy=%.6f\n",
                          (long long)c2, lift_rel, activity, directional_ratio, score,
                          cand_ui[c2].size(), c2 < (idx_t)cand_energy.size() ? cand_energy[c2] : 0.0);
            continue;
          }

          // ATOM_CLUSTER_PROD_V1: bounded SOFT structural correction.
          // This is deliberately AFTER the native oracle admissibility gates:
          // clustering changes ranking only, never curvature eligibility.
          if (st.atom_cluster_enable && st.atom_oracle_enable) {
            const double cluster_curvature =
                c2 < (idx_t)cand_ray.size() ? cand_ray[c2] : lam_after;
            auto cr = atom_clusters.classify(
                cand_ui[c2], cand_uv[c2], cluster_curvature);
            cand_cluster_result[(std::size_t)c2] = cr;
            info.n_atom_cluster_classify++;
            info.atom_cluster_last_similarity = cr.valid ? cr.sim.combined : 0.0;
            info.atom_cluster_tau_in_final = cr.tau_in_effective;
            info.atom_cluster_tau_out_final = cr.tau_out_effective;
            score += AtomCluster::score_adjustment(
                cr, st.atom_cluster_duplicate_penalty,
                st.atom_cluster_novelty_bonus);
          }
          key = score;
        } else if (st.atom_selection_criterion == 2 && admissible) {
          key = (lam_after - lam_base) / std::max(cand_beta[c2], 1e-300);
        }
        if (admissible && key > best_key) {
          best_key = key; best_lam = lam_after; best = c2;
          best_activity = activity;
          best_activity_scaled = activity_scaled;
          best_lift_rel = lift_rel;
          best_score = score;
          best_directional_ratio = directional_ratio;
          best_directional_ratio_after = directional_ratio_after;
          best_rho_scalar = rho_scalar_eval;
          best_rho_spectral = rho_spectral_eval;
          best_rho_residual = rho_residual_eval;
          best_rho_hybrid = rho_hybrid_eval;
        }
      }
      if (best < 0) {
        if (st.atom_oracle_enable && st.atom_oracle_scalar_fallback) {
          info.n_atom_scalar_fallbacks++;
          if (st.verbose >= 1)
            std::printf("[LazyDCA-atomOracle] NO-ATOM choisi : aucun candidat ne bat le modele scalaire (rho=%.3e)\n", rho);
        }
        break;
      }
      // ATOM_CLUSTER_PROD_V1: commit structural state exactly once.
      if (st.atom_cluster_enable && st.atom_oracle_enable &&
          best >= 0 && best < (idx_t)cand_cluster_result.size()) {
        const double cluster_curvature =
            best < (idx_t)cand_ray.size() ? cand_ray[best] : best_lam;
        auto cc = atom_clusters.commit(
            cand_cluster_result[(std::size_t)best],
            cand_ui[best], cand_uv[best], cluster_curvature, best_score);
        if (seeding_phase) atom_cluster_seed_stop.observe_commit(cc.decision);
        if (cc.decision == AtomClusterDecision::Novel) info.n_atom_cluster_novel++;
        else if (cc.decision == AtomClusterDecision::Existing) info.n_atom_cluster_existing++;
        else if (cc.decision == AtomClusterDecision::Ambiguous) info.n_atom_cluster_ambiguous++;
        if (cc.representative_replaced) info.n_atom_cluster_rep_replacements++;
        if (cc.registry_saturated) info.n_atom_cluster_saturated++;
        info.atom_cluster_count_final = (idx_t)atom_clusters.size();
        info.atom_cluster_tau_in_final = cc.tau_in_effective;
        info.atom_cluster_tau_out_final = cc.tau_out_effective;
        info.atom_cluster_last_similarity = cc.valid ? cc.sim.combined : 0.0;
      }
      last_lam_after = best_lam;
      atoms.idx.push_back(std::move(cand_ui[best]));
      atoms.val.push_back(std::move(cand_uv[best]));
      atoms.beta.push_back(cand_beta[best]);
      // v4c: COMMIT mathématique, jamais exécuté pour un candidat rejeté.
      if (st.atom_oracle_enable &&
          st.atom_oracle_directional_residual) {
        // Floor of the CURRENT committed atomic model.
        rho_oracle_residual_floor = best_rho_residual;

        // During seeding, apply_rho_model_floor() below injects this floor
        // into rho_pending before inertia certification.
        // During later growth it must be enforced immediately if necessary.
        if (!seeding_phase && !rho_model_fixed &&
            rho < rho_oracle_residual_floor) {
          rho = rho_oracle_residual_floor;
          rho_x = 0.0;
          rho_pending = -1.0;
          fact_mu = -1.0;
          reuse_streak = 0;
          info.n_rho_model_updates++;
          info.n_rho_model_increases++;
        }
      }

      // v4d: outside seeding, rho itself carries the safe commit level.
      if (st.atom_oracle_enable &&
          st.atom_oracle_directional_residual &&
          !seeding_phase)
        rho_oracle_residual_floor = 0.0;

      added++;
      // ADAPTIVE_MARGINAL_RETURN_V21: score every oracle COMMIT,
      // including seed commits whose subsequent Lanczos recertification is noisy.
      if (st.adaptive_marginal_return_enable && st.atom_oracle_enable) {
        const double amr21_lift = std::max(0.0, best_lift_rel);
        const double amr21_activity =
            std::max(0.0, std::min(1.0, best_activity_scaled));
        const double amr21_effective = amr21_lift * amr21_activity;
        const AdaptiveReturnSource amr21_source = seeding_phase
            ? AdaptiveReturnSource::Seed
            : (st.atom_event_enable ? AdaptiveReturnSource::EventOracle
                                    : AdaptiveReturnSource::Update);
        const auto amr21_dec = adaptive_return.observe_scored(
            amr21_lift, amr21_effective, (std::size_t)atoms.size(), amr21_source);
        sync_adaptive_return_info();
        if (amr21_dec.action == AdaptiveReturnAction::StopSeed)
          amr21_seed_stop_requested = true;
        if (st.verbose >= 2)
          std::printf("[AMRS-v2.3] COMMIT src=%d atom=%lld lift=%.3e eff=%.3e "
                      "thr=(%.3e,%.3e) med=(%.3e,%.3e) action=%d cooldown=%.1fx\n",
                      (int)amr21_source, (long long)atoms.size(), amr21_lift,
                      amr21_effective, amr21_dec.lift_threshold,
                      amr21_dec.effective_threshold, amr21_dec.recent_lift_median,
                      amr21_dec.recent_effective_median, (int)amr21_dec.action,
                      amr21_dec.cooldown_multiplier);
        else if (st.verbose >= 1 &&
                 (amr21_dec.action == AdaptiveReturnAction::Cooldown ||
                  amr21_dec.action == AdaptiveReturnAction::Reactivate))
          std::printf("[AMRS-v2.3] event spacing action=%d cooldown=%.1fx current=%lld\n",
                      (int)amr21_dec.action, amr21_dec.cooldown_multiplier,
                      (long long)info.adaptive_return_event_spacing_current);
      }

      if (st.atom_oracle_enable && st.verbose >= 1)
        std::printf(
            "[LazyDCA-atomOracle-v4] COMMIT "
            "actRaw=%.3e actScaled=%.3e "
            "dirBefore=%.3e dirAfter=%.3e "
            "rhoScalar=%.3e rhoSpectral=%.3e "
            "rhoResidual=%.3e rhoHybrid=%.3e\n",
            best_activity, best_activity_scaled,
            best_directional_ratio,
            best_directional_ratio_after,
            best_rho_scalar, best_rho_spectral,
            best_rho_residual, best_rho_hybrid);

      if (st.atom_oracle_enable) {
        info.atom_last_score = best_score;
        info.atom_last_activity = best_activity;
        info.atom_last_lift_rel = best_lift_rel;
        info.atom_last_energy = best < (idx_t)cand_energy.size() ? cand_energy[best] : 0.0;
        info.atom_last_support_frac = best < (idx_t)cand_support_frac.size() ? cand_support_frac[best] : 0.0;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-atomOracle] +atom #%lld score=%.3e lift=%.3e act=%.3e "
                      "dirRatio=%.3e nnz=%zu energy=%.9f lam_after=%.3e beta=%.3e\n",
                      (long long)atoms.size(), best_score, best_lift_rel, best_activity,
                      best_directional_ratio, atoms.idx.back().size(), info.atom_last_energy,
                      best_lam, cand_beta[best]);
      } else if (st.verbose >= 2) {
        std::printf("[LazyDCA-atom] +atom #%lld (selection gloutonne, lam_apres=%.3e beta=%.3e)\n",
                    (long long)atoms.size(), best_lam, cand_beta[best]);
      }
      cand_ui.erase(cand_ui.begin() + best); cand_uv.erase(cand_uv.begin() + best);
      cand_beta.erase(cand_beta.begin() + best);
      if (best < (idx_t)cand_energy.size()) cand_energy.erase(cand_energy.begin() + best);
      if (best < (idx_t)cand_ray.size()) cand_ray.erase(cand_ray.begin() + best);
      if (best < (idx_t)cand_activity.size()) cand_activity.erase(cand_activity.begin() + best);
      if (best < (idx_t)cand_support_frac.size()) cand_support_frac.erase(cand_support_frac.begin() + best);
      if (best < (idx_t)cand_coupling.size()) cand_coupling.erase(cand_coupling.begin() + best);
    }
    if (added) {
      // Commit oracle curvature only after an atom is accepted.
      if (st.atom_oracle_enable && std::isfinite(last_lam_after))
        last_lambda_est = last_lam_after;
      atoms_dirty = true;
      info.n_atoms = atoms.size(); info.atoms_nnz = atoms.nnz();
      rho_cap = -1.0; hint_valid = false;   // le spectre du modele a change
      // Re-semis de rho au SEEDING : le seed etait pris sur lambda_min(Q)
      // AVANT insertion des atomes initiaux — 1, 3 ou 6 atomes donnaient le
      // meme rho de depart. On re-mesure lambda_min(Q+H_seed) (le glouton
      // crit-1 l'a deja : lam_apres de la derniere insertion) et on re-seme.
      if (seeding_phase && added > 0) {
        double lam_seed = last_lam_after;
        if (!std::isfinite(lam_seed)) {
          auto rep_s = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit,
                                              st.lanczos_tol, 43, &atoms);
          info.n_lanczos_calls++;
          if (rep_s.converged && std::isfinite(rep_s.lmin)) lam_seed = rep_s.lmin;
        }
        if (std::isfinite(lam_seed)) {
          // Les estimees de Krylov MAJORENT lambda_min : un re-semis commis
          // directement sous-regularise le modele (cascade de rounds trop
          // optimistes -> instabilite duale observee en crit-1). Discipline
          // de la lignee : la decroissance passe par rho_pending et n'est
          // COMMISE que si la factorisation d'essai certifie l'inertie.
          double target2 = apply_rho_model_floor(
              std::fabs(std::min(0.0, lam_seed)) * (1.0 + st.rho_rel_margin));

          if (st.atom_oracle_enable &&
              st.atom_oracle_directional_residual &&
              !rho_model_fixed)
            target2 = std::max(target2, rho_oracle_residual_floor);
          probe_hist[probe_pos % 3] = target2; probe_pos++;
          if (target2 < rho) {
            const double target2_floor = apply_rho_model_floor(std::max(target2, 0.0));
            rho_pending = (rho_pending >= 0.0) ? std::min(rho_pending, target2_floor)
                                               : target2_floor;
            rho_bisect_left = 5;   // bisection certifiee armee pour ce seed
            if (st.verbose >= 1)
              std::printf("[LazyDCA-rho] re-semis post-seeding PROPOSE : lambda(Q+H)=%.3e, candidat %.3e (a certifier, %lld atomes)\n",
                          lam_seed, std::max(target2, 0.0), (long long)atoms.size());
          }
        }
      }
      if (!st.atom_oracle_enable) {
        if (rep.lmin < 0.0) info.lambda_min_reduced = rep.lmin;
      } else if (added > 0 && std::isfinite(last_lam_after)) {
        info.lambda_min_reduced = last_lam_after;
      }
    }
    return added;
  };
  auto compute_cap = [&]() {
    double lmin_hat;
    if (std::isfinite(st.lambda_min_hint) && hint_valid) {
      lmin_hat = st.lambda_min_hint * s_f;
      if (st.verbose >= 2) std::printf("[LazyDCA-lanczos] using external lambda_min hint = %.6e\n", lmin_hat);
    } else if (feas_dominated || !proj || !proj->ok()) {
      lmin_hat = P.Q.gershgorin_lmin();
      if (st.verbose >= 2) std::printf("[LazyDCA-lanczos] feasibility-dominated/no-proj => Gershgorin lmin >= %.3e\n", lmin_hat);
    } else {
      info.n_lanczos_calls++;
      auto rep = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit, st.lanczos_tol,
                                        42, atoms.size() ? &atoms : nullptr);
      lmin_hat = rep.lmin;
      if (st.verbose >= 2) std::printf("[LazyDCA-lanczos] reduced lambda_min_hat = %.6e (it=%lld, conv=%d)\n",
                                  lmin_hat, (long long)rep.iters, (int)rep.converged);
    }
    info.lambda_min_reduced = lmin_hat;
    double cap = std::fabs(std::min(0.0, lmin_hat)) * (1.0 + st.rho_rel_margin);
    return std::max(cap, 1e-6);
  };

  // Full-space scalar target used by the article's DYN ablation.  H_k=0, so
  // the curvature check is performed on Q itself.  The raw Ritz value is
  // guarded by its residual before it is used to COMMIT a smaller model rho:
  //   lambda_guard = lambda_hat - safety * residual.
  // This is a numerical curvature test (not a theorem-level eigenvalue bound),
  // but it is materially stronger than accepting a decrease merely because
  // the barrier KKT factorization happened to succeed.  A non-finite probe
  // falls back to the guaranteed Gershgorin lower bound.
  auto dyn_scalar_target = [&](unsigned seed, double* lhat_out = nullptr,
                               double* guard_out = nullptr) {
    info.n_lanczos_calls++;
    info.n_rho_curvature_checks++;
    auto rep = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit,
                                      st.lanczos_tol, seed, nullptr);
    const double lhat = std::isfinite(rep.lmin) ? rep.lmin : gersh_lb;
    const double lguard = guarded_lanczos_lmin(
        rep.lmin, rep.residual, st.rho_lanczos_residual_safety, gersh_lb);
    last_lambda_est = lhat;
    info.lambda_min_reduced = lhat;
    info.rho_last_curvature_guard = lguard;
    if (lhat_out) *lhat_out = lhat;
    if (guard_out) *guard_out = lguard;
    const double target = spectral_shift_from_lmin(lguard, st.rho_dyn_floor_rel_margin,
                                                   st.rho_init_abs_margin);
    if (st.verbose >= 2)
      std::printf("[LazyDCA-rho DYN] probe lambda_hat=%.6e residual=%.3e guard=%.6e "
                  "(it=%lld conv=%d) -> target=%.6e\n",
                  lhat, rep.residual, lguard, (long long)rep.iters,
                  (int)rep.converged, target);
    return std::clamp(target, 0.0, st.rho_abs_max);
  };

  KKT kkt(P, st);
  // KKT factorization reuse state
  double fact_rho = -1.0;      // rho frozen in the current factorization
  double prev_Fmu = INF;
  idx_t mu_inner = 0;          // iterations internes depuis la derniere baisse de mu
  idx_t armijo_fail_streak = 0;
  bool last_clean_full = false;// last iter: clean factorization + near-full step
  double sigma = st.sigma0;
  Residuals R{};
  bool converged = false;
  // IPOPT-style iteration log state (previous step quantities)
  double prev_apr = -1.0, prev_adu = -1.0, prev_dnrm = -1.0;
  idx_t prev_ls = 0;
  // colonnes du tableau d'iterations (valeurs du pas precedent)
  double last_apr = 0.0, last_adu = 0.0, last_dnorm = 0.0;
  idx_t last_ls = 0, banner_rows = 0;

  // clip defensif : quel que soit le chemin d'initialisation, la barriere
  // exige l'interieur strict de la boite.
  for (idx_t i = 0; i < n; ++i) {
    if (w.hl[i] && w.hu[i]) { double k = 1e-6 * std::max(1e-8, P.u[i] - P.l[i]); w.x[i] = clampd(w.x[i], P.l[i] + k, P.u[i] - k); }
    else if (w.hl[i]) w.x[i] = std::max(w.x[i], P.l[i] + 1e-6);
    else if (w.hu[i]) w.x[i] = std::min(w.x[i], P.u[i] - 1e-6);
  }
  // reference DCA initiale + premier ancrage (exempte de descente, jamais de
  // faisabilite — zip 19g)
  x_ref = w.x;
  {
    Residuals R0 = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
    double cv0 = std::max(R0.eq_abs, R0.ineq_abs);
    bool feas0 = cv0 <= st.sd_cv_abs ||
                 (st.sd_cv_rel > 0 && R0.feas_rel <= st.sd_cv_rel && cv0 <= st.sd_rel_feas_abs_cap);
    if (feas0) { ref_obj = objective(P_in, w); best_feas_obj = ref_obj; accepts = 1;
      dca_accept_obj_hist.push_back(ref_obj);
      if (st.verbose >= 2) std::printf("[LazyDCA-ACCEPT] ancrage initial faisable, ref=%.10e\n", ref_obj); }
  }
  snapshot_momentum_metric();
  // atomes initiaux (Q constante en QP : directions negatives globales)
  // Seeding en ROUNDS successifs de un atome : un run de Lanczos ne livre
  // qu'UN vecteur par espace propre degenere/groupe (limite Krylov) ; chaque
  // round re-sonde Q+H mis a jour (graine differente), l'espace restant
  // re-emerge, et le re-semis de rho (certifie) s'applique en cascade.
  //
  // Mode COUVERTURE (--seed-coverage x) : au lieu d'un k aveugle, on seme
  // jusqu'a couvrir x% de la courbure negative : couverture = 1 - lam_k/lam_0
  // (lam_0 = lambda_min(Q)). Garde de stagnation : sur un spectre en
  // continuum, si un round n'ameliore pas lambda de 1% de |lam_0|, les
  // atomes n'achetent plus rien — on s'arrete et on le dit.
  if (st.atoms_seed_coverage > 0.0 && atoms_enabled) {
    seeding_phase = true;
    double lam0 = std::numeric_limits<double>::quiet_NaN();
    if (hint_valid && std::isfinite(st.lambda_min_hint)) lam0 = st.lambda_min_hint * s_f;
    else {
      auto rep0 = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit, st.lanczos_tol, 41, &atoms);
      info.n_lanczos_calls++;
      if (rep0.converged && std::isfinite(rep0.lmin)) lam0 = rep0.lmin;
    }
    if (std::isfinite(lam0) && lam0 < 0.0) {
      // lam_best = meilleure (moins negative) valeur JAMAIS observee. H ne
      // fait que croitre (chaque atome ajoute beta*aa', beta>0, PSD) donc
      // lambda_min(Q+H) DOIT etre non-decroissant round apres round. Un
      // recul observe (lam_k < lam_best) est un signal NON-AMBIGU de Lanczos
      // non converge (spectre serre/degenere, cas QPNSTAIR) — la couverture,
      // le critere de stagnation et la proposition de rho utilisent lam_best,
      // jamais la lecture brute potentiellement bruitee.
      double lam_prev = lam0, lam_best = lam0;
      idx_t stall_rounds = 0, regress_streak = 0;
      idx_t maxit_local = st.lanczos_maxit;
      while (atoms.size() < max_seed_atoms_eff) {
        if (try_add_atoms(1) == 0) break;
        auto repk = projected_lanczos_lmin(P.Q, nullptr, maxit_local, st.lanczos_tol,
                                           44 + (unsigned)atoms.size(), &atoms);
        info.n_lanczos_calls++;
        if (!(repk.converged && std::isfinite(repk.lmin))) break;
        const double lam_k = repk.lmin;
        const bool regressed = lam_k < lam_best - st.atoms_seed_stall_tol * std::fabs(lam0);
        if (regressed) {

          // ADAPTIVE_MARGINAL_RETURN_V23: noisy round is elapsed committed rank.
          // Keep lam_best as the monotone certified coverage state, but retain
          // this commit in the plateau horizon instead of deleting evidence.
          if (st.adaptive_marginal_return_enable && st.atom_oracle_enable) {
            const double amr23_cover_best = std::max(
                0.0, std::min(1.0, 1.0 - lam_best / lam0));
            const auto amr23_dec = adaptive_return.observe_seed_progress(
                amr23_cover_best, std::max(0.0, info.atom_last_lift_rel),
                (std::size_t)atoms.size(), false);
            sync_adaptive_return_info();
            if (st.verbose >= 1)
              std::printf("[AMRS-v2.3] SEED noisy atom=%lld cover=%.6e dcov=%.3e lift=%.3e "
                          "plateau=%d(%zu/%lld) Hgain=%.3e Hrate=%.3e Hlift=%.3e action=%d\n",
                          (long long)atoms.size(), amr23_dec.seed_coverage,
                          amr23_dec.seed_coverage_gain, amr23_dec.seed_lift_rel,
                          amr23_dec.seed_low_plateau ? 1 : 0,
                          amr23_dec.seed_plateau_confirmations,
                          (long long)st.adaptive_return_seed_plateau_confirmations,
                          amr23_dec.seed_plateau_horizon_gain,
                          amr23_dec.seed_plateau_gain_per_commit,
                          amr23_dec.seed_plateau_lift_median,
                          (int)amr23_dec.action);
            // The native block ends with `continue`, so a v2.3 stop detected on
            // a noisy round must be committed here, before that continue.
            if (amr23_dec.action == AdaptiveReturnAction::StopSeed) {
              info.n_adaptive_return_seed_stops++;
              info.adaptive_return_seed_stopped = true;
              info.adaptive_return_seed_stop_atom = (idx_t)atoms.size();
              if (st.verbose >= 1)
                std::printf("[AMRS-v2.3] STOP seed(noisy): atom=%lld reason=%lld "
                            "cover=%.6e Hrate=%.3e Hlift=%.3e\n",
                            (long long)atoms.size(),
                            (long long)info.adaptive_return_seed_stop_reason,
                            info.adaptive_return_seed_stop_coverage,
                            info.adaptive_return_seed_stop_plateau_gain_per_commit,
                            info.adaptive_return_seed_stop_plateau_lift_median);
              break;
            }
          }
          // Recul : la lecture est jugee non fiable (Lanczos pas converge sur
          // ce round). On double l'effort de Krylov pour les rounds suivants
          // (plafonne a 8x), et on NE MET PAS A JOUR lam_best sur ce round.
          maxit_local = std::min(maxit_local * 2, 8 * st.lanczos_maxit);
          if (st.verbose >= 1)
            std::printf("[LazyDCA-seed] atome %lld : lambda %.3e -> %.3e (RECUL, lanczos non fiable, "
                        "maxit -> %lld) — ignore, lam_best conserve = %.3e\n",
                        (long long)atoms.size(), lam_prev, lam_k, (long long)maxit_local, lam_best);
          lam_prev = lam_k;
          // NE PAS abandonner le seeding sur des reculs : sur un spectre en
          // CONTINUUM (ex. NCVXQP7), quelques rounds bruites parmi beaucoup
          // de bons ne signent pas une degenerescence globale — arreter ici
          // prive le solveur de couverture spectrale legitime, reportant le
          // travail sur le mecanisme reactif (evenements, plus lent/fragile)
          // — c'etait la cause du plateau lent observe. On cesse seulement
          // d'ESCALADER l'effort Lanczos au-dela du plafond deja atteint (le
          // cout par round reste borne), et on continue les rounds jusqu'a
          // la cible ou max_atoms. lam_best reste protege dans tous les cas.
          if (++regress_streak >= st.atoms_seed_regress_patience && st.verbose >= 1)
            std::printf("[LazyDCA-seed] %lld reculs consecutifs (maxit plafonne a %lld) : "
                        "on continue sans re-escalader, lam_best conserve = %.3e\n",
                        (long long)regress_streak, (long long)maxit_local, lam_best);
          if (st.adaptive_marginal_return_enable && amr21_seed_stop_requested) {
            info.n_adaptive_return_seed_stops++;
            info.adaptive_return_seed_stopped = true;
            info.adaptive_return_seed_stop_atom = (idx_t)atoms.size();
            if (st.verbose >= 1)
              std::printf("[AMRS-v2.3] STOP seed on noisy round: atom=%lld\n",
                          (long long)atoms.size());
            break;
          }
          continue;
        }
        regress_streak = 0;
        const double cover_best = 1.0 - lam_best / lam0;
        const double cover_k = 1.0 - lam_k / lam0;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-seed] atome %lld : lambda %.3e -> %.3e, couverture %.1f%% (cible %.1f%%)\n",
                      (long long)atoms.size(), lam_prev, lam_k, 100.0 * cover_k,
                      100.0 * st.atoms_seed_coverage);
        const bool coverage_hit = cover_best >= st.atoms_seed_coverage ||
                                  cover_k >= st.atoms_seed_coverage;
        lam_best = lam_k;
        if (coverage_hit) {
          seed_coverage_reached = true;
          break;
        }
        // ADAPTIVE_MARGINAL_RETURN_V23: reliable seed progress + robust plateau.
        if (st.adaptive_marginal_return_enable && st.atom_oracle_enable) {
          const double amr23_cover =
              std::max(0.0, std::min(1.0, std::max(cover_best, cover_k)));
          const auto amr23_dec = adaptive_return.observe_seed_progress(
              amr23_cover, std::max(0.0, info.atom_last_lift_rel),
              (std::size_t)atoms.size(), true);
          sync_adaptive_return_info();
          if (amr23_dec.action == AdaptiveReturnAction::StopSeed)
            amr21_seed_stop_requested = true;
          if (st.verbose >= 1)
            std::printf("[AMRS-v2.3] SEED atom=%lld cover=%.6e dcov=%.3e lift=%.3e "
                        "rel=%d(%zu/%lld) plateau=%d(%zu/%lld) "
                        "Hgain=%.3e Hrate=%.3e Hlift=%.3e action=%d\n",
                        (long long)atoms.size(), amr23_dec.seed_coverage,
                        amr23_dec.seed_coverage_gain, amr23_dec.seed_lift_rel,
                        amr23_dec.seed_low_relative ? 1 : 0,
                        amr23_dec.seed_relative_confirmations,
                        (long long)st.adaptive_return_confirmations,
                        amr23_dec.seed_low_plateau ? 1 : 0,
                        amr23_dec.seed_plateau_confirmations,
                        (long long)st.adaptive_return_seed_plateau_confirmations,
                        amr23_dec.seed_plateau_horizon_gain,
                        amr23_dec.seed_plateau_gain_per_commit,
                        amr23_dec.seed_plateau_lift_median,
                        (int)amr23_dec.action);
        }
        // ADAPTIVE_MARGINAL_RETURN_V23: robust seed stop request.
        if (st.adaptive_marginal_return_enable && amr21_seed_stop_requested) {
          info.n_adaptive_return_seed_stops++;
          info.adaptive_return_seed_stopped = true;
          info.adaptive_return_seed_stop_atom = (idx_t)atoms.size();
          if (st.verbose >= 1)
            std::printf("[AMRS-v2.3] STOP seed: atom=%lld reason=%lld cover=%.6e "
                        "dcov=%.3e lift=%.3e Hrate=%.3e Hlift=%.3e\n",
                        (long long)atoms.size(),
                        (long long)info.adaptive_return_seed_stop_reason,
                        info.adaptive_return_seed_stop_coverage,
                        info.adaptive_return_seed_stop_coverage_gain,
                        info.adaptive_return_seed_stop_lift_rel,
                        info.adaptive_return_seed_stop_plateau_gain_per_commit,
                        info.adaptive_return_seed_stop_plateau_lift_median);
          break;
        }
        // ADAPTIVE_MARGINAL_SEED_V2: observe only committed seed progress.
        // Native C5 coverage stop above remains sovereign; later event-driven growth remains active.
        if (st.adaptive_seed_return_enable && st.atom_oracle_enable) {
          const double amrs_cover =
              std::max(0.0, std::min(1.0, std::max(cover_best, cover_k)));
          const auto ad = adaptive_seed_return.observe(
              amrs_cover, std::max(0.0, info.atom_last_lift_rel),
              (std::size_t)atoms.size());
          if (ad.observed) {
            const auto &as = adaptive_seed_return.stats();
            info.n_seed_return_observations = (idx_t)as.observations;
            info.n_seed_return_low_rounds = (idx_t)as.low_return_rounds;
            info.n_seed_return_multi_atom_jumps = (idx_t)as.multi_atom_jumps;
            info.seed_return_seed_atoms_final = (idx_t)atoms.size();
            info.seed_return_cov_ref = as.coverage_ref;
            info.seed_return_lift_ref = as.lift_ref;
            info.seed_return_cov_threshold = as.coverage_threshold;
            info.seed_return_lift_threshold = as.lift_threshold;
            info.seed_return_last_cov_gain = as.last_coverage_gain;
            info.seed_return_last_lift_rel = as.last_lift_rel;
            info.seed_return_recent_cov_median = as.recent_coverage_median;
            info.seed_return_recent_lift_median = as.recent_lift_median;
          }
          if (ad.stop) {
            const auto &as = adaptive_seed_return.stats();
            info.n_seed_return_stops++;
            info.seed_return_stopped = true;
            info.seed_return_stop_atom = (idx_t)as.stop_atom;
            info.seed_return_stop_coverage = as.stop_coverage;
            info.seed_return_stop_cov_gain = as.stop_coverage_gain;
            info.seed_return_stop_lift_rel = as.stop_lift_rel;
            if (st.verbose >= 1)
              std::printf("[LazyDCA-seed-AMRS-v2] STOP: atoms=%lld coverage=%.6e dcov=%.3e lift=%.3e med=(%.3e,%.3e) thr=(%.3e,%.3e) conf=%zu\n",
                          (long long)atoms.size(), as.stop_coverage,
                          as.stop_coverage_gain, as.stop_lift_rel,
                          as.recent_coverage_median, as.recent_lift_median,
                          as.coverage_threshold, as.lift_threshold,
                          ad.confirmations);
            break;
          }
        }
        // ATOM_CLUSTER_PROD_V1: stop initial seeding only.
        // Later event-driven atom acquisition remains active.
        if (st.atom_cluster_enable && st.atom_oracle_enable) {
          const double cover_gate =
              std::max(0.0, std::min(1.0, std::max(cover_best, cover_k)));
          const auto cs = atom_cluster_seed_stop.after_round(
              cover_gate, std::max(0.0, info.atom_last_lift_rel),
              (std::size_t)atoms.size());
          info.atom_cluster_seed_novelty_rate = cs.novelty_rate;
          info.atom_cluster_seed_coverage_gain =
              std::isfinite(cs.coverage_gain) ? cs.coverage_gain : 0.0;
          info.atom_cluster_seed_lift_rel =
              std::isfinite(cs.marginal_lift_rel) ? cs.marginal_lift_rel : 0.0;
          if (cs.stop) {
            info.n_atom_cluster_seed_stops++;
            if (st.verbose >= 1)
              std::printf("[LazyDCA-seed-cluster] STOP: atoms=%lld clusters=%lld novelty=%.3e dcoverage=%.3e lift=%.3e confirmations=%zu\n",
                          (long long)atoms.size(), (long long)atom_clusters.size(),
                          cs.novelty_rate, cs.coverage_gain,
                          cs.marginal_lift_rel, cs.confirmations);
            break;
          }
        }
        // Plateau != continuum : sur un espace propre DEGENERE, lambda reste
        // constant jusqu'au dernier atome du groupe (chaque round ne retire
        // qu'une direction). Patience de N rounds avant de conclure au
        // continuum spectral.
        if (lam_k - lam_prev < st.atoms_seed_stall_tol * std::fabs(lam0)) {
          if (++stall_rounds >= st.atoms_seed_stall_patience) {
            if (st.verbose >= 1)
              std::printf("[LazyDCA-seed] stagnation (continuum spectral, %lld rounds < %.2f%%/round) : arret a %lld atomes, couverture %.1f%%\n",
                          (long long)st.atoms_seed_stall_patience, 100.0 * st.atoms_seed_stall_tol,
                          (long long)atoms.size(), 100.0 * cover_k);
            break;
          }
        } else stall_rounds = 0;
        lam_prev = lam_k;
      }
      // AUTO_FULLCOVERAGE_V4HE0: observe only; no state change.
      info.auto_seed_lambda0 = lam0;
      info.auto_seed_lambda_best = lam_best;
      info.auto_seed_coverage =
          std::clamp(1.0 - lam_best / lam0, 0.0, 1.0);
      info.auto_seed_atoms = atoms.size();

      info.n_seed_atoms = atoms.size();
      info.seed_coverage_est =
          std::max(0.0, 1.0 - lam_best / lam0);
      info.seed_atom_cap_hit =
          (max_seed_atoms_eff < max_atoms_eff &&
           atoms.size() >= max_seed_atoms_eff &&
           !seed_coverage_reached);

      if (info.seed_atom_cap_hit && st.verbose >= 1)
        std::printf(
            "[LazyDCA-seed] CAP initial atteint : %lld/%lld atomes "
            "(max global %lld), couverture Ritz estimee %.4f%%, cible %.4f%%; "
            "croissance post-seed reste autorisee.\n",
            (long long)atoms.size(),
            (long long)max_seed_atoms_eff,
            (long long)max_atoms_eff,
            100.0 * info.seed_coverage_est,
            100.0 * st.atoms_seed_coverage);

      // La proposition de rho (rho_pending, mise a jour par try_add_atoms a
      // chaque round via le mecanisme habituel) peut porter la trace d'un
      // round bruite. On la RE-PROPOSE ici a partir de lam_best (la valeur la
      // plus fiable observee) avant de quitter la phase de seeding.
      // A rejected oracle candidate is not a committed seeding event.
      if (!st.atom_oracle_enable || atoms.size() > 0) {
        double target_best = apply_rho_model_floor(
            std::fabs(std::min(0.0, lam_best)) * (1.0 + st.rho_rel_margin));

        if (st.atom_oracle_enable &&
            st.atom_oracle_directional_residual &&
            !rho_model_fixed)
          target_best = std::max(target_best, rho_oracle_residual_floor);
        rho_pending = (rho_pending >= 0.0) ? std::min(rho_pending, target_best) : target_best;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-seed] BILAN : %lld atomes, couverture %.1f%%, lambda %.3e -> %.3e (best), rho candidat %.3e (seed %.3e)\n",
                      (long long)atoms.size(), 100.0 * (1.0 - lam_best / lam0), lam0, lam_best,
                      rho_pending, rho);
      }
    }
    seeding_phase = false;
    if (reg_lazy && st.atoms_freeze_after_seed && seed_coverage_reached) {
      atom_growth_frozen = true;
      evt_pending_probe = false; evt_has_direction = false;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-seed] GEL post-seeding : %lld atome(s) figes apres atteinte de la couverture cible; aucun ajout/retuning atomique ulterieur.\n",
                    (long long)atoms.size());
    }
  } else if (st.initial_atoms > 0) {
    seeding_phase = true;
    const idx_t fixed_seed_count =
        std::min(st.initial_atoms, max_seed_atoms_eff);
    for (idx_t s2 = 0; s2 < fixed_seed_count; ++s2) {
      if (try_add_atoms(1) == 0) break;
      if (st.adaptive_marginal_return_enable && amr21_seed_stop_requested) {
        info.n_adaptive_return_seed_stops++;
        info.adaptive_return_seed_stopped = true;
        info.adaptive_return_seed_stop_atom = (idx_t)atoms.size();
        break;
      }
    }
    info.n_seed_atoms = atoms.size();
    seeding_phase = false;
    if (reg_lazy && st.atoms_freeze_after_seed && atoms.size() > 0) {
      atom_growth_frozen = true;
      evt_pending_probe = false; evt_has_direction = false;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-seed] GEL post-seeding fixe : %lld atome(s) figes; aucun ajout/retuning atomique ulterieur.\n",
                    (long long)atoms.size());
    }
  }

  // AUTO_LCP_NOTRIGGER_V4HE12
  //
  // v4h-E1.2:
  //
  // Algebraic LCP recognition is independent of the negative-curvature
  // routing decision.
  //
  // If AUTO does NOT enter the concentrated-curvature portfolio but the
  // QP is exactly
  //
  //       min x'(Cx-cl)
  //       s.t. x >= 0, Cx-cl >= 0,
  //
  // then objective = complementarity gap.
  //
  // No portfolio is needed in this regime. Restart the exact native legacy
  // path with a componentwise complementarity tolerance sufficient to imply
  //
  //       |x'w| <= gap_target.
  //
  // The concentrated E1.1 path immediately below is untouched.
  if (st.auto_atoms_enable &&
      reg_lazy &&
      atoms_enabled) {

    const bool auto_concentrated =
        info.auto_seed_atoms > 0 &&
        info.auto_seed_lambda0 < 0.0 &&
        info.auto_seed_coverage >= st.auto_coverage_trigger;

    if (!auto_concentrated &&
        st.auto_lcp_structure_enable) {

      const idx_t nn=P_in.n;
      double struct_res=INF;

      auto detect_notrigger_lcp = [&]() -> bool {
        if (P_in.p != 0 || P_in.m != nn)
          return false;

        if (P_in.C.rows != nn || P_in.C.cols != nn ||
            P_in.Q.U.rows != nn || P_in.Q.U.cols != nn)
          return false;

        if ((idx_t)P_in.q.size()!=nn ||
            (idx_t)P_in.cl.size()!=nn ||
            (idx_t)P_in.cu.size()!=nn ||
            (idx_t)P_in.l.size()!=nn ||
            (idx_t)P_in.u.size()!=nn)
          return false;

        const double tol=
            std::max(0.0,st.auto_lcp_structure_tol);

        double scale=1.0;
        double er=std::fabs(P_in.c0);

        for (idx_t k=0;k<nn;++k) {
          // x >= 0.
          if (!std::isfinite(P_in.l[k]) ||
              std::fabs(P_in.l[k]) > tol)
            return false;

          if (!(std::isinf(P_in.u[k]) && P_in.u[k]>0.0))
            return false;

          // Cx >= cl.
          if (!std::isfinite(P_in.cl[k]))
            return false;

          if (!(std::isinf(P_in.cu[k]) && P_in.cu[k]>0.0))
            return false;

          scale=std::max(
              {scale,
               std::fabs(P_in.q[k]),
               std::fabs(P_in.cl[k])});

          er=std::max(
              er,
              std::fabs(P_in.q[k]+P_in.cl[k]));
        }

        std::map<std::pair<idx_t,idx_t>,double> D;

        // Upper triangle of C+C'.
        for (idx_t col=0;col<P_in.C.cols;++col) {
          for (idx_t pp=P_in.C.colptr[col];
               pp<P_in.C.colptr[col+1];++pp) {

            const idx_t row=P_in.C.rowind[pp];
            const double v=P_in.C.values[pp];

            const idx_t i=std::min(row,col);
            const idx_t j=std::max(row,col);

            D[{i,j}] += (row==col ? 2.0*v : v);
            scale=std::max(scale,std::fabs(v));
          }
        }

        // Minus Q.
        for (idx_t col=0;col<P_in.Q.U.cols;++col) {
          for (idx_t pp=P_in.Q.U.colptr[col];
               pp<P_in.Q.U.colptr[col+1];++pp) {

            const idx_t row=P_in.Q.U.rowind[pp];
            const double v=P_in.Q.U.values[pp];

            const idx_t i=std::min(row,col);
            const idx_t j=std::max(row,col);

            D[{i,j}] -= v;
            scale=std::max(scale,std::fabs(v));
          }
        }

        for (const auto& kv:D)
          er=std::max(er,std::fabs(kv.second));

        struct_res=er/std::max(1.0,scale);

        return struct_res <= tol;
      };

      const bool lcp_like=detect_notrigger_lcp();

      if (lcp_like) {
        info.auto_lcp_structure=true;
        info.auto_lcp_structure_residual=struct_res;

        // Sufficient componentwise tolerance for the aggregate gap:
        //
        // |sum x_i w_i| <= n * ||x_i w_i||_inf.
        double eps_gap_component=st.auto_lcp_eps_comp;

        if (st.auto_lcp_gap_target > 0.0 && nn > 0)
          eps_gap_component=
              std::min(
                  eps_gap_component,
                  st.auto_lcp_gap_target /
                  static_cast<double>(nn));

        const double strict_eps=
            std::min(st.eps_comp,
                     std::max(0.0,eps_gap_component));

        // E14_POSTSOLVE_UNCONDITIONAL_V1
        //
        // Exact-LCP postsolve certification must also run when the caller
        // has already supplied the strict componentwise tolerance.
        // The recursive child disables AUTO below, so recursion terminates.
        if (strict_eps <= st.eps_comp) {
          Settings ls=st;

          // Recursion guard: child is the exact native legacy solve.
          ls.auto_atoms_enable=false;
          ls.auto_portfolio_enable=false;

          ls.eps_comp=strict_eps;

          Result X=solve(P_in,ls);

          // AUTO_LCP_SEMISMOOTH_V4HE14
          //
          // E1.4 is strictly postsolve:
          //   - keep the certified child bit-for-bit;
          //   - otherwise polish F(x)=min(x,Cx-cl);
          //   - accept only with LCP + reconstructed QP-KKT certificate.
          //
          // IPM KKT factorizations are NOT incremented by this stage.

          X.info.auto_triggered=false;
          X.info.auto_scalar_committed=false;
          X.info.auto_portfolio_used=false;
          X.info.auto_selected_mode=0;

          X.info.auto_seed_coverage=info.auto_seed_coverage;
          X.info.auto_seed_lambda0=info.auto_seed_lambda0;
          X.info.auto_seed_lambda_best=info.auto_seed_lambda_best;
          X.info.auto_seed_atoms=info.auto_seed_atoms;

          X.info.auto_lcp_structure=true;
          X.info.auto_lcp_structure_residual=struct_res;

          // Probe cost only; child KKT count remains untouched.
          X.info.n_lanczos_calls += info.n_lanczos_calls;

          // E14_COMMON_FINALIZER_CALL_NATIVE_V1
            return finalize_exact_lcp_e14(
                P_in, st, std::move(X), struct_res, elapsed());
        }
      }
    }
  }

  // AUTO_PORTFOLIO_V4HE11_RUNTIME
  //
  // diffuse curvature -> exact historical legacy continuation
  // concentrated curvature -> conditional native micro-portfolio
  //
  // For an algebraically recognized complementarity QP,
  //
  //   l=0, u=+inf,
  //   Cx >= cl,
  //   Q = C+C',
  //   q = -cl,
  //
  // the QP objective is exactly x'(Cx-cl), the complementarity gap.
  //
  // No problem/benchmark name participates in this decision.
  if (st.auto_atoms_enable &&
      reg_lazy &&
      atoms_enabled &&
      info.auto_seed_atoms > 0 &&
      info.auto_seed_lambda0 < 0.0 &&
      info.auto_seed_coverage >= st.auto_coverage_trigger) {

    const double auto_cov      = info.auto_seed_coverage;
    const double auto_lam0     = info.auto_seed_lambda0;
    const double auto_lam_best = info.auto_seed_lambda_best;
    const idx_t  auto_natoms   = info.auto_seed_atoms;
    const idx_t  probe_lanczos = info.n_lanczos_calls;

    auto annotate_probe = [&](Result& X) {
      X.info.auto_triggered = true;
      X.info.auto_seed_coverage = auto_cov;
      X.info.auto_seed_lambda0 = auto_lam0;
      X.info.auto_seed_lambda_best = auto_lam_best;
      X.info.auto_seed_atoms = auto_natoms;
    };

    // --------------------------------------------------------
    // Intrinsic algebraic detector.
    // --------------------------------------------------------
    double lcp_struct_res = INF;

    auto detect_lcp_structure = [&]() -> bool {
      if (!st.auto_lcp_structure_enable)
        return false;

      const Problem& P0=P_in;
      const idx_t nn=P0.n;

      if (P0.p != 0 || P0.m != nn)
        return false;

      if (P0.C.rows != nn || P0.C.cols != nn ||
          P0.Q.U.rows != nn || P0.Q.U.cols != nn)
        return false;

      if ((idx_t)P0.q.size()!=nn ||
          (idx_t)P0.cl.size()!=nn ||
          (idx_t)P0.cu.size()!=nn ||
          (idx_t)P0.l.size()!=nn ||
          (idx_t)P0.u.size()!=nn)
        return false;

      const double tol=std::max(0.0,st.auto_lcp_structure_tol);
      double scale=1.0;
      double er=std::fabs(P0.c0);

      for (idx_t k=0;k<nn;++k) {
        // x >= 0 only.
        if (!std::isfinite(P0.l[k]) ||
            std::fabs(P0.l[k]) > tol)
          return false;

        if (!(std::isinf(P0.u[k]) && P0.u[k]>0.0))
          return false;

        // Cx >= cl only.
        if (!std::isfinite(P0.cl[k]))
          return false;

        if (!(std::isinf(P0.cu[k]) && P0.cu[k]>0.0))
          return false;

        scale=std::max(
            {scale,std::fabs(P0.q[k]),std::fabs(P0.cl[k])});

        er=std::max(er,std::fabs(P0.q[k]+P0.cl[k]));
      }

      // Upper triangle of C+C' minus Q.
      std::map<std::pair<idx_t,idx_t>,double> D;

      for (idx_t col=0;col<P0.C.cols;++col) {
        for (idx_t pp=P0.C.colptr[col];
             pp<P0.C.colptr[col+1];++pp) {
          const idx_t row=P0.C.rowind[pp];
          const double v=P0.C.values[pp];

          const idx_t i=std::min(row,col);
          const idx_t j=std::max(row,col);

          // diagonal of C+C' is 2*C_ii
          D[{i,j}] += (row==col ? 2.0*v : v);
          scale=std::max(scale,std::fabs(v));
        }
      }

      for (idx_t col=0;col<P0.Q.U.cols;++col) {
        for (idx_t pp=P0.Q.U.colptr[col];
             pp<P0.Q.U.colptr[col+1];++pp) {
          const idx_t row=P0.Q.U.rowind[pp];
          const double v=P0.Q.U.values[pp];

          const idx_t i=std::min(row,col);
          const idx_t j=std::max(row,col);

          D[{i,j}] -= v;
          scale=std::max(scale,std::fabs(v));
        }
      }

      for (const auto& kv:D)
        er=std::max(er,std::fabs(kv.second));

      lcp_struct_res=er/std::max(1.0,scale);
      return lcp_struct_res <= tol;
    };

    const bool lcp_like=detect_lcp_structure();

    struct LcpAudit {
      double feas=INF;
      double comp=INF;
      double gap=INF;
    };

    auto audit_lcp = [&](const Result& X) -> LcpAudit {
      LcpAudit a;

      if (!lcp_like || (idx_t)X.x.size()!=P_in.n)
        return a;

      std::vector<double> cx((size_t)P_in.n,0.0);
      P_in.C.gaxpy(X.x.data(),cx.data());

      double feas=0.0;
      double comp=0.0;
      double gap=0.0;

      for (idx_t k=0;k<P_in.n;++k) {
        const double xk=X.x[k];
        const double wk=cx[k]-P_in.cl[k];
        const double prod=xk*wk;

        feas=std::max(feas,std::max(0.0,-xk));
        feas=std::max(feas,std::max(0.0,-wk));
        comp=std::max(comp,std::fabs(prod));
        gap += prod;
      }

      a.feas=feas;
      a.comp=comp;
      a.gap=gap;
      return a;
    };

    // Generic QP hierarchy.
    auto quality_rank = [&](const Result& X) -> int {
      const double f=X.info.res_feas_rel;
      const double g=X.info.res_stat_rel;
      const double c=X.info.res_comp_rel;

      if (!(std::isfinite(X.info.obj) &&
            std::isfinite(f) &&
            std::isfinite(g) &&
            std::isfinite(c)))
        return 0;

      if (f>st.eps_feas)
        return 0;

      if (X.info.status==Status::SolvedSOC &&
          g<=st.auto_quality_stat_tol &&
          c<=st.auto_quality_comp_tol)
        return 5;

      if (g<=st.eps_stat && c<=st.eps_comp)
        return 4;

      if (g<=st.auto_quality_stat_tol &&
          c<=st.auto_quality_comp_tol)
        return 3;

      return 2;
    };

    if (st.verbose>=1)
      std::printf(
        "[LazyDCA-AUTO-v4hE1.1] concentrated curvature: "
        "coverage %.5f%% seedAtoms=%lld lcpStructure=%s structRes=%.3e\n",
        100.0*auto_cov,
        (long long)auto_natoms,
        lcp_like ? "true" : "false",
        lcp_struct_res);

    // --------------------------------------------------------
    // E0.1 compatibility.
    // --------------------------------------------------------
    if (!st.auto_portfolio_enable) {
      Settings ss=st;
      ss.auto_atoms_enable=false;
      ss.auto_portfolio_enable=false;
      ss.atoms_enable=false;
      ss.atom_oracle_enable=false;
      ss.atom_energy_sparsify=false;
      ss.atom_oracle_retune_enable=false;
      ss.atom_oracle_rhox_rollback=false;

      Result X=solve(P_in,ss);

      const idx_t kkt=X.info.n_factorizations;
      const idx_t lan=X.info.n_lanczos_calls;
      const LcpAudit ax=audit_lcp(X);

      annotate_probe(X);

      X.info.auto_scalar_committed=true;
      X.info.auto_portfolio_used=false;
      X.info.auto_selected_mode=1;
      X.info.auto_lcp_structure=lcp_like;
      X.info.auto_lcp_structure_residual=lcp_struct_res;

      X.info.auto_scalar_factorizations=kkt;
      X.info.auto_portfolio_factorizations=kkt;
      X.info.auto_scalar_objective=X.info.obj;
      X.info.auto_scalar_stat_rel=X.info.res_stat_rel;
      X.info.auto_scalar_feas_rel=X.info.res_feas_rel;
      X.info.auto_scalar_comp_rel=X.info.res_comp_rel;
      X.info.auto_scalar_lcp_feas=ax.feas;
      X.info.auto_scalar_lcp_comp=ax.comp;
      X.info.auto_scalar_lcp_gap=ax.gap;

      X.info.n_factorizations=kkt;
      X.info.n_lanczos_calls=probe_lanczos+lan;

      return X;
    }

    // --------------------------------------------------------
    // ATOMIC child.
    // --------------------------------------------------------
    Settings as=st;

    as.auto_atoms_enable=false;
    as.auto_portfolio_enable=false;

    as.atoms_enable=true;
    as.atom_oracle_enable=true;
    as.atom_energy_sparsify=true;
    as.atom_oracle_retune_enable=true;
    as.atom_oracle_rhox_rollback=false;

    as.max_atoms=
        std::max<idx_t>(
          1,
          std::min<idx_t>(
            st.max_atoms,
            std::max<idx_t>(1,st.auto_atomic_max_atoms)));

    idx_t abudget=std::max<idx_t>(1,st.auto_atomic_kkt_budget);

    if (st.max_kkt_factorizations>0)
      abudget=std::min<idx_t>(
          abudget,st.max_kkt_factorizations);

    as.max_kkt_factorizations=abudget;

    // Historical complementarity-specific child precision.
    if (lcp_like) {
      as.init_strategy=st.auto_lcp_init_strategy;
      as.eps_comp=std::min(as.eps_comp,st.auto_lcp_eps_comp);
    }

    Result A=solve(P_in,as);

    const idx_t akkt=A.info.n_factorizations;
    const idx_t alan=A.info.n_lanczos_calls;
    const double aobj=A.info.obj;
    const double astat=A.info.res_stat_rel;
    const double afeas=A.info.res_feas_rel;
    const double acomp=A.info.res_comp_rel;
    const int qa=quality_rank(A);
    const LcpAudit aa=audit_lcp(A);

    const double ltol=
        std::max(0.0,st.auto_lcp_admissible_tol);

    const bool a_lcp_ok=
        lcp_like &&
        aa.feas<=ltol &&
        aa.comp<=ltol;

    // Generic QP: strong SOC remains a valid early winner.
    //
    // Complementarity QP: SOC alone is NOT sufficient.  The aggregate
    // complementarity gap must already reach its requested target.
    const bool atomic_early =
        (!lcp_like && qa>=5) ||
        (lcp_like &&
         a_lcp_ok &&
         std::fabs(aa.gap)<=st.auto_lcp_gap_target);

    if (atomic_early) {
      annotate_probe(A);

      A.info.auto_scalar_committed=false;
      A.info.auto_portfolio_used=true;
      A.info.auto_selected_mode=2;

      A.info.auto_lcp_structure=lcp_like;
      A.info.auto_lcp_structure_residual=lcp_struct_res;

      A.info.auto_atomic_factorizations=akkt;
      A.info.auto_scalar_factorizations=0;
      A.info.auto_portfolio_factorizations=akkt;

      A.info.auto_atomic_objective=aobj;
      A.info.auto_atomic_stat_rel=astat;
      A.info.auto_atomic_feas_rel=afeas;
      A.info.auto_atomic_comp_rel=acomp;

      A.info.auto_atomic_lcp_feas=aa.feas;
      A.info.auto_atomic_lcp_comp=aa.comp;
      A.info.auto_atomic_lcp_gap=aa.gap;

      A.info.n_factorizations=akkt;
      A.info.n_lanczos_calls=probe_lanczos+alan;

      if (st.verbose>=1)
        std::printf(
          "[LazyDCA-AUTO-v4hE1.1] ATOMIC early winner: "
          "genericQ=%d lcpOk=%s gap=%.3e obj=%.3e KKT=%lld\n",
          qa,
          a_lcp_ok ? "true" : "false",
          aa.gap,aobj,(long long)akkt);

      return A;
    }

    // --------------------------------------------------------
    // SCALAR challenger.
    // --------------------------------------------------------
    idx_t sbudget=std::max<idx_t>(1,st.auto_scalar_kkt_budget);

    if (st.max_kkt_factorizations>0) {
      const idx_t remain=st.max_kkt_factorizations-akkt;

      if (remain<=0) {
        annotate_probe(A);

        A.info.auto_portfolio_used=true;
        A.info.auto_selected_mode=2;
        A.info.auto_lcp_structure=lcp_like;
        A.info.auto_lcp_structure_residual=lcp_struct_res;

        A.info.auto_atomic_factorizations=akkt;
        A.info.auto_portfolio_factorizations=akkt;
        A.info.auto_atomic_lcp_feas=aa.feas;
        A.info.auto_atomic_lcp_comp=aa.comp;
        A.info.auto_atomic_lcp_gap=aa.gap;

        A.info.n_factorizations=akkt;
        A.info.n_lanczos_calls=probe_lanczos+alan;

        return A;
      }

      sbudget=std::min<idx_t>(sbudget,remain);
    }

    Settings ss=st;

    ss.auto_atoms_enable=false;
    ss.auto_portfolio_enable=false;
    ss.atoms_enable=false;
    ss.atom_oracle_enable=false;
    ss.atom_energy_sparsify=false;
    ss.atom_oracle_retune_enable=false;
    ss.atom_oracle_rhox_rollback=false;
    ss.max_kkt_factorizations=sbudget;

    if (lcp_like) {
      ss.init_strategy=st.auto_lcp_init_strategy;
      ss.eps_comp=std::min(ss.eps_comp,st.auto_lcp_eps_comp);
    }

    Result B=solve(P_in,ss);

    const idx_t skkt=B.info.n_factorizations;
    const idx_t slan=B.info.n_lanczos_calls;
    const double sobj=B.info.obj;
    const double sstat=B.info.res_stat_rel;
    const double sfeas=B.info.res_feas_rel;
    const double scomp=B.info.res_comp_rel;
    const int qs=quality_rank(B);
    const LcpAudit sa=audit_lcp(B);

    const bool s_lcp_ok=
        lcp_like &&
        sa.feas<=ltol &&
        sa.comp<=ltol;

    bool choose_scalar=false;

    if (lcp_like) {
      // Exact complementarity-QP:
      // primal feasibility + componentwise complementarity establish
      // admissibility; then minimize the aggregate complementarity gap.
      if (a_lcp_ok && s_lcp_ok) {
        const double scale=
            std::max({1.0,std::fabs(aa.gap),std::fabs(sa.gap)});
        const double tie=1e-14*scale;

        if (sa.gap < aa.gap-tie)
          choose_scalar=true;
        else if (std::fabs(sa.gap-aa.gap)<=tie) {
          if (sa.comp < aa.comp)
            choose_scalar=true;
          else if (sa.comp==aa.comp && skkt<akkt)
            choose_scalar=true;
        }
      } else if (s_lcp_ok) {
        choose_scalar=true;
      } else if (!a_lcp_ok) {
        // Neither candidate is LCP-admissible: fall back to generic quality.
        if (qs>qa)
          choose_scalar=true;
        else if (qs==qa && sobj<aobj)
          choose_scalar=true;
      }
    } else {
      // Generic QP: unchanged E1 quality-first rule.
      if (qs>qa)
        choose_scalar=true;
      else if (qs==qa) {
        const double scale=
            std::max({1.0,std::fabs(aobj),std::fabs(sobj)});
        const double tie=1e-12*scale;

        if (sobj<aobj-tie)
          choose_scalar=true;
        else if (std::fabs(sobj-aobj)<=tie && skkt<akkt)
          choose_scalar=true;
      }
    }

    Result X=choose_scalar ? std::move(B) : std::move(A);

    annotate_probe(X);

    X.info.auto_scalar_committed=choose_scalar;
    X.info.auto_portfolio_used=true;
    X.info.auto_selected_mode=choose_scalar ? 1 : 2;

    X.info.auto_lcp_structure=lcp_like;
    X.info.auto_lcp_structure_residual=lcp_struct_res;

    X.info.auto_atomic_factorizations=akkt;
    X.info.auto_scalar_factorizations=skkt;
    X.info.auto_portfolio_factorizations=akkt+skkt;

    X.info.auto_atomic_objective=aobj;
    X.info.auto_scalar_objective=sobj;

    X.info.auto_atomic_stat_rel=astat;
    X.info.auto_scalar_stat_rel=sstat;
    X.info.auto_atomic_feas_rel=afeas;
    X.info.auto_scalar_feas_rel=sfeas;
    X.info.auto_atomic_comp_rel=acomp;
    X.info.auto_scalar_comp_rel=scomp;

    X.info.auto_atomic_lcp_feas=aa.feas;
    X.info.auto_scalar_lcp_feas=sa.feas;
    X.info.auto_atomic_lcp_comp=aa.comp;
    X.info.auto_scalar_lcp_comp=sa.comp;
    X.info.auto_atomic_lcp_gap=aa.gap;
    X.info.auto_scalar_lcp_gap=sa.gap;

    // Honest portfolio cost.
    X.info.n_factorizations=akkt+skkt;
    X.info.n_lanczos_calls=probe_lanczos+alan+slan;

    if (st.verbose>=1)
      std::printf(
        "[LazyDCA-AUTO-v4hE1.1] selected %s : "
        "LCP=%s ; "
        "A(q=%d,obj=%.3e,gap=%.3e,compLCP=%.3e,KKT=%lld) ; "
        "S(q=%d,obj=%.3e,gap=%.3e,compLCP=%.3e,KKT=%lld) ; "
        "totalKKT=%lld\n",
        choose_scalar ? "SCALAR" : "ATOMIC",
        lcp_like ? "true" : "false",
        qa,aobj,aa.gap,aa.comp,(long long)akkt,
        qs,sobj,sa.gap,sa.comp,(long long)skkt,
        (long long)(akkt+skkt));

    // E14_COMMON_FINALIZER_CALL_PORTFOLIO_V1
    if (lcp_like) {
      X=finalize_exact_lcp_e14(
          P_in, st, std::move(X), lcp_struct_res, elapsed());
    }
    return X;
  }

  if (st.adaptive_marginal_return_enable) {
    info.adaptive_return_atoms_after_seed = (idx_t)atoms.size();
    sync_adaptive_return_info();
  }

  // ---- Certification du rho de seeding AVANT la boucle principale ----
  // Sans ce verrou, un rho_pending issu d'une estimee de Lanczos bruitee
  // (spectre serre/degenere, ex. QPNSTAIR) est commis tel quel ; la premiere
  // factorisation REACTIVE a l'interieur de la boucle principale peut alors
  // spiraler (dizaines de corrections d'inertie, rho_x explosif) SANS que le
  // garde-fou de coherence n'ait la moindre chance de s'executer — il vit
  // dans le corps de la boucle principale, apres le retour de la
  // sous-routine de factorisation, qui elle abandonne (NumericalError) avant
  // de lui rendre la main. On certifie donc ICI, hors boucle chaude, avec une
  // escalade geometrique aggressive (le cout est paye une fois).
  if (atoms.size() > 0) {
    if (atoms_dirty) { kkt.set_atoms(atoms); atoms_dirty = false; }
    for (idx_t i = 0; i < n; ++i) {
      double t = 0.0;
      const double eps_sl = 1e-12 * std::max(1.0, std::fabs(w.x[i]));
      if (w.hl[i]) t += w.vl[i] / std::max(w.x[i] - P.l[i], eps_sl);
      if (w.hu[i]) t += w.vu[i] / std::max(P.u[i] - w.x[i], eps_sl);
      w.thx[i] = t;
    }
    for (idx_t i = 0; i < m; ++i) {
      double t = 0.0;
      // v0.16.0 : garde eps (miroir de thx) — slack exactement nul =>
      // Theta_s=inf => NaN en aval (fausse convergence a obj NaN observee).
      const double eps_ss = 1e-12 * std::max(1.0, std::fabs(w.s[i]));
      if (w.hsl[i]) t += w.wl[i] / std::max(w.s[i] - P.cl[i], eps_ss);
      if (w.hsu[i]) t += w.wu[i] / std::max(P.cu[i] - w.s[i], eps_ss);
      w.ths[i] = std::max(t, 1e-14);
    }
    const double delta0 = st.delta_dual / (1.0 + norm_inf(w.y));
    double rho_model_seed = apply_rho_model_floor((rho_pending >= 0.0) ? rho_pending : rho);

    if (st.atom_oracle_enable &&
        st.atom_oracle_directional_residual &&
        !rho_model_fixed)
      rho_model_seed = std::max(rho_model_seed, rho_oracle_residual_floor);
    double cand0 = rho_model_seed;
    KKT::Fact fc0 = kkt.factorize(w.thx, w.ths, cand0, delta0, info);
    bool ok0 = (fc0 == KKT::Fact::Ok);
    idx_t tries = 0;
    while (!ok0 && fc0 != KKT::Fact::Budget && tries < 20) {
      cand0 = std::max(cand0 * 4.0, 1e-6);
      fc0 = kkt.factorize(w.thx, w.ths, cand0, delta0, info);
      ok0 = (fc0 == KKT::Fact::Ok);
      tries++;
    }
    if (ok0) {
      if (rho_model_fixed) {
        // Keep the DCA model shift locked; any extra amount needed only for
        // the seed factorization is numerical stabilization rho_x.
        rho = rho_model_seed;
        rho_x = std::max(0.0, cand0 - rho);
      } else {
        rho = cand0; rho_x = 0.0;
      }
      rho_pending = -1.0; fact_mu = -1.0; reuse_streak = 0;

      // ADAPTIVE_ORACLE_HYBRID_V4D
      // Atomic model certified at its safe commit level.
      // Release the transient guard; LAZY controls rho afterwards.
      if (st.atom_oracle_enable &&
          st.atom_oracle_directional_residual)
        rho_oracle_residual_floor = 0.0;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-seed] rho de depart CERTIFIE : modele %.3e, rho_x %.3e (%lld essai(s) d'escalade)\n",
                    rho, rho_x, (long long)tries);
    } else if (st.verbose >= 1) {
      std::printf("[LazyDCA-seed] ATTENTION : rho de depart NON certifie apres %lld essais "
                  "(dernier total %.3e, modele verrouille %.3e) — la boucle principale escaladera rho_x de maniere reactive.\n",
                  (long long)tries, cand0, rho_model_seed);
    }
  }

  info.rho_initial = rho;

  if (st.atom_oracle_enable && st.verbose >= 1)
    std::printf(
        "[LazyDCA-atomOracle-v4] directional_policy=%s "
        "theta=%.3e activity_scaling=%s "
        "min_scaled_activity=%.3e residual_floor=%.3e\n",
        st.atom_oracle_directional_residual ? "residual" : "reject",
        st.atom_oracle_min_directional_ratio,
        st.atom_oracle_activity_sqrt_nfree ? "sqrt-nfree" : "raw",
        st.atom_oracle_min_scaled_activity,
        rho_oracle_residual_floor);
  if (st.verbose >= 1 && reg_controlled) {
    std::printf("[LazyDCA-rho] controlled OMS mode=%s, H_k=%s, rho_model=%s, rho_initial=%.6e, rho_model_min=%.6e, rho_model_lock=%s, freeze_after_seed=%s, atoms_frozen=%s, lazy_no_atoms=%s, atom_max_nnz=%lld, sparse_beta_recert=%s, atom_oracle=%s, energy_sparsify=%s, energy_target=%.6f\n",
                regularization_mode_name(st.regularization_mode),
                atoms.size() > 0 ? "low-rank" : "0", rho_model_fixed ? "fixed" : "adaptive",
                rho, rho_model_floor, st.rho_model_lock ? "true" : "false",
                st.atoms_freeze_after_seed ? "true" : "false",
                atom_growth_frozen ? "true" : "false",
                (reg_lazy && !atoms_enabled) ? "true" : "false",
                (long long)st.atom_max_nnz, st.atom_sparse_recertify_beta ? "true" : "false",
                st.atom_oracle_enable ? "true" : "false",
                st.atom_energy_sparsify ? "true" : "false", st.atom_energy_target);
  }

  // =============================== main loop ================================
  // KKT_FACTORIZATION_BUDGET_V4G: seed may itself exhaust a deliberately tiny scouting budget.
  if (info.status == Status::FactorizationBudget) goto finish;
  for (info.iters = 0; info.iters < st.max_iter; ++info.iters) {
    if (elapsed() > st.time_limit) { info.status = Status::TimeLimit; break; }
    // Heartbeat : a l'affichage DCA (verbose 1-2), rien ne s'imprime entre deux
    // accepts -- apres une restauration/re-ancrage le modele peut rejeter en
    // silence pendant longtemps (nouvelles factorisations, corrections
    // d'inertie, Armijo qui recule), donnant l'illusion d'un blocage. On
    // imprime un signe de vie regulier tant qu'aucun accept n'arrive.
    iters_since_last_accept++;
    if (st.verbose >= 1 && st.verbose < 3 && st.heartbeat_every > 0 &&
        iters_since_last_accept % st.heartbeat_every == 0) {
      Residuals Rhb = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
      std::printf("       ... %lld iterations sans accept (inf_du=%.3e feas=%.3e rho=%.3e t=%.1fs)\n",
                  (long long)iters_since_last_accept, Rhb.stat_abs, Rhb.feas_rel, rho, elapsed());
    }
    // Soupape anti-blocage : la decroissance certifiee ne certifie QUE la
    // factorisation (inertie), pas que les pas resultants seront ACCEPTABLES.
    // Sur une longue serie de rejets, rho est probablement devenu trop petit
    // pour amortir correctement le pas (observe : 100-140 rejets consecutifs
    // apres une cascade cut x16). Symetrique de la decroissance sur succes :
    // on REMONTE rho sur une serie d'echecs, au lieu d'attendre soit un accept
    // chanceux, soit le seuil catastrophique du garde-fou de coherence.
    const bool reject_streak_has_regularization =
        (reg_const || reg_dyn) ? (rho > 0.0 || rho_x > 0.0) : (rho > 0.0);
    if (st.reject_streak_rho_bump > 0 && reject_streak_has_regularization &&
        iters_since_last_accept % st.reject_streak_rho_bump == 0) {
      if (reg_const || reg_dyn || rho_model_fixed) {
        // For the controlled CONST/DYN ablations rho_x is deliberately kept
        // separate from the DC model shift.  A reject streak is not, by
        // itself, evidence that Q+rho I lost convexity.
        const double current_total = rho + rho_x;
        if (current_total >= st.rho_abs_max) {
          classify_budget_exhausted("rho_abs_max (reject-streak safety valve)",
              "reject-streak safety valve: total primal regularization already at/above rho_abs_max "
              "(" + std::to_string(iters_since_last_accept) +
              " consecutive rejects, no admissible step found)");
          if (st.verbose >= 1)
            std::printf("[LazyDCA-rho] soupape : rho+rho_x=%.3e deja au plafond rho_abs_max=%.3e "
                        "apres %lld rejets consecutifs -- abandon propre\n",
                        current_total, st.rho_abs_max, (long long)iters_since_last_accept);
          goto finish;
        }
        const double total_bumped = std::min(
            std::max(current_total * 4.0, 10.0 * st.rho_floor), st.rho_abs_max);
        if (rho_model_fixed) {
          const double old_rx = rho_x;
          rho_x = std::max(0.0, total_bumped - rho);
          if (st.verbose >= 1)
            std::printf("[LazyDCA-rho CONST] streak de %lld rejets : rho modele FIXE %.3e, "
                        "rho_x %.3e -> %.3e\n",
                        (long long)iters_since_last_accept, rho, old_rx, rho_x);
        } else { // DYN
          // Change the DC shift only if a fresh curvature probe asks for it;
          // otherwise this is treated as a numerical KKT-conditioning event.
          double lhat = std::numeric_limits<double>::quiet_NaN();
          const double target = dyn_scalar_target(149u + (unsigned)(info.iters % 10007), &lhat);
          if (target > rho) {
            const double old_rho = rho;
            rho = target; rho_x = 0.0; rho_cut = 2.0;
            info.n_rho_model_updates++; info.n_rho_model_increases++;
            if (st.verbose >= 1)
              std::printf("[LazyDCA-rho DYN] streak de %lld rejets + lambda_hat=%.3e : rho %.3e -> %.3e\n",
                          (long long)iters_since_last_accept, lhat, old_rho, rho);
          } else {
            // Globalization event: the DC floor is already satisfied, but the
            // current proximal model is not producing an accepted step.
            // Increase the MODEL regularization above the certified floor;
            // later accepted anchors decay it back toward that floor.
            const double old_rho = rho;
            const double bumped = rho_dyn_growth_candidate(
                rho, target + st.rho_init_abs_margin,
                st.rho_dyn_reject_growth, st.rho_abs_max);
            if (bumped > rho) {
              rho = bumped; rho_x = 0.0; rho_pending = -1.0; rho_cut = 2.0;
              info.n_rho_model_updates++; info.n_rho_model_increases++;
              if (st.verbose >= 1)
                std::printf("[LazyDCA-rho DYN] streak de %lld rejets : "
                            "rho modele %.3e -> %.3e (plancher DC %.3e)\n",
                            (long long)iters_since_last_accept, old_rho, rho, target);
            }
          }
        }
      } else {
        // Legacy/lazy path: preserve the v0.17.1 safety-valve semantics
        // exactly, so selecting a new OMS mode is opt-in and does not alter
        // the historical default trajectory.
        if (rho >= st.rho_abs_max) {
          classify_budget_exhausted("rho_abs_max (reject-streak safety valve)",
              "reject-streak safety valve: rho already at/above rho_abs_max "
              "(" + std::to_string(iters_since_last_accept) +
              " consecutive rejects, no admissible step found)");
          if (st.verbose >= 1)
            std::printf("[LazyDCA-rho] soupape : rho=%.3e deja au plafond rho_abs_max=%.3e "
                        "apres %lld rejets consecutifs -- abandon propre (pas de spirale)\n",
                        rho, st.rho_abs_max, (long long)iters_since_last_accept);
          goto finish;
        }
        const double old_rho = rho;
        const double rho_bumped = std::min(
            std::max(rho * 4.0, 10.0 * st.rho_floor), st.rho_abs_max);
        if (st.verbose >= 1)
          std::printf("[LazyDCA-rho] streak de %lld rejets : rho %.3e -> %.3e (soupape anti-blocage)\n",
                      (long long)iters_since_last_accept, rho, rho_bumped);
        rho = rho_bumped; rho_x = 0.0; rho_cut = 2.0;
        if (rho != old_rho) { info.n_rho_model_updates++; info.n_rho_model_increases++; }
      }
      fact_mu = -1.0; reuse_streak = 0; rho_pending = -1.0;
    }
    // TERMINAL_QUALIFICATION_V261: certify the DCA stall first. If it is
    // feasible and near the strict KKT set, allow one explicit 50-iteration
    // recovery. Outside that episode the historical DCA stop remains active.
    if (terminal_recovery_active && terminal_recovery_deadline >= 0 &&
        info.iters >= terminal_recovery_deadline) {
      Residuals Rbudget = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
      const bool strict_budget_kkt =
          Rbudget.feas_rel <= st.eps_feas &&
          Rbudget.stat_rel <= st.eps_stat &&
          Rbudget.comp_rel <= st.eps_comp;
      if (strict_budget_kkt) {
        info.status = Status::Solved;
        info.message = "original-QP KKT certificate reached during bounded terminal recovery";
        converged = true;
        terminal_recovery_active = false;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-terminal-v262] strict KKT reached within recovery start=%lld deadline=%lld now=%lld\n",
                      (long long)terminal_recovery_start, (long long)terminal_recovery_deadline,
                      (long long)info.iters);
        break;
      }
      if (terminal_recovery_entry_valid) {
        w = terminal_recovery_entry_work;
        mu = terminal_recovery_entry_mu;
        rho = terminal_recovery_entry_rho;
        rho_x = terminal_recovery_entry_rho_x;
        rho_pending = terminal_recovery_entry_rho_pending;
        rho_cut = terminal_recovery_entry_rho_cut;
        fact_mu = -1.0; reuse_streak = 0;
      }
      terminal_recovery_active = false;
      terminal_polish_active = false;
      dca_converged_flag = true;
      info.status = Status::DCAConverged;
      info.message = "bounded terminal recovery exhausted; pre-recovery iterate restored";
      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v262] recovery exhausted start=%lld deadline=%lld now=%lld; restored entry iterate\n",
                    (long long)terminal_recovery_start, (long long)terminal_recovery_deadline,
                    (long long)info.iters);
    }

    if (dca_converged_flag) {
      Residuals Rdca = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
      const bool strict_dca_kkt =
          Rdca.feas_rel <= st.eps_feas &&
          Rdca.stat_rel <= st.eps_stat &&
          Rdca.comp_rel <= st.eps_comp;
      if (strict_dca_kkt) {
        info.status = Status::Solved;
        info.message = "DCA objective stall with original-QP KKT certificate";
        converged = true;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-terminal-v262] DCA stall is strict original-QP KKT; solved at it=%lld\n",
                      (long long)info.iters);
        break;
      }

      const bool near_terminal =
          st.terminal_polish &&
          Rdca.feas_rel <= st.eps_feas &&
          Rdca.stat_rel <= st.terminal_anchor_stat_trigger &&
          Rdca.comp_rel <= std::max(10.0 * st.eps_comp, terminal_recovery_comp_trigger);

      if (near_terminal && !terminal_recovery_attempted) {
        terminal_recovery_attempted = true;
        terminal_recovery_active = true;
        terminal_recovery_entry_valid = true;
        terminal_recovery_start = info.iters;
        terminal_recovery_deadline = info.iters + terminal_recovery_budget;
        terminal_recovery_entry_work = w;
        terminal_recovery_entry_mu = mu;
        terminal_recovery_entry_rho = rho;
        terminal_recovery_entry_rho_x = rho_x;
        terminal_recovery_entry_rho_pending = rho_pending;
        terminal_recovery_entry_rho_cut = rho_cut;

        dca_converged_flag = false;
        dca_stall_streak = 0;
        dca_delta_hist.clear();
        terminal_polish_active = true;
        rho_pending = -1.0;
        reset_momentum_epoch("DCA-stall terminal qualification recovery");
        x_ref = w.x;
        dca_inner = 0;
        dca_model_start = std::numeric_limits<double>::infinity();
        fact_mu = -1.0; reuse_streak = 0;
        if (Rdca.comp_rel <= st.eps_comp) {
          terminal_best_work = w;
          terminal_best_stat = Rdca.stat_rel;
          terminal_best_valid = true;
        }
        if (st.verbose >= 1)
          std::printf("[LazyDCA-terminal-v262] recovery start=%lld deadline=%lld stat=%.3e comp=%.3e\n",
                      (long long)terminal_recovery_start, (long long)terminal_recovery_deadline,
                      Rdca.stat_rel, Rdca.comp_rel);
      } else {
        break;
      }
    }

    R = compute_residuals(P, w, mu);
    refresh_momentum_force();
    // stationnarite du MODELE (espace scale) : DOIT etre calculee ICI, avant
    // residuals_original qui ecrase w.Qx/w.Aty/w.Ctz avec les produits du
    // probleme ORIGINAL (melange d'espaces sinon).
    double stat_model = 0.0;
    double stat_model_base = 0.0;
    auto compute_complete_stat = [&]() {
      double sb=0.0,si=0.0; std::vector<double> pgm(n);
      for(idx_t i2=0;i2<n;++i2) pgm[i2]=rho*(w.x[i2]-x_ref[i2]);
      if(atoms.size()){std::vector<double>dref(n);for(idx_t i2=0;i2<n;++i2)dref[i2]=w.x[i2]-x_ref[i2];atoms.apply(dref.data(),pgm.data());}
      for(idx_t i2=0;i2<n;++i2){const double common=w.Qx[i2]+P.q[i2]+pgm[i2]-w.Aty[i2]-w.Ctz[i2]-w.vl[i2]+w.vu[i2];sb=std::max(sb,std::fabs(common));si=std::max(si,std::fabs(common-momentum_force[i2]));}
      return std::pair<double,double>{sb,si};
    };
    const double stat_scale=(R.stat_rel>0?std::max(1.0,R.stat_abs/R.stat_rel):1.0);
    auto stpair=compute_complete_stat();stat_model_base=stpair.first;stat_model=stpair.second;
    double stat_base_rel_guard=stat_model_base/stat_scale;
    double stat_inertial_rel_guard=stat_model/stat_scale;
    if(st.dca_momentum_complete && info.momentum_force_inf_final>0.0 &&
       stat_inertial_rel_guard > st.dca_momentum_stat_guard_factor*std::max({stat_base_rel_guard,st.model_solved_tol,1e-16})) {
      if(shrink_complete("model-stationarity guard")){
        info.n_momentum_stat_guard_shrinks++;refresh_momentum_force();stpair=compute_complete_stat();
        stat_model_base=stpair.first;stat_model=stpair.second;
        stat_base_rel_guard=stat_model_base/stat_scale;stat_inertial_rel_guard=stat_model/stat_scale;
      }
    }
    stat_model_rel_iter=stat_inertial_rel_guard;info.stat_model_base_rel=stat_base_rel_guard;info.stat_model_inertial_rel=stat_model_rel_iter;
    if(st.verbose>=2&&info.iters%100==0)std::printf("[LazyDCA-model] it=%lld stat_inertial=%.3e stat_base=%.3e stat_scaled=%.3e |p|=%.3e mu=%.1e rho=%.1e\n",(long long)info.iters,stat_model_rel_iter,info.stat_model_base_rel,R.stat_rel,info.momentum_force_inf_final,mu,rho);
    // L1 barrier update, gated as in the paper's Algorithm 2:
    // reduce mu only once the barrier-KKT residual satisfies ||F_mu|| <= theta*mu.
    // Gate RELATIF (invariant d'echelle) : sur des problemes a gradients ~1e5,
    // le residu absolu ne passe jamais sous theta*mu et mu reste bloque a mu0
    // (sur-resolution de la barriere en rampant sur la courbure negative).
    const double xsc_g = 1.0 + norm_inf(w.x);
    // stationnarite du MODELE (ancre x_ref) pour piloter mu : entre deux
    // acceptations, c'est le sous-probleme convexe que la barriere resout.

    const double Fmu_now = std::max({stat_model_rel_iter, R.feas_rel, R.comp_dev / xsc_g});
    // plancher de mu lie a la precision cible : en dessous, le pas de Newton
    // tente une correction de complementarite geante que le merite rejette.
    const double mu_min = std::max(1e-13, 0.1 * st.eps_comp);
    // re-centrage : mu tres en dessous du gap reel = desalignement primal-dual
    // (spirale d'echecs Armijo) -> remonter mu vers le gap mesure.
    const double mu_gap = R.gap / (double)R.ncomp;
    if (mu_gap > 0 && mu < 1e-3 * mu_gap) {
      mu = std::max(mu_min, 0.1 * mu_gap); mu_inner = 0;
      if (st.verbose >= 2) std::printf("[LazyDCA-mu] re-centrage vers le gap: mu=%.2e\n", mu);
    }
    mu_inner++;
    const bool gate_ok = Fmu_now <= st.barrier_theta * std::max(mu, mu_min);
    const bool inner_cap = st.mu_inner_max > 0 && mu_inner >= st.mu_inner_max;
    // debloqueur : mu gele (>mu_stall_iters sans bouger, gate ferme) => un pas
    // sigma-rule force (la pathologie NCVXQP3 : mu fige a 1e-2 pendant ~1000
    // iterations, gate structurellement infranchissable en re-ancrage/iter).
    if (!st.mehrotra_enable && R.ncomp > 0 &&
        info.iters >= mu_last_change_iter + st.mu_stall_iters &&
        mu > std::max(1e-13, 0.01 * st.eps_comp)) {
      double mu_new2 = std::max(std::max(1e-13, 0.01 * st.eps_comp),
                                std::min(st.barrier_r * mu, 0.2 * R.gap / (double)R.ncomp));
      if (mu_new2 < mu) {
        mu = mu_new2; mu_last_change_iter = info.iters; fact_mu = -1.0; reuse_streak = 0;
        if (st.verbose >= 2) std::printf("[LazyDCA-mu] debloqueur sigma : mu -> %.2e (gel > %lld its)\n", mu, (long long)st.mu_stall_iters);
      }
    }
    if (!st.mehrotra_enable && (gate_ok || inner_cap)) {   // baisse pilotee par Mehrotra sinon
                                  // comp peut encore avoir des decades a faire)
      double mu_new = std::max(mu_min, std::min(st.barrier_r * mu, st.barrier_r * mu_gap));
      if (mu_new < mu) { mu_last_change_iter = info.iters; mu = mu_new; mu_inner = 0; }
    }

    // terminal regime ("accurate terminal solve"): feasibility and
    // complementarity converged, only stationarity (the -rho*dx DCA
    // linearization tail) lags => drive rho down aggressively; the inertia
    // certificate protects against over-decay.
    const bool terminal = (R.feas_rel <= st.eps_feas && R.comp_rel <= 10.0 * st.eps_comp &&
                           R.stat_rel > st.eps_stat);
    term_floor = std::max(term_floor * 0.5, rho_decay_floor);
    if (terminal && rho_x > term_floor && armijo_fail_streak == 0 &&
        last_apr >= 0.1 && info.iters >= last_rho_change_iter + 5) {
      double cand = std::max({rho_x * st.rho_decay * st.rho_decay, term_floor, rho_decay_floor});
      if (cand < rho_x) {
        rho_x = (cand < st.rho_floor) ? 0.0 : cand; last_rho_change_iter = info.iters;
        if (st.verbose >= 2) std::printf("[LazyDCA-rho] terminal decay -> %.3e (floor %.1e)\n", rho, term_floor);
      }
    }

    // diagonal barrier terms
    for (idx_t i = 0; i < n; ++i) {
      double t = 0.0;
      const double eps_sl = 1e-12 * std::max(1.0, std::fabs(w.x[i]));
      if (w.hl[i]) t += w.vl[i] / std::max(w.x[i] - P.l[i], eps_sl);
      if (w.hu[i]) t += w.vu[i] / std::max(P.u[i] - w.x[i], eps_sl);
      w.thx[i] = t;
    }
    for (idx_t i = 0; i < m; ++i) {
      double t = 0.0;
      // v0.16.0 : garde eps (miroir de thx) — slack exactement nul =>
      // Theta_s=inf => NaN en aval (fausse convergence a obj NaN observee).
      const double eps_ss = 1e-12 * std::max(1.0, std::fabs(w.s[i]));
      if (w.hsl[i]) t += w.wl[i] / std::max(w.s[i] - P.cl[i], eps_ss);
      if (w.hsu[i]) t += w.wu[i] / std::max(P.cu[i] - w.s[i], eps_ss);
      w.ths[i] = std::max(t, 1e-14);
    }

    // ---------------- L2: lazy convexification + factorization -------------
    // Two-branch inertia correction:
    //   neg > p+m or Fail  => curvature deficit  => escalate rho (capped);
    //   neg < p+m or null  => rank/conditioning  => escalate dual reg delta.
    if (atoms_dirty) { kkt.set_atoms(atoms); atoms_dirty = false; fact_mu = -1.0; }

    bool clean_first_try = true;
    const double rhox_entry = rho_x;   // valeur testee en premier cette iteration
    idx_t curvfail_streak = 0;
    // PLANCHER ABSOLU sur la regularisation duale : la forme 1e-9/(1+|y|)
    // etait auto-aggravante en degenerescence (|y|~1e6 -> delta=1e-15 :
    // bloc (2,2) numeriquement nul, quasi-definitude LDL detruite, et plus
    // les multiplicateurs derivent moins ils sont bornes). Tikhonov :
    // |y| <~ residu/delta — le plancher borne le continuum dual.
    double delta = st.delta_dual / (1.0 + norm_inf(w.y));
    // ---- KKT factorization reuse (inexact Newton, line-search safeguarded):
    // same mu, same rho, previous iteration clean with near-full step, and the
    // barrier residual decreased => skip the (dominant-cost) factorization.
    const bool did_reuse =
        st.kkt_reuse && last_clean_full && !atoms_dirty && fact_mu == mu && fact_rho == rho + rho_x &&
        reuse_streak < st.kkt_reuse_max && Fmu_now < prev_Fmu;
    prev_Fmu = Fmu_now;
    if (did_reuse) { info.n_kkt_reuses++; reuse_streak++; rho_pending = -1.0; }
    else while (true) {
      // Decrease of the MODEL rho.  In DYN, a smaller candidate must first
      // pass the same numerical DC-curvature gate used to construct the
      // scalar target; KKT factorization is then an additional linear-algebra
      // gate, never the sole reason for changing the DC decomposition.
      if (!rho_model_fixed && rho_pending >= 0.0) {
        const double cand = apply_rho_model_floor(rho_pending); rho_pending = -1.0;
        bool curvature_ok = true;
        if (reg_dyn && cand < rho) {
          double lhat2 = std::numeric_limits<double>::quiet_NaN();
          double lguard2 = std::numeric_limits<double>::quiet_NaN();
          const double target2 = dyn_scalar_target(
              307u + (unsigned)((info.iters + info.n_rho_curvature_checks) % 10007),
              &lhat2, &lguard2);
          curvature_ok = rho_candidate_passes_curvature(
              lguard2, cand, st.rho_curvature_tol);
          if (!curvature_ok) {
            info.n_rho_curvature_rejects++;
            rho_cut = 2.0;
            rho_spacing_cur = std::min(rho_spacing_cur * 2, 8 * st.rho_trial_every);
            const double tol_rho = 1e-12 * std::max(1.0, std::fabs(rho));
            if (target2 > rho + tol_rho) {
              const double old_rho = rho;
              rho = target2; rho_x = 0.0;
              info.n_rho_model_updates++; info.n_rho_model_increases++;
              if (st.verbose >= 1)
                std::printf("[LazyDCA-rho DYN] baisse %.3e refusee par courbure "
                            "(lambda_hat=%.3e guard=%.3e) ; rho %.3e -> %.3e\n",
                            cand, lhat2, lguard2, old_rho, rho);
            } else if (target2 + tol_rho < rho) {
              // Re-propose the updated curvature target; it will itself be
              // checked again when the next trial is committed.
              rho_pending = target2;
              if (st.verbose >= 1)
                std::printf("[LazyDCA-rho DYN] baisse %.3e refusee par courbure "
                            "(lambda_hat=%.3e guard=%.3e) ; nouvelle cible %.3e\n",
                            cand, lhat2, lguard2, target2);
            } else if (st.verbose >= 1) {
              std::printf("[LazyDCA-rho DYN] baisse %.3e refusee par courbure "
                          "(lambda_hat=%.3e guard=%.3e) ; rho conserve %.3e\n",
                          cand, lhat2, lguard2, rho);
            }
          }
        }

        KKT::Fact rho_trial_fc = KKT::Fact::Fail;
        if (cand < rho && curvature_ok) {
          rho_trial_fc = kkt.factorize(w.thx, w.ths, cand, delta, info);
          if (rho_trial_fc == KKT::Fact::Budget) goto finish;
        }
        // KKT_FACTORIZATION_BUDGET_V4G: Budget is a voluntary stop, never a curvature rejection.
        if (cand < rho && curvature_ok &&
            rho_trial_fc == KKT::Fact::Ok) {
          const double old_rho = rho;
          if (st.verbose >= 1) {
            if (reg_dyn)
              std::printf("[LazyDCA-rho DYN] decroissance acceptee (courbure + KKT) : "
                          "rho modele %.3e -> %.3e (rho_x=0)\n", rho, cand);
            else
              std::printf("[LazyDCA-rho] decroissance certifiee : rho modele %.3e -> %.3e "
                          "(rho_x=0, cut x%.0f)\n", rho, cand, rho_cut);
          }
          rho = cand; rho_x = 0.0;
          if (rho != old_rho) { info.n_rho_model_updates++; info.n_rho_model_decreases++; }
          rho_cut = std::min(rho_cut * 2.0, 16.0);            // accelerer (x2 par succes)
          rho_spacing_cur = st.rho_trial_every;               // cadence pleine
          break;                       // factorisation deja faite et propre
        } else if (cand < rho && curvature_ok) {
          rho_cut = 2.0;                                      // prudence
          rho_spacing_cur = std::min(rho_spacing_cur * 2, 8 * st.rho_trial_every);
          // Historical lazy/legacy bisection after a KKT-factorization
          // refusal.  DYN can use it too, but the resulting candidate must
          // pass the curvature gate above before it can be committed.
          if (amr21_atom_event_budget_ok()) {
            // [B] rho veut descendre, l'inertie refuse : de la courbure non
            // couverte bloque — c'est le moment exact d'un atome cible.
            last_event_atom_iter = info.iters;
            if (!evt_pending_probe) {
              evt_pending_probe = true; evt_pending_tag = 'B'; evt_pending_since = info.iters;
              evt_has_direction = false;
              if (st.verbose >= 1)
                std::printf("[LazyDCA-atomEvt B] essai rho %.3e refuse : sonde en attente d'accept\n", cand);
            }
          }
          if (rho_bisect_left > 0) {
            rho_bisect_left--;
            rho_pending = std::sqrt(std::max(cand, 1e-12) * rho);
            if (st.verbose >= 2)
              std::printf("[LazyDCA-rho] essai %.3e refuse, bisection -> %.3e (%lld restants)\n",
                          cand, rho_pending, (long long)rho_bisect_left);
          }
        }
      }
      KKT::Fact fc = kkt.factorize(w.thx, w.ths, rho + rho_x, delta, info);
      if (fc == KKT::Fact::Budget) goto finish;
      if (fc == KKT::Fact::Ok) break;
      clean_first_try = false;
      // KKT_FACTORIZATION_BUDGET_V4G: only genuine factorization/inertia failures count here.
      info.n_inertia_corrections++;
      if (fc == KKT::Fact::Fail) {
        if (st.verbose >= 2) std::fprintf(stderr, "[LazyDCA-inertia] MUMPS factorization error INFOG(1)=%d\n", kkt.info1());
        // ---- Etage 1 (v0.16.0) : sauvetage par push selectif, gate par la
        // SIGNATURE slack pathologique. Signature absente => le push ne
        // servirait a rien (ce n'est pas un probleme de bord) : comportement
        // 0.15.x inchange. Jamais declenche par un CurvatureDeficit.
        if (st.bp_adaptive && bp_events < st.bp_max_events && bp_signature()) {
          bp_events++; info.n_bp_events++;
          if (bp_push_escalate(mu, "fact_fail") > 0) {
            bp_recompute_theta();
            delta = st.delta_dual / (1.0 + norm_inf(w.y));  // retry propre
            fact_mu = -1.0; reuse_streak = 0;
            continue;
          }
        }
        info.status = Status::NumericalError;
        info.message = "MUMPS factorization error INFOG(1)=" + std::to_string(kkt.info1());
        goto finish;
      }
      if (fc == KKT::Fact::RankDeficit) {
        delta = std::max(delta * 100.0, 1e-8);
        if (st.verbose >= 2) std::printf("[LazyDCA-inertia] neg=%d (target %lld) rank/cond -> delta=%.1e\n",
                                         kkt.neg_pivots(), (long long)(p + m), delta);
        if (delta > 1e-2) {
          // ---- Etage 1 (v0.16.0) : meme sauvetage sur plafond de
          // regularisation duale — des slacks ecrases rendent le bloc (1,1)
          // si dominant qu'aucun delta raisonnable ne restaure le rang.
          if (st.bp_adaptive && bp_events < st.bp_max_events && bp_signature()) {
            bp_events++; info.n_bp_events++;
            if (bp_push_escalate(mu, "delta_cap") > 0) {
              bp_recompute_theta();
              delta = st.delta_dual / (1.0 + norm_inf(w.y));
              fact_mu = -1.0; reuse_streak = 0;
              continue;
            }
          }
          info.status = Status::NumericalError; info.message = "dual regularization exceeded cap"; goto finish;
        }
        continue;
      }
      // curvature branch : escalade de rho SEULE (zip : la chaine d'inertie
      // n'ajoute pas d'atomes ; ils viennent du calendrier d'acceptation).
      // Au 2e echec consecutif : saut direct a la borne de Gershgorin
      // plein-espace (garantie K_hat PSD) au lieu de l'echelle x10 —
      // supprime le flicker +-1 des sondes marginales et le combat
      // escalade/decay (zip : bisection bornee par Gershgorin).
      curvfail_streak++;
      if (curvfail_streak >= 2 && gersh_lb < 0.0) {
        double gtar = std::fabs(gersh_lb) * (1.0 + st.rho_rel_margin);
        if (rho + rho_x < gtar) {
          rho_x = gtar - rho; rho_cap = std::max(rho_cap, gtar);
          rho_decay_floor = std::max(rho_decay_floor, 0.9 * rho_x); floor_mu = mu;
          if (st.verbose >= 2) std::printf("[LazyDCA-inertia] saut Gershgorin -> rho_x=%.3e (garanti, modele intact)\n", rho_x);
          info.n_inertia_corrections++;
          continue;
        }
      }
      if (rho_cap < 0.0) rho_cap = compute_cap();
      double rx_old = rho_x;
      if (rho + rho_x <= 0.0) rho_x = std::max({st.rho_floor, 1e-8 * (1.0 + P.Q.U.inf_norm()), 1e-4 * rho_cap});
      else if (rho_x <= 0.0) rho_x = std::max(st.rho_floor, 1e-4 * std::max(rho_cap, rho));
      else rho_x *= st.rho_growth;
      if (rho + rho_x >= rho_cap) {
        if (rho + rx_old >= rho_cap) {  // sous-estimation : agrandir le cap
          rho_cap *= st.rho_growth;
          if (st.verbose >= 2) std::printf("[LazyDCA-rho] cap enlarged to %.3e (residual curvature)\n", rho_cap);
        }
        rho_x = std::min(rho_x, std::max(rho_cap - rho, st.rho_floor));
      }
      term_floor = std::max(term_floor, 1.2 * rx_old);  // rho_old had bad inertia
      if (st.verbose >= 2) std::printf("[LazyDCA-inertia] neg=%d (target %lld) curvature -> rho_x %.3e => %.3e (modele rho=%.3e fixe)\n",
                                  kkt.neg_pivots(), (long long)(p + m), rx_old, rho_x, rho);
      if (rho + rho_x > st.rho_abs_max) {
        classify_budget_exhausted("rho_abs_max (curvature escalation)", "rho exceeded absolute cap");
        goto finish;
      }
    }
    if (!did_reuse) { fact_mu = mu; fact_rho = rho + rho_x; reuse_streak = 0; }
    if (!did_reuse && clean_first_try) {
      if (++clean_streak >= st.rho_clean_streak && rho_x > rho_decay_floor &&
          last_apr >= 0.1 && info.iters >= last_rho_change_iter + 5) {
        double cand = std::max(rho_x * st.rho_decay, rho_decay_floor);
        if (cand < rho_x) {
          if (st.verbose >= 2) std::printf("[LazyDCA-rho] lazy decay rho_x %.3e -> %.3e\n", rho_x, cand);
          rho_x = (cand < st.rho_floor) ? 0.0 : cand;
          last_rho_change_iter = info.iters;
        }
        clean_streak = 0;
      }
    } else if (!did_reuse) {
      clean_streak = 0;
      // esprit bisection : interdit de re-tenter la valeur qui vient
      // d'echouer (rho_entry), mais l'intervalle au-dessus reste explorable.
      rho_decay_floor = std::max(rho_decay_floor, 1.5 * rhox_entry);
      floor_mu = mu;
      last_rho_change_iter = info.iters;
    }
    if (mu < 0.1 * floor_mu) { rho_decay_floor = 0.0; floor_mu = mu; }  // stale floor
    info.rho_final = rho;

    // ---------------- L3: Newton right-hand side ---------------------------
    // rho/atoms may have changed during inertia correction/factorization after
    // the model-residual evaluation above.  Rebuild (or reset) the force on
    // the exact metric used by this Newton system.
    refresh_momentum_force();
    setzero(w.Qx); P.Q.gaxpy_sym(w.x.data(), w.Qx.data());
    setzero(w.Aty); if (p) P.A.gaxpy_t(w.y.data(), w.Aty.data());
    setzero(w.Ctz); if (m) P.C.gaxpy_t(w.z.data(), w.Ctz.data());
    // gradient proximal du modele DCA ancre en x_ref : (rho I + H_k)(x - x_ref)
    std::vector<double> prox_g(n, 0.0);
    { std::vector<double> dref(n);
      for (idx_t i = 0; i < n; ++i) dref[i] = w.x[i] - x_ref[i];
      for (idx_t i = 0; i < n; ++i) prox_g[i] = rho * dref[i];
      atoms.apply(dref.data(), prox_g.data());
      for (idx_t i = 0; i < n; ++i) prox_g[i] -= momentum_force[i]; }
    if (p) { setzero(w.Ax); P.A.gaxpy(w.x.data(), w.Ax.data()); }
    if (m) { setzero(w.Cx); P.C.gaxpy(w.x.data(), w.Cx.data()); }
    std::vector<double> rhs_s(m, 0.0);
    // direction de Newton pour la cible mu_t, avec corrections du 2nd ordre
    // de Mehrotra (corr = produits affines Δslack·Δdual, nulles au predicteur)
    auto newton_dir = [&](double mu_t, bool use_corr) {
      for (idx_t i = 0; i < n; ++i) {
        double r = -(w.Qx[i] + P.q[i] + prox_g[i] - w.Aty[i] - w.Ctz[i]);
        if (w.hl[i]) r += (mu_t - (use_corr ? mcl_[i] : 0.0)) / (w.x[i] - P.l[i]);
        if (w.hu[i]) r -= (mu_t - (use_corr ? mcu_[i] : 0.0)) / (P.u[i] - w.x[i]);
        w.rhs[i] = r;
      }
      for (idx_t i = 0; i < p; ++i) w.rhs[n + i] = -(w.Ax[i] - P.b[i]);
      for (idx_t i = 0; i < m; ++i) {
        double t = -w.z[i];
        if (w.hsl[i]) t += (mu_t - (use_corr ? msl_[i] : 0.0)) / (w.s[i] - P.cl[i]);
        if (w.hsu[i]) t -= (mu_t - (use_corr ? msu_[i] : 0.0)) / (P.cu[i] - w.s[i]);
        rhs_s[i] = t;
        w.rhs[n + p + i] = -(w.Cx[i] - w.s[i]) + t / w.ths[i];
      }
      kkt.solve(w.rhs);
      for (idx_t i = 0; i < n; ++i) w.dx[i] = w.rhs[i];
      for (idx_t i = 0; i < p; ++i) w.dy[i] = -w.rhs[n + i];        // dy = -dŷ
      for (idx_t i = 0; i < m; ++i) w.dz[i] = -w.rhs[n + p + i];    // dz = -dẑ
      for (idx_t i = 0; i < m; ++i) w.ds[i] = (rhs_s[i] - w.dz[i]) / w.ths[i];
      for (idx_t i = 0; i < n; ++i) {
        double sl = w.x[i] - P.l[i], su = P.u[i] - w.x[i];
        w.dvl[i] = w.hl[i] ? ((mu_t - (use_corr ? mcl_[i] : 0.0)) / sl - w.vl[i] - w.vl[i] / sl * w.dx[i]) : 0.0;
        w.dvu[i] = w.hu[i] ? ((mu_t - (use_corr ? mcu_[i] : 0.0)) / su - w.vu[i] + w.vu[i] / su * w.dx[i]) : 0.0;
      }
      for (idx_t i = 0; i < m; ++i) {
        double sl = w.s[i] - P.cl[i], su = P.cu[i] - w.s[i];
        w.dwl[i] = w.hsl[i] ? ((mu_t - (use_corr ? msl_[i] : 0.0)) / sl - w.wl[i] - w.wl[i] / sl * w.ds[i]) : 0.0;
        w.dwu[i] = w.hsu[i] ? ((mu_t - (use_corr ? msu_[i] : 0.0)) / su - w.wu[i] + w.wu[i] / su * w.ds[i]) : 0.0;
      }
    };
    mehrotra_corr_used = false;
    if (st.mehrotra_enable && R.ncomp > 0 && R.gap > 0.0) {
      // ---- predicteur affine (mu=0, meme factorisation) ----
      newton_dir(0.0, false);
      double aaf = 1.0;
      auto ftb = [&](double v2, double dv2) { if (dv2 < 0.0) aaf = std::min(aaf, -0.995 * v2 / dv2); };
      for (idx_t i = 0; i < n; ++i) {
        if (w.hl[i]) { ftb(w.x[i] - P.l[i], w.dx[i]);  ftb(w.vl[i], w.dvl[i]); }
        if (w.hu[i]) { ftb(P.u[i] - w.x[i], -w.dx[i]); ftb(w.vu[i], w.dvu[i]); }
      }
      for (idx_t i = 0; i < m; ++i) {
        if (w.hsl[i]) { ftb(w.s[i] - P.cl[i], w.ds[i]);  ftb(w.wl[i], w.dwl[i]); }
        if (w.hsu[i]) { ftb(P.cu[i] - w.s[i], -w.ds[i]); ftb(w.wu[i], w.dwu[i]); }
      }
      double gap_aff = 0.0;
      for (idx_t i = 0; i < n; ++i) {
        if (w.hl[i]) gap_aff += std::max(0.0, (w.vl[i] + aaf * w.dvl[i]) * (w.x[i] - P.l[i] + aaf * w.dx[i]));
        if (w.hu[i]) gap_aff += std::max(0.0, (w.vu[i] + aaf * w.dvu[i]) * (P.u[i] - w.x[i] - aaf * w.dx[i]));
      }
      for (idx_t i = 0; i < m; ++i) {
        if (w.hsl[i]) gap_aff += std::max(0.0, (w.wl[i] + aaf * w.dwl[i]) * (w.s[i] - P.cl[i] + aaf * w.ds[i]));
        if (w.hsu[i]) gap_aff += std::max(0.0, (w.wu[i] + aaf * w.dwu[i]) * (P.cu[i] - w.s[i] - aaf * w.ds[i]));
      }
      double sig = std::pow(std::min(1.0, gap_aff / R.gap), 3.0);
      sig = std::min(std::max(sig, st.mehrotra_sigma_min), st.mehrotra_sigma_max);
      // Recuit nonconvexe : la descente LENTE de mu explore (les runs gate
      // trouvent -3.1291e9 la ou Mehrotra plein plafonne a -3.1278e9 dans le
      // premier bassin venu). Tant que la courbure du cone critique n'est pas
      // certifiee (lambda_est < -tol), plancher sigma : mu descend doucement,
      // le recuit est preserve ; Mehrotra a pleine vitesse ensuite.
      if (st.mehrotra_anneal && std::isfinite(last_lambda_est) &&
          last_lambda_est < -st.atom_neg_tol)
        sig = std::max(sig, 0.7);
      const double mu_min_g = std::max(1e-13, 0.01 * st.eps_comp);
      double mu_meh = std::max(mu_min_g, sig * R.gap / (double)R.ncomp);
      if (mu_meh < mu) { mu = mu_meh; mu_last_change_iter = info.iters; }
      // ---- corrections du 2nd ordre : Δslack_aff · Δdual_aff ----
      for (idx_t i = 0; i < n; ++i) {
        mcl_[i] = w.hl[i] ? (w.dx[i] * w.dvl[i]) : 0.0;      // Δsl = +dx
        mcu_[i] = w.hu[i] ? (-w.dx[i] * w.dvu[i]) : 0.0;     // Δsu = -dx
      }
      for (idx_t i = 0; i < m; ++i) {
        msl_[i] = w.hsl[i] ? (w.ds[i] * w.dwl[i]) : 0.0;
        msu_[i] = w.hsu[i] ? (-w.ds[i] * w.dwu[i]) : 0.0;
      }
      newton_dir(mu, true); mehrotra_corr_used = true;
    } else {
      newton_dir(mu, false);
    }

    // ---------------- primal step: ratio test + Armijo on exact-l1 merit ---
    double amax = 1.0;
    auto ratio = [&](double v, double dv, double bnd, int side) {
      // side=+1: v + a*dv >= bnd ; side=-1: v + a*dv <= bnd
      double g = side > 0 ? (v - bnd) : (bnd - v);
      double d = side > 0 ? -dv : dv;
      if (d > 0) amax = std::min(amax, st.ftb_delta * g / d);
    };
    for (idx_t i = 0; i < n; ++i) {
      if (w.hl[i]) ratio(w.x[i], w.dx[i], P.l[i], +1);
      if (w.hu[i]) ratio(w.x[i], w.dx[i], P.u[i], -1);
    }
    for (idx_t i = 0; i < m; ++i) {
      if (w.hsl[i]) ratio(w.s[i], w.ds[i], P.cl[i], +1);
      if (w.hsu[i]) ratio(w.s[i], w.ds[i], P.cu[i], -1);
    }
    // penalty update (paper §3.2.1): sigma >= ||z~||_inf + sigma_bar --
    // PLAFONNE : sur dual degenere (LICQ deficient, cf. NCVXQP7), ||z~||_inf
    // peut exploser (1e8+), forcant sigma a le suivre jusqu'au blow-up total
    // (>1e14 -> arret sec). Au-dela du plafond, l'exactitude theorique de la
    // penalite L1 n'est plus garantie -- mais c'est deja le regime ou le
    // primal est convergeant (feas machine) : mieux vaut continuer a iterer
    // avec un sigma borne que de s'arreter net sur l'explosion.
    double zt_inf = 0.0;
    for (idx_t i = 0; i < p; ++i) zt_inf = std::max(zt_inf, std::fabs(w.y[i] + w.dy[i]));
    for (idx_t i = 0; i < m; ++i) zt_inf = std::max(zt_inf, std::fabs(w.z[i] + w.dz[i]));
    if (sigma < zt_inf + st.sigma_bar)
      sigma = std::min(std::max(st.sigma_tau * sigma, zt_inf + st.sigma_bar), st.sigma_max);

    // directional derivative of the merit (paper §3.2, convexified G)
    double dGd = 0.0;
    { setzero(w.Qx); P.Q.gaxpy_sym(w.dx.data(), w.Qx.data());
      dGd = dot(w.dx, w.Qx) + (rho + rho_x) * dot(w.dx, w.dx) + atoms.quad(w.dx.data());
      for (idx_t i = 0; i < n; ++i) dGd += w.thx[i] * w.dx[i] * w.dx[i];
      for (idx_t i = 0; i < m; ++i) dGd += w.ths[i] * w.ds[i] * w.ds[i]; }
    double pen_now = 0.0, ztr = 0.0;
    if (p) { setzero(w.Ax); P.A.gaxpy(w.x.data(), w.Ax.data());
      for (idx_t i = 0; i < p; ++i) { double r = w.Ax[i] - P.b[i]; pen_now += std::fabs(r); ztr += (w.y[i] + w.dy[i]) * r; } }
    if (m) { setzero(w.Cx); P.C.gaxpy(w.x.data(), w.Cx.data());
      for (idx_t i = 0; i < m; ++i) { double r = w.Cx[i] - w.s[i]; pen_now += std::fabs(r); ztr += (w.z[i] + w.dz[i]) * r; } }
    double mprime = -dGd - ztr - sigma * pen_now;
    if (mprime > -1e-16) mprime = -1e-16;

    // v19av-convdesc: certified strict convexity of the DCA model on ker(A)
    // => drop the sufficient-descent margin to 0 (accept any genuine descent).
    double armijo_c = st.armijo_rho;
    if (st.accept_convex_descent) {
      if (convdesc_lam_rho != rho || !std::isfinite(convdesc_lam)) {
        if (std::isfinite(info.lambda_min_reduced)) convdesc_lam = info.lambda_min_reduced;
        else if (proj && proj->ok() && !feas_dominated) {
          info.n_lanczos_calls++;
          convdesc_lam = projected_lanczos_lmin(P.Q, proj.get(), st.lanczos_maxit, st.lanczos_tol,
                                                42, atoms.size() ? &atoms : nullptr).lmin;
          info.lambda_min_reduced = convdesc_lam;
        } else convdesc_lam = P.Q.gershgorin_lmin();
        convdesc_lam_rho = rho;
      }
      const double safety = std::max(1.0, st.convex_descent_safety);
      if (std::isfinite(convdesc_lam) && rho + safety * convdesc_lam > 0.0) {
        armijo_c = 0.0;
        if (st.verbose >= 2)
          std::printf("[LazyDCA-CONVDESC] relaxed required->0 (rho=%.6e lam_red=%.6e safety=%.3e margin=%.6e)\n",
                      rho, convdesc_lam, safety, rho + safety * convdesc_lam);
      }
    }
    bool interior = false;
    double m0 = merit(P, w, w.x, w.s, mu, sigma, rho, interior, &atoms, x_ref, &momentum_force);
    double alpha = amax, aP = 0.0;
    last_ls = 0;
    for (idx_t ls = 0; ls < st.armijo_max; ++ls) {
      last_ls = ls + 1;
      for (idx_t i = 0; i < n; ++i) w.xt[i] = w.x[i] + alpha * w.dx[i];
      for (idx_t i = 0; i < m; ++i) w.st[i] = w.s[i] + alpha * w.ds[i];
      double mt = merit(P, w, w.xt, w.st, mu, sigma, rho, interior, &atoms, x_ref, &momentum_force);
      if (interior && mt <= m0 + armijo_c * alpha * mprime) { aP = alpha; break; }
      alpha *= st.armijo_beta;
    }
    if (aP <= std::min(amax, 1e-5) && mehrotra_corr_used) {
      // le correcteur de Mehrotra n'est pas garanti direction de descente du
      // merite (les implementations standard n'ont pas d'Armijo) ; pres des
      // bornes actives a mu plancher, ses termes du 2nd ordre dominent et
      // Armijo rejette tout (cycle bit-identique observe sur NCVXQP3).
      // Repli : direction Newton pure, meme factorisation, re-tenter.
      newton_dir(mu, false); mehrotra_corr_used = false;
      double am2 = 1.0;
      for (idx_t i = 0; i < n; ++i) {
        if (w.hl[i] && w.dx[i] < 0.0) am2 = std::min(am2, -0.995 * (w.x[i] - P.l[i]) / w.dx[i]);
        if (w.hu[i] && w.dx[i] > 0.0) am2 = std::min(am2,  0.995 * (P.u[i] - w.x[i]) / w.dx[i]);
      }
      for (idx_t i = 0; i < m; ++i) {
        if (w.hsl[i] && w.ds[i] < 0.0) am2 = std::min(am2, -0.995 * (w.s[i] - P.cl[i]) / w.ds[i]);
        if (w.hsu[i] && w.ds[i] > 0.0) am2 = std::min(am2,  0.995 * (P.cu[i] - w.s[i]) / w.ds[i]);
      }
      double al2 = am2;
      for (idx_t ls2 = 0; ls2 < st.armijo_max; ++ls2) {
        bool int2 = false;
        for (idx_t i = 0; i < n; ++i) w.xt[i] = w.x[i] + al2 * w.dx[i];
        for (idx_t i = 0; i < m; ++i) w.st[i] = w.s[i] + al2 * w.ds[i];
        double mt2 = merit(P, w, w.xt, w.st, mu, sigma, rho, int2, &atoms, x_ref, &momentum_force);
        if (int2 && mt2 <= m0) { aP = al2; break; }   // non-augmentation (repli)
        al2 *= st.armijo_beta;
      }
      if (st.verbose >= 2 && aP > 1e-5)
        std::printf("[LazyDCA-mehrotra] repli direction pure : alpha %.1e\n", aP);
    }
    if (aP == 0.0) aP = std::min(amax, 1e-8);  // safeguarded minimal step
    if (aP <= 1e-6) {
      ++armijo_fail_streak;
      if (armijo_fail_streak == 2) {
        // 1) erreur de stationnarite souvent purement duale : LS refresh de y
        ls_multiplier_refresh();
        // completion des duals de bornes : vl/vu portent le gradient sur les
        // bornes quasi-actives (m=0 : TOUT le residu dual vit ici).
        // UNIQUEMENT en proximite terminale : en pleine traversee, g ~ 1e5 et
        // la completion fait exploser les produits (compl rel 514 observee),
        // cascade gap -> mu -> Theta -> echec de factorisation.
        if (R.feas_rel <= 10.0 * st.eps_feas && R.comp_rel <= 1e3 * st.eps_comp)
        { std::vector<double> g2(n, 0.0);
          setzero(w.Qx); P.Q.gaxpy_sym(w.x.data(), w.Qx.data());
          setzero(w.Aty); if (p) P.A.gaxpy_t(w.y.data(), w.Aty.data());
          setzero(w.Ctz); if (m) P.C.gaxpy_t(w.z.data(), w.Ctz.data());
          for (idx_t i = 0; i < n; ++i) {
            double gi = w.Qx[i] + P.q[i] + rho * (w.x[i] - x_ref[i])
                        - momentum_force[i] - w.Aty[i] - w.Ctz[i];
            g2[i] = gi;
          }
          atoms.apply(w.x.data(), g2.data());   // approx : H x ; correction fine par x_ref
          std::vector<double> hr(n, 0.0); atoms.apply(x_ref.data(), hr.data());
          for (idx_t i = 0; i < n; ++i) {
            double gi = g2[i] - hr[i];
            double sl = w.hl[i] ? (w.x[i] - P.l[i]) : 1e300;
            double su = w.hu[i] ? (P.u[i] - w.x[i]) : 1e300;
            if (w.hl[i] && sl <= su && gi > 0.0) { w.vl[i] = gi + mu / sl; }
            else if (w.hu[i] && gi < 0.0) { w.vu[i] = -gi + mu / su; }
          }
        }
      } else if (armijo_fail_streak == 3 && event_budget_ok()) {
        // [C] le merite predit une descente que f refuse : mesurer la
        // courbure du modele le long de la direction courante (UNE
        // application de matrice). Si d'(Q+H)d < 0, le modele ment le long
        // de d — et d elle-meme est le candidat-atome.
        double dn2 = 0.0; for (idx_t i = 0; i < n; ++i) dn2 += w.dx[i] * w.dx[i];
        if (dn2 > 1e-300) {
          std::vector<double> Hd(n, 0.0);
          P.Q.gaxpy_sym(w.dx.data(), Hd.data());
          atoms.apply(w.dx.data(), Hd.data());
          double dQd = 0.0; for (idx_t i = 0; i < n; ++i) dQd += w.dx[i] * Hd[i];
          const double r_d = dQd / dn2;
          if (r_d < -st.atom_neg_tol) {
            std::vector<double> zc2 = w.dx;
            for (idx_t i = 0; i < n; ++i)
              if ((w.hl[i] && std::fabs(x_ref[i] - P.l[i]) <= st.atom_bound_active_tol) ||
                  (w.hu[i] && std::fabs(P.u[i] - x_ref[i]) <= st.atom_bound_active_tol))
                zc2[i] = 0.0;
            std::vector<idx_t> ui2; std::vector<double> uv2; double e2 = 0.0;
            const bool evt_sparse_ok = st.atom_energy_sparsify
                ? AtomSet::sparsify_energy(zc2, st.atom_max_nnz, st.atom_energy_target,
                                           st.atom_min_nnz, ui2, uv2, &e2)
                : AtomSet::sparsify(zc2, st.atom_max_nnz, ui2, uv2);
            if (evt_sparse_ok && !evt_pending_probe &&
                (!st.adaptive_marginal_return_enable ||
                 info.iters >= last_event_atom_iter +
                     (idx_t)adaptive_return.event_spacing(
                         (std::size_t)std::max<idx_t>(1, st.atom_event_every)))) {
              evt_ui = std::move(ui2); evt_uv = std::move(uv2);
              evt_beta = -r_d + st.atom_margin;
              // Under the adaptive oracle, event C is evidence that an atom
              // may be useful, not permission to bypass candidate selection.
              // Re-run the pooled oracle at the next accepted anchor instead
              // of inserting the raw Newton direction directly.
              evt_has_direction = !st.atom_oracle_enable;
              evt_pending_probe = true; evt_pending_tag = 'C'; evt_pending_since = info.iters;
              last_event_atom_iter = info.iters;
              if (st.verbose >= 1)
                std::printf("[LazyDCA-atomEvt C] direction menteuse (r=%.3e) : atome en attente d'accept\n", r_d);
            }
          }
        }
      } else if (armijo_fail_streak == 4) {
        // 2) pousser mu (une fois, plancher lie a la precision)
        mu = std::max(std::max(1e-13, 0.1 * st.eps_comp), st.barrier_r * mu); mu_inner = 0;
        if (st.verbose >= 2) std::printf("[LazyDCA-stall] echec Armijo repete -> mu force a %.2e\n", mu);
      } else if (armijo_fail_streak >= 6) {
        // 3) amortissement proximal SIGNIFICATIF (echelle de la courbure),
        //    protege du decay terminal pendant sa prise d'effet
        double rx_old = rho_x;
        double jump = 1e-4 * (rho_cap > 0 ? rho_cap : (1.0 + P.Q.U.inf_norm()));
        rho_x = std::min(std::max({rho_x * st.rho_growth, st.rho_floor, jump}), st.rho_abs_max - rho);
        term_floor = std::max(term_floor, 0.5 * rho_x);
        armijo_fail_streak = 0;
        if (st.verbose >= 2) std::printf("[LazyDCA-stall] echec persistant -> amortissement rho_x %.2e => %.2e (modele intact)\n", rx_old, rho_x);
      }
    } else armijo_fail_streak = 0;

    // ---------------- dual step: Yamashita-type box rule --------------------
    // keep every complementarity product inside [beta_l*mu, beta_u*mu] when
    // possible, never exceed fraction-to-boundary on duals; common step.
    double aD = 1.0;
    auto dual_ratio = [&](double dual, double dd) {
      if (dd < 0) aD = std::min(aD, st.ftb_delta * dual / (-dd));
    };
    for (idx_t i = 0; i < n; ++i) { if (w.hl[i]) dual_ratio(w.vl[i], w.dvl[i]); if (w.hu[i]) dual_ratio(w.vu[i], w.dvu[i]); }
    for (idx_t i = 0; i < m; ++i) { if (w.hsl[i]) dual_ratio(w.wl[i], w.dwl[i]); if (w.hsu[i]) dual_ratio(w.wu[i], w.dwu[i]); }
    const double beta_u = 1e10;  // upper box (keeps D bounded)
    auto dual_box = [&](double slack_new, double dual, double dd) {
      if (dd > 0) { double lim = (beta_u * std::max(mu, 1e-12) / std::max(slack_new, 1e-300) - dual) / dd; if (lim > 0) aD = std::min(aD, std::max(lim, 1e-4)); }
    };
    for (idx_t i = 0; i < n; ++i) {
      if (w.hl[i]) dual_box(w.x[i] + aP * w.dx[i] - P.l[i], w.vl[i], w.dvl[i]);
      if (w.hu[i]) dual_box(P.u[i] - w.x[i] - aP * w.dx[i], w.vu[i], w.dvu[i]);
    }
    for (idx_t i = 0; i < m; ++i) {
      if (w.hsl[i]) dual_box(w.s[i] + aP * w.ds[i] - P.cl[i], w.wl[i], w.dwl[i]);
      if (w.hsu[i]) dual_box(P.cu[i] - w.s[i] - aP * w.ds[i], w.wu[i], w.dwu[i]);
    }

    // ---------------- update iterates ---------------------------------------
    // ================= acceptation zip : descente suffisante + faisabilite ===
    // evaluee en COORDONNEES ORIGINALES au nouveau point (19g : les deux
    // ensemble, point de passage unique) ; la reference DCA x_ref ne bouge
    // que sur acceptation (mode integre).
    {
      Residuals Ra = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
      const double cv = std::max(Ra.eq_abs, Ra.ineq_abs);
      const bool feasible = cv <= st.sd_cv_abs ||
          (st.sd_cv_rel > 0 && Ra.feas_rel <= st.sd_cv_rel && cv <= st.sd_rel_feas_abs_cap);
      const double f_orig = objective(P_in, w);
      // Acceptance reference is an independent A/B axis.  Monotone keeps the
      // historical semantics byte-for-byte; window replaces the last-anchor
      // reference by the maximum over the last W accepted objectives.
      double ref_for_accept = ref_obj;
      if (st.dca_acceptance == DcaAcceptanceMode::Monotone) {
        if (st.sd_accept_against_best_feasible && std::isfinite(best_feas_obj))
          ref_for_accept = std::min(ref_for_accept, best_feas_obj);
      } else if (st.dca_acceptance == DcaAcceptanceMode::Window && !dca_accept_obj_hist.empty()) {
        ref_for_accept = -INF;
        const idx_t W = std::max<idx_t>(1, st.dca_accept_window);
        idx_t skip = std::max<idx_t>(0, (idx_t)dca_accept_obj_hist.size() - W);
        idx_t q = 0;
        for (double fv : dca_accept_obj_hist) { if (q++ >= skip) ref_for_accept = std::max(ref_for_accept, fv); }
        if (!std::isfinite(ref_for_accept)) ref_for_accept = ref_obj;
      }
      std::vector<double> accept_step(n);
      double step2 = 0.0;
      for (idx_t i = 0; i < n; ++i) {
        accept_step[i] = w.x[i] - x_ref[i];
        step2 += accept_step[i] * accept_step[i];
      }
      double required = std::max(st.sd_abs, st.sd_rel * std::max(1.0, std::fabs(ref_for_accept)))
                        + st.sd_sigma * step2;
      // convdesc (19av) : modele certifie strictement convexe => required = 0
      if (st.accept_convex_descent && std::isfinite(convdesc_lam) &&
          convdesc_lam_rho == rho && rho + std::max(1.0, st.convex_descent_safety) * convdesc_lam > 0.0)
        required = 0.0;
      const double descent = ref_for_accept - f_orig;
      const bool first_anchor = (accepts == 0);
      // zip 19al (fixpoint) : pas DCA <= tol en un point faisable = point fixe
      // de l'application DCA = point KKT (papier, Th. 1(ii)) -> accepter pour
      // annuler le gradient proximal residuel et laisser la stationnarite
      // ORIGINALE converger.
      // tolerance = bruit flottant de f (zip : descente strictement positive ;
      // en float64, |f| ~ 1e9 rend descent > 0 strict non evaluable)
      const double f_noise = 8.0 * std::numeric_limits<double>::epsilon() *
                             std::max(1.0, std::fabs(ref_for_accept));
      const bool fixpoint = std::sqrt(step2) <= 1e-8 * (1.0 + norm_inf(x_ref)) &&
                            descent >= -f_noise;
      // zip soft-accept (solved-stall) : le modele est resolu (stationnarite
      // du modele au niveau cible), le point est faisable et l'amelioration
      // originale est authentiquement positive -> accepter sous le seuil.
      // fixpoint du zip : au modele resolu, |f - ref| <= tol vaut point fixe
      // -> re-ancrer (prox_g s'annule, la stationnarite ORIGINALE converge).
      // GARANTIE DCA : tout point SOLUTION du sous-probleme convexe est
      // accepte des lors qu'il est faisable et non-ascendant (au bruit
      // flottant pres) — la majorisation assure f(x+) <= f(x_ref), la marge
      // de decroissance suffisante ne filtre que les pas a modele NON
      // resolu (son terme sigma*|pas|^2 rejetait des pas francs legitimes).
      const bool model_solved = stat_model_rel_iter <=
          std::max(10.0 * st.eps_stat, st.model_solved_tol);
      const bool soft_accept = descent > -f_noise && model_solved;
      if (soft_accept && st.verbose >= 2 && descent < 0.0)
        std::printf("[LazyDCA-accept] sous-probleme resolu (stat_model %.2e) : accept au point fixe (descent %.3e)\n",
                    stat_model_rel_iter, descent);
      if (dca_inner == 0 || !std::isfinite(dca_model_start))
        dca_model_start = std::max(stat_model_rel_iter, 1e-16);
      dca_inner++;
      const double contracted_gate = std::max(st.dca_model_abs,
          st.dca_model_reduction * dca_model_start);
      const bool model_ready = dca_inner >= st.dca_inner_min &&
                               stat_model_rel_iter <= contracted_gate;
      const bool terminal_entry = st.terminal_polish && !terminal_polish_active &&
          info.iters >= polish_cooldown_until &&
          Ra.feas_rel <= st.eps_feas && Ra.comp_rel <= 10.0 * st.eps_comp &&
          Ra.stat_rel <= st.terminal_anchor_stat_trigger;
      if (terminal_entry) {
        terminal_polish_active = true; rho_pending = -1.0;
        reset_momentum_epoch("entree terminal polish");
        x_ref = w.x; dca_inner = 0; dca_model_start = std::numeric_limits<double>::infinity();
        fact_mu = -1.0; reuse_streak = 0;
        if (Ra.comp_rel <= st.eps_comp) {
          terminal_best_work = w; terminal_best_stat = Ra.stat_rel; terminal_best_valid = true;
        }
        terminal_reanchors++;
        if (st.verbose >= 2)
          std::printf("[LazyDCA-polish] entree+re-ancrage : stat=%.3e (gel mu/rho/atomes)\n", Ra.stat_rel);
      }
      const double terminal_comp_degrade_limit = terminal_recovery_active
          ? std::max(100.0 * st.eps_comp, terminal_recovery_comp_trigger)
          : 100.0 * st.eps_comp;
      if (terminal_polish_active &&
          (Ra.feas_rel > 100.0 * st.eps_feas || Ra.comp_rel > terminal_comp_degrade_limit)) {
        terminal_polish_active = false;
        polish_cooldown_until = info.iters + 20;
        if (st.verbose >= 2) std::printf("[LazyDCA-polish] sortie : degradation (cooldown 20)\n");
        if (terminal_recovery_active && terminal_recovery_entry_valid) {
          w = terminal_recovery_entry_work;
          mu = terminal_recovery_entry_mu;
          rho = terminal_recovery_entry_rho;
          rho_x = terminal_recovery_entry_rho_x;
          rho_pending = terminal_recovery_entry_rho_pending;
          rho_cut = terminal_recovery_entry_rho_cut;
          fact_mu = -1.0; reuse_streak = 0;
          terminal_recovery_active = false;
          dca_converged_flag = true;
          info.status = Status::DCAConverged;
          info.message = "bounded terminal recovery aborted after residual degradation; pre-recovery iterate restored";
          if (st.verbose >= 1)
            std::printf("[LazyDCA-terminal-v262] recovery degradation: restored entry iterate\n");
          goto finish;
        }
      }
      const bool terminal_anchor = terminal_polish_active && feasible &&
          Ra.comp_rel <= 10.0 * st.eps_comp && Ra.stat_rel <= st.terminal_anchor_stat_trigger;
      if (terminal_polish_active && feasible && Ra.comp_rel <= st.eps_comp &&
          (!terminal_best_valid || Ra.stat_rel < terminal_best_stat)) {
        terminal_best_work = w; terminal_best_stat = Ra.stat_rel; terminal_best_valid = true;
      }
      // Acceptance policy.  Feasibility remains mandatory in all modes.
      // The terminal-polish gate is intentionally shared by all policies.
      const bool dca_monotone_ok = (accepts == 0) ||
          (f_orig <= ref_obj + st.dca_monotone_tol_abs +
                     st.dca_monotone_tol_rel * std::max(1.0, std::fabs(ref_obj)));
      const bool descent_gate = feasible &&
          (first_anchor || fixpoint || soft_accept || (descent >= required && model_ready));

      bool policy_accept = false;
      double lyap_candidate = INF, lyap_reference = INF, lyap_step_m2 = 0.0;
      if (st.dca_acceptance == DcaAcceptanceMode::Monotone) {
        policy_accept = dca_monotone_ok && descent_gate;
      } else if (st.dca_acceptance == DcaAcceptanceMode::Window) {
        // Non-monotone Grippo-Lampariello-Lucidi style reference: descent is
        // required relative to the maximum of the recent accepted values,
        // not necessarily relative to the immediately preceding anchor.
        policy_accept = descent_gate;
      } else {
        const bool inertial_energy_active = st.dca_momentum != DcaMomentumMode::Original &&
            momentum_have_step && !momentum_suppressed_epoch;
        const double theta_bar = inertial_energy_active ? std::max(0.0,
            st.dca_lyapunov_theta_bar > 0.0 ? st.dca_lyapunov_theta_bar : st.dca_momentum_theta) : 0.0;
        lyap_step_m2 = dca_metric_quad(accept_step, rho, atoms);
        const double prev_m2 = momentum_have_step ? dca_metric_quad(momentum_v, rho, atoms) : 0.0;
        lyap_reference = ref_obj + 0.5 * theta_bar * prev_m2;
        lyap_candidate = f_orig + 0.5 * theta_bar * lyap_step_m2;
        const double lyap_noise = 8.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::fabs(lyap_reference), std::fabs(lyap_candidate)});
        const bool lyap_ok = lyap_candidate <= lyap_reference -
            std::max(0.0, st.dca_lyapunov_eta) * lyap_step_m2 + lyap_noise;
        policy_accept = feasible && (first_anchor ||
            (lyap_ok && (fixpoint || model_solved || model_ready)));
      }

      const bool final_accept = terminal_polish_active
          ? (terminal_anchor && (st.dca_acceptance != DcaAcceptanceMode::Monotone || dca_monotone_ok))
          : policy_accept;
      if (final_accept) {
        // [D] sur-convexification : part de la penalite beta*(a'Delta)^2 de
        // chaque atome dans la descente de cet accept. Un atome qui FREINE
        // systematiquement (penalite > moitie de la descente, 5 accepts) est
        // candidat au beta/2 — certifie par essai (beta ne touche que les
        // VALEURS de la colonne KKT, pas le pattern : cout = 1 factorisation).
        if ((st.atom_event_enable || st.atom_oracle_retune_enable) && atoms.size() > 0) {
          if ((idx_t)atom_brake.size() < atoms.size()) atom_brake.resize(atoms.size(), 0);
          const double df = std::max(ref_obj - f_orig, 1e-12 * (1.0 + std::fabs(f_orig)));
          for (idx_t j2 = 0; j2 < (idx_t)atoms.size(); ++j2) {
            double sdot = 0.0;
            for (size_t k2 = 0; k2 < atoms.idx[j2].size(); ++k2)
              sdot += atoms.val[j2][k2] * (w.x[atoms.idx[j2][k2]] - x_ref[atoms.idx[j2][k2]]);
            const double pen = atoms.beta[j2] * sdot * sdot;
            if (pen > st.atom_oracle_brake_fraction * df) atom_brake[j2]++;
            else if (atom_brake[j2] > 0) atom_brake[j2]--;
            const bool retune_budget_ok = st.atom_event_enable ? event_budget_ok()
                : (st.atom_oracle_retune_enable && !atom_growth_frozen && !terminal_polish_active &&
                   info.iters >= last_event_atom_iter + st.atom_event_every);
            if (atom_brake[j2] >= st.atom_oracle_brake_patience && retune_budget_ok) {
              const double b_old = atoms.beta[j2];
              atoms.beta[j2] *= 0.5;
              kkt.set_atoms(atoms);
              KKT::Fact frc = kkt.factorize(w.thx, w.ths, rho, delta, info);
              if (frc == KKT::Fact::Budget) {
                // KKT_FACTORIZATION_BUDGET_V4G: the retune was only a probe; restore committed beta.
                atoms.beta[j2] = b_old;
                kkt.set_atoms(atoms);
                goto finish;
              }
              if (frc == KKT::Fact::Ok) {
                last_event_atom_iter = info.iters; atom_brake[j2] = 0;
                info.n_atom_retunes++;
                fact_mu = -1.0; reuse_streak = 0; atoms_dirty = false;
                if (st.verbose >= 1)
                  std::printf("[LazyDCA-atomEvt D] atome #%lld sur-convexifiant : beta %.3e -> %.3e (certifie)\n",
                              (long long)(j2 + 1), b_old, atoms.beta[j2]);
              } else {
                atoms.beta[j2] = b_old; kkt.set_atoms(atoms);
                atom_brake[j2] = 0;   // le spectre a besoin de ce beta : acquitte
              }
            }
          }
        }
        const double dca_prev_f = ref_obj;   // objectif ACCEPTE precedent (avant ecrasement)
        // Update inertial memory from ACCEPTED anchors only.  During terminal
        // polishing inertia is quenched so that the final KKT/SOC certificate
        // is formed on the original, non-inertial problem.
        if (st.dca_momentum != DcaMomentumMode::Original && !terminal_polish_active) {
          sync_complete_metric();
          momentum_v = accept_step;
          momentum_have_step = std::sqrt(step2) > 1e-14 * (1.0 + norm_inf(x_ref));
          momentum_suppressed_epoch = false;
          if (st.dca_momentum_complete && momentum_have_step) {
            const double cap=complete_theta_cap(st), inc=std::max(0.0,st.dca_momentum_theta_inc);
            if (st.dca_momentum==DcaMomentumMode::InertialAtomic) {
              momentum_theta_atom_complete.resize(atoms.beta.size(),0.0);
              const auto aw=complete_atomic_weights(st.dca_momentum_atom_weighting,momentum_v,atoms,rho,st.dca_momentum_energy_weight_rho);
              momentum_theta_rho_complete=std::min(cap,momentum_theta_rho_complete+inc*aw.rho_weight);
              for(size_t j=0;j<momentum_theta_atom_complete.size();++j){
                if(j<momentum_reset_atom_complete.size() && momentum_reset_atom_complete[j]){ momentum_theta_atom_complete[j]=0.0; continue; }
                const double wj=j<aw.atom.size()?aw.atom[j]:0.0;
                if(wj<st.dca_momentum_align_threshold&&st.dca_momentum_align_decay>=0.0&&st.dca_momentum_align_decay<1.0){momentum_theta_atom_complete[j]*=st.dca_momentum_align_decay;info.n_theta_align_decays++;}
                else momentum_theta_atom_complete[j]=std::min(cap,momentum_theta_atom_complete[j]+inc*wj);
              }
              info.momentum_align_mean=aw.align_mean;info.momentum_align_max=std::max(info.momentum_align_max,aw.align_max);
              info.momentum_energy_rho_share=aw.energy_rho_share;info.momentum_energy_atom_max_share=std::max(info.momentum_energy_atom_max_share,aw.energy_atom_max_share);
              info.momentum_energy_atom_effective_count=std::max(info.momentum_energy_atom_effective_count,aw.energy_atom_effective_count);
              std::fill(momentum_reset_atom_complete.begin(), momentum_reset_atom_complete.end(), 0);
            } else momentum_theta_global_complete=std::min(cap,momentum_theta_global_complete+inc);
            momentum_theta_cur=complete_headline_theta();info.n_momentum_complete_accepts++;
          } else {
            const double base_theta=std::clamp(st.dca_momentum_theta,std::max(0.0,st.dca_momentum_theta_min),std::max(std::max(0.0,st.dca_momentum_theta_min),st.dca_momentum_theta_max));
            if(st.dca_momentum_growth>1.0&&momentum_theta_cur<base_theta)momentum_theta_cur=std::min(base_theta,momentum_theta_cur*st.dca_momentum_growth);
          }
        } else if (terminal_polish_active) {
          std::fill(momentum_v.begin(), momentum_v.end(), 0.0);
          std::fill(momentum_force.begin(), momentum_force.end(), 0.0);
          momentum_have_step = false; momentum_suppressed_epoch = true;
          if(st.dca_momentum_complete) full_reset_complete();
        }
        x_ref = w.x; ref_obj = f_orig;
        dca_accept_obj_hist.push_back(f_orig);
        const idx_t hist_cap = std::max<idx_t>(2, std::max(st.dca_accept_window, st.dca_smooth_window) + 2);
        while ((idx_t)dca_accept_obj_hist.size() > hist_cap) dca_accept_obj_hist.pop_front();
        if (st.dca_acceptance == DcaAcceptanceMode::Lyapunov)
          info.lyapunov_energy_final = std::isfinite(lyap_candidate) ? lyap_candidate : f_orig;
        // insertion des evenements EN ATTENTE : x vient de devenir x_ref, le
        // terme du nouvel atome est nul au point courant — sans couture.
        if (evt_pending_probe && !atom_growth_frozen && atoms.size() < max_atoms_eff) {
          idx_t added_e = 0;
          if (evt_has_direction) {
            atoms.idx.push_back(std::move(evt_ui)); atoms.val.push_back(std::move(evt_uv));
            atoms.beta.push_back(evt_beta); added_e = 1;
            // ADAPTIVE_MARGINAL_RETURN_V21: direct emergency commit has no oracle
            // lift score; count it and treat it as strong reactivation evidence.
            if (st.adaptive_marginal_return_enable) {
              const auto amr21_direct = adaptive_return.observe_unscored(
                  (std::size_t)atoms.size(), AdaptiveReturnSource::EventDirect, true);
              (void)amr21_direct;
              sync_adaptive_return_info();
            }
            atoms_dirty = true; info.n_atoms = atoms.size(); info.atoms_nnz = atoms.nnz();
            rho_cap = -1.0; hint_valid = false;
            dca_inner = 0; dca_model_start = std::numeric_limits<double>::infinity();
          } else added_e = try_add_atoms(1);
          if (added_e > 0 && st.verbose >= 1)
            std::printf("[LazyDCA-atomEvt %c] atome #%lld insere a l'accept\n",
                        evt_pending_tag, (long long)atoms.size());
          evt_pending_probe = false; evt_has_direction = false;
        }
        if (f_orig <= best_feas_obj) {
          std::copy(w.x.begin(), w.x.end(), bx_.begin());
          std::copy(w.s.begin(), w.s.end(), bs_.begin());
          std::copy(w.y.begin(), w.y.end(), by_.begin());
          std::copy(w.z.begin(), w.z.end(), bz_.begin());
          std::copy(w.vl.begin(), w.vl.end(), bvl_.begin());
          std::copy(w.vu.begin(), w.vu.end(), bvu_.begin());
          std::copy(w.wl.begin(), w.wl.end(), bwl_.begin());
          std::copy(w.wu.begin(), w.wu.end(), bwu_.begin());
          best_feas_valid = true;
        }
        dca_inner = 0; dca_model_start = std::numeric_limits<double>::infinity();
        if (terminal_anchor) terminal_reanchors++;
        best_feas_obj = std::min(best_feas_obj, f_orig);
        accepts++; info.n_accepts = accepts; iters_since_last_accept = 0;
        info.ref_obj = ref_obj; info.best_feasible_obj = best_feas_obj;
        // ---- Affichage DCA (defaut) : SEULEMENT les objectifs des pas
        // ACCEPTES — la sequence que la theorie DCA garantit decroissante et
        // convergente, contrairement au tableau IPM complet (verbose>=3) qui
        // montre chaque essai de Newton y compris les rejets.
        if (st.verbose >= 1 && st.verbose < 3) {
          if (dca_banner_rows % 30 == 0)
            std::printf("\n%6s  %16s  %11s  %10s  %6s  %5s  %6s\n",
                        "accept", "objectif", "delta_f", "feas_pr", "lg(rg)", "atm", "temps");
          dca_banner_rows++;
          const double dca_delta_disp = dca_prev_f - f_orig;
          char dca_rhcol[8];
          if (rho > 0.0) std::snprintf(dca_rhcol, sizeof(dca_rhcol), "%6.1f", std::log10(rho));
          else std::snprintf(dca_rhcol, sizeof(dca_rhcol), "   -  ");
          std::printf("%6lld  %+.9e  %11.3e  %10.3e  %s  %5lld  %5.1fs\n",
                      (long long)accepts, f_orig, dca_delta_disp, cv, dca_rhcol,
                      (long long)atoms.size(), elapsed());
        }
        // ---- Test d'arret DCA : sequence des objectifs ACCEPTES devenue
        // stationnaire sur dca_stall_iters accepts consecutifs. Independant
        // du test KKT/dual — capte les cas ou le primal est deja converge
        // mais la certification duale echoue (dual degenere, ex. NCVXQP7).
        if (st.dca_stop_enable && accepts >= 2 && !terminal_recovery_active) {
          // Robustesse aux pics isoles (meme patron que le test IPOPT
          // "acceptable" lisse, 0.15.17) : un unique gros ecart de bruit
          // (observe : delta=0.645 alors que la sequence est plate depuis 20
          // accepts) remettait le streak a zero et empechait TOUT arret,
          // malgre une sequence f(x_k) visuellement stationnaire. On compare
          // desormais le MAX glissant sur une petite fenetre au seuil,
          // plutot que le delta instantane.
          const double dca_delta = std::fabs(dca_prev_f - f_orig);
          const double dca_tol = st.dca_stall_tol_abs +
                                 st.dca_stall_tol_rel * std::max(1.0, std::fabs(f_orig));
          dca_delta_hist.push_back(dca_delta);
          if ((idx_t)dca_delta_hist.size() > st.dca_smooth_window) dca_delta_hist.pop_front();
          double dca_delta_smoothed = 0.0;
          for (double d : dca_delta_hist) dca_delta_smoothed = std::max(dca_delta_smoothed, d);
          if (dca_delta_smoothed <= dca_tol) {
            if (++dca_stall_streak >= st.dca_stall_iters) {
              info.status = Status::DCAConverged;
              info.message = "sequence des objectifs DCA acceptes stationnaire ("
                             + std::to_string((long long)st.dca_stall_iters) + " accepts consecutifs, lisse)";
              dca_converged_flag = true;
              if (st.verbose >= 1)
                std::printf("[LazyDCA-DCAstop] sequence f(x_k) stationnaire sur %lld accepts "
                            "(Delta_f lisse=%.2e <= %.2e) : convergence DCA\n",
                            (long long)st.dca_stall_iters, dca_delta_smoothed, dca_tol);
            }
          } else dca_stall_streak = 0;
        }
        // DCA_ACCEPT_KKT_COUNTER_V1
        if (st.verbose >= 2)
          std::printf("[LazyDCA-ACCEPT] #%lld f=%.10e descente=%.3e requise=%.3e cv=%.2e ||dx_ref||2=%.2e kktFact=%lld kktReuse=%lld\n",
                      (long long)accepts, f_orig, descent, required, cv, std::sqrt(step2),
                      (long long)info.n_factorizations,
                      (long long)info.n_kkt_reuses);

        // Optional emergency atomic rollback.  This is deliberately opt-in:
        // if low-rank geometry is accompanied by a factorisation-stability
        // shift rho_x much larger than model rho, return to scalar-LAZY and
        // re-establish a full-space spectral shift before considering atoms
        // again.  It prevents a bad H from becoming irreversible.
        if (st.atom_oracle_enable && st.atom_oracle_rhox_rollback && atoms.size() > 0 &&
            rho_x > st.atom_oracle_rhox_ratio_trigger * std::max(rho, st.rho_floor)) {
          const idx_t old_atoms = atoms.size();
          atoms = AtomSet{}; atom_brake.clear(); atoms_dirty = true;
          rho_oracle_residual_floor = 0.0;
          if (st.atom_cluster_enable) {
            atom_clusters.clear();
            atom_cluster_seed_stop.clear();
            info.atom_cluster_count_final = 0;
          }
          if (st.adaptive_seed_return_enable) {
            adaptive_seed_return.clear();
            info.n_seed_return_observations = 0;
            info.n_seed_return_low_rounds = 0;
            info.n_seed_return_multi_atom_jumps = 0;
            info.seed_return_seed_atoms_final = 0;
          }
          if (st.adaptive_marginal_return_enable) {
            adaptive_return.reset_model();
            amr21_seed_stop_requested = false;
            sync_adaptive_return_info();
          }
          info.n_atoms = 0; info.atoms_nnz = 0; info.n_atom_rollbacks++;
          hint_valid = true; rho_cap = -1.0; last_lambda_est = std::numeric_limits<double>::quiet_NaN();
          info.n_lanczos_calls++;
          auto rr0 = projected_lanczos_lmin(P.Q, nullptr, st.lanczos_maxit, st.lanczos_tol,
                                            9101u + (unsigned)info.n_atom_rollbacks, nullptr);
          if (std::isfinite(rr0.lmin)) {
            last_lambda_est = rr0.lmin; info.lambda_min_reduced = rr0.lmin;
            const double scalar_target = apply_rho_model_floor(
                std::fabs(std::min(0.0, rr0.lmin)) * (1.0 + st.rho_rel_margin));
            if (!rho_model_fixed) rho = std::max(rho, scalar_target);
          }
          rho_x = 0.0; rho_pending = -1.0; fact_mu = -1.0; reuse_streak = 0;
          reset_momentum_epoch("rollback atomique vers scalar-LAZY");
          if (st.verbose >= 1)
            std::printf("[LazyDCA-atomOracle] ROLLBACK %lld atome(s) : rho_x/rho trop grand, H=0 rho=%.3e\n",
                        (long long)old_atoms, rho);
        }
        // zip 19e (progressive_convexify) : apres chaque pas ACCEPTE, ajouter
        // atoms_per_update atomes tant que Q+H_k mesure du negatif.
        accepts_since_atoms++;
        const bool budget_left = atoms.size() < max_atoms_eff;
        const bool still_nonconvex = !std::isfinite(last_lambda_est) ||
                                     last_lambda_est < -st.atom_neg_tol;
        bool force_update = budget_left && still_nonconvex;

        // GUARDED_PROGRESSIVE_FALLBACK_V269: keep event-driven C as primary.
        const idx_t guarded_event_horizon = std::max<idx_t>(1, st.atom_event_every);
        const idx_t guarded_rank_now = (idx_t)atoms.size();
        if (guarded_last_rank_seen < 0) {
          guarded_last_rank_seen = guarded_rank_now;
          guarded_last_rank_change_iter = info.iters;
        } else if (guarded_rank_now != guarded_last_rank_seen) {
          guarded_last_rank_seen = guarded_rank_now;
          guarded_last_rank_change_iter = info.iters;
          if (st.verbose >= 2)
            std::printf("[LazyDCA-C-v270] retained-rank change it=%lld rank=%lld\\n",
                        (long long)info.iters, (long long)guarded_rank_now);
        }
        // ADAPTIVE_ONLY_EARLY_TAKEOVER_V275A: EARLY takeover is available only
        // to the adaptive selector. Coverage/legacy A-B paths keep v2.7.1 event behavior.
        const bool guarded_event_starved =
            st.atom_event_enable && st.atom_oracle_enable && !guarded_progressive_takeover &&
            event_feas_ok &&
            guarded_rank_now <= guarded_progressive_rank_gate;
        if (guarded_event_starved) {
          guarded_progressive_active = true;
          guarded_progressive_takeover = true;
          guarded_progressive_start_iter = info.iters;
          guarded_progressive_start_rank = guarded_rank_now;
          accepts_since_atoms = 0;
          evt_pending_probe = false;
          evt_has_direction = false;
          inertia_corr_streak = 0;
          if (st.verbose >= 1)
            std::printf("[LazyDCA-C-v271] EARLY low-rank TAKEOVER it=%lld rank=%lld last_rank_change=%lld horizon=%lld\\n",
                        (long long)info.iters, (long long)guarded_rank_now,
                        (long long)guarded_last_rank_change_iter, (long long)guarded_event_horizon);
        }

        if (guarded_progressive_active &&
            (idx_t)atoms.size() >= guarded_progressive_start_rank + guarded_progressive_max_new_atoms) {
          guarded_progressive_active = false;
          if (st.verbose >= 1)
            std::printf("[LazyDCA-C-v271] progressive CAP it=%lld rank=%lld added=%lld\n",
                        (long long)info.iters, (long long)atoms.size(),
                        (long long)((idx_t)atoms.size() - guarded_progressive_start_rank));
        }

        const bool progressive_channel_enabled = !st.atom_event_enable || guarded_progressive_active;
        if (progressive_channel_enabled && budget_left &&
            st.atom_update_every_accepts > 0 &&
            accepts_since_atoms >= st.atom_update_every_accepts)
          { force_update = true; accepts_since_atoms = 0; }

        if (progressive_channel_enabled &&
            !terminal_polish_active && atoms_enabled && !atom_growth_frozen && force_update &&
            info.iters >= last_atom_iter + st.atom_update_every_iters) {
          const idx_t guarded_rank_before = (idx_t)atoms.size();
          const idx_t guarded_added = try_add_atoms(st.atoms_per_update);
          if (guarded_added > 0) {
            last_atom_iter = info.iters;
            if (guarded_progressive_active && st.verbose >= 1)
              std::printf("[LazyDCA-C-v271] progressive COMMIT it=%lld rank=%lld->%lld\n",
                          (long long)info.iters, (long long)guarded_rank_before,
                          (long long)atoms.size());
          }
        }
        // Scalar-shift update at accepted DCA anchors.  CONST deliberately
        // skips this block.  DYN (H_k=0) refreshes a full-space spectral
        // target; lazy/legacy keep the historical low-rank-aware schedule.
        if (!terminal_polish_active && !rho_model_fixed &&
            info.iters >= last_rho_change_iter + rho_spacing_cur) {
          if (reg_dyn) {
            double lhat = std::numeric_limits<double>::quiet_NaN();
            const double target = dyn_scalar_target(73u + (unsigned)(accepts % 10007), &lhat);
            const double tol_rho = 1e-12 * std::max(1.0, std::fabs(rho));
            if (target > rho + tol_rho) {
              const double old_rho = rho;
              rho = target; rho_x = 0.0; rho_pending = -1.0; rho_cut = 2.0;
              fact_mu = -1.0; reuse_streak = 0;
              info.n_rho_model_updates++; info.n_rho_model_increases++;
              if (st.verbose >= 1)
                std::printf("[LazyDCA-rho DYN] curvature refresh lambda_hat=%.3e : rho %.3e -> %.3e\n",
                            lhat, old_rho, rho);
            } else if (target + tol_rho < rho && rho > st.rho_floor) {
              // Genuine scalar dynamics: decay the excess proximal shift but
              // never cross the freshly checked full-space DC floor.
              const double cand = rho_dyn_decay_candidate(rho, target, st.rho_decay);
              if (cand + tol_rho < rho) {
                rho_pending = (rho_pending >= 0.0) ? std::min(rho_pending, cand) : cand;
                if (st.verbose >= 1)
                  std::printf("[LazyDCA-rho DYN] proposition de baisse %.3e -> %.3e "
                              "(plancher %.3e, lambda_hat=%.3e)\n",
                              rho, cand, target, lhat);
              }
            }
            last_rho_change_iter = info.iters;
          } else if (st.rho_monotone_after_atoms && rho > st.rho_floor) {
            // Historical lazy schedule: low-rank coverage can justify a lower
            // isotropic shift.  Every decrease is tried with rho_x=0 first.
            const bool near_term = st.rho_dynamic &&
                                   Ra.feas_rel <= 10.0 * st.eps_feas &&
                                   Ra.comp_rel <= 1e3 * st.eps_comp;
            double cand;
            if (near_term && std::isfinite(last_lambda_est) &&
                last_lambda_est >= -st.atom_neg_tol)
              cand = st.rho_floor;
            else if (near_term)
              cand = rho / rho_cut;
            else
              cand = rho * st.rho_decay;
            cand = apply_rho_model_floor(cand);
            rho_pending = (rho_pending >= 0.0) ? std::min(rho_pending, cand) : cand;
            last_rho_change_iter = info.iters;
          }
        }
      } else if (aP > 1e-6) {
        rejects++; info.n_rejects = rejects;
        if (st.dca_momentum != DcaMomentumMode::Original &&
            info.momentum_force_inf_final > 0.0 && st.dca_momentum_backtrack > 0.0 &&
            st.dca_momentum_backtrack < 1.0) {
          if (st.dca_momentum_complete) { shrink_complete("DCA rejection"); momentum_theta_cur=complete_headline_theta(); }
          else {
            const double old_theta = momentum_theta_cur;
            momentum_theta_cur = std::max(std::max(0.0, st.dca_momentum_theta_min), momentum_theta_cur * st.dca_momentum_backtrack);
            if (momentum_theta_cur < old_theta) { info.n_momentum_backtracks++; if(st.verbose>=2) std::printf("[LazyDCA-momentum] rejet : theta %.3e -> %.3e\n",old_theta,momentum_theta_cur); }
          }
        }
        if (st.dca_inner_max > 0 && dca_inner > 0 &&
            dca_inner % st.dca_inner_max == 0 && !model_ready) {
          ls_multiplier_refresh(); fact_mu = -1.0; reuse_streak = 0;
          if (st.verbose >= 2)
            std::printf("[LazyDCA-inner] cap %lld atteint, recovery a mu/x_ref fixes\n", (long long)dca_inner);
        }
      }
    }
    { double sig2 = objective(P_in, w) + 1e3 * norm_inf(w.dx) * aP;
      if (sig2 == frozen_sig) { if (++frozen_count >= 40) {
        // Le gel bit-identique n'est un ECHEC que si le point est LOIN d'une
        // solution. Avec un sigma desormais borne (dual moins chaotique), le
        // point peut se stabiliser LEGITIMEMENT tres pres de l'optimum (observe :
        // stat_rel=8.5e-6, comp_rel=4.6e-7, quasi convergence) -- le declarer en
        // erreur serait puni un succes. On verifie avant de trancher.
        Residuals Rfz = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
        if (Rfz.feas_rel <= 100.0 * st.eps_feas && Rfz.stat_rel <= 100.0 * st.eps_stat &&
            Rfz.comp_rel <= 100.0 * st.eps_comp) {
          info.status = Status::Solved;
          info.message = "frozen at a near-optimal point (bit-identical iterations, "
                         "residuals within 100x tolerance)";
          if (st.verbose >= 1)
            std::printf("[LazyDCA] gel bit-identique a un point quasi-optimal "
                        "(stat_rel=%.2e, comp_rel=%.2e) : accepte\n",
                        Rfz.stat_rel, Rfz.comp_rel);
        } else {
          // ---- Etage 1 (v0.16.0) : un gel bit-identique LOIN de l'optimum
          // avec slacks ecrases est le symptome classique du Theta infini
          // (pas nul, alpha nul, iteres figes). Push signature-gate, budget
          // partage, puis on laisse la course repartir.
          if (st.bp_adaptive && bp_events < st.bp_max_events && bp_signature()) {
            bp_events++; info.n_bp_events++;
            if (bp_push_escalate(mu, "frozen_stall") > 0) {
              frozen_sig = std::numeric_limits<double>::quiet_NaN();
              frozen_count = 0;
              fact_mu = -1.0; reuse_streak = 0;
              continue;   // iteration suivante : residus/Theta recalcules
            }
          }
          info.status = Status::NumericalError;
          info.message = "stationary stall: bit-identical iterations";
          // v0.16.2 : meme classificateur -- un gel LOIN de l'optimum peut
          // quand meme etre sur un point FAISABLE (stat/comp mauvais =
          // duaux degeneres, pas primal rate). bp_signature() a deja
          // echoue/epuise son budget ci-dessus ; ce n'est pas un probleme
          // de bord, donc pas un "echec numerique" si feas est bon.
          classify_budget_exhausted("frozen-stall recovery", "stationary stall: bit-identical iterations");
        }
        goto finish; } }
      else { frozen_sig = sig2; frozen_count = 0; } }
    // Garde-fou de coherence : rho_x >> rho = le modele a perdu le contact
    // avec la factorisation (essais /2 "certifies" contre un Theta volatil,
    // perimes trois iterations plus tard — NCVXQP8 : rho 24, rho_x 8.5e6,
    // alpha 5e-10). Restauration de rho sur la courbure MESUREE, rho_x purge.
    event_feas_ok = (R.feas_rel <= 1e3 * st.eps_feas);
    if (!rho_model_fixed && rho_x > 100.0 * std::max(rho, 1.0) &&
        info.iters >= last_coherence_iter + 30) {
      last_coherence_iter = info.iters;
      double rho_new = rho;
      if (reg_dyn) {
        // DYN never transfers a purely numerical rho_x into the DC model
        // unless a fresh full-space curvature probe asks for a larger shift.
        rho_new = std::max(rho, dyn_scalar_target(211u + (unsigned)(info.iters % 10007)));
      } else {
        rho_new = std::isfinite(last_lambda_est) && last_lambda_est < 0.0
                  ? std::fabs(last_lambda_est) * (1.0 + st.rho_rel_margin)
                  : rho + 0.5 * rho_x;
        rho_new = std::max(rho_new, rho);
      }
      if (rho_new > rho) {
        const double old_rho = rho;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-rho] coherence restauree : rho %.3e -> %.3e (rho_x %.3e purge)\n",
                      rho, rho_new, rho_x);
        rho = std::min(rho_new, st.rho_abs_max); rho_x = 0.0;
        info.n_rho_model_updates++; info.n_rho_model_increases++;
        probe_hist[probe_pos % 3] = rho; probe_pos++;
        fact_mu = -1.0; reuse_streak = 0; rho_pending = -1.0;
        // Re-ancrage : l'ancre x_ref date d'avant la corruption du modele.
        if (Ra_ok_for_anchor()) {
          reset_momentum_epoch("restauration de coherence / re-ancrage");
          x_ref = w.x; ref_obj = objective(P_in, w);
          dca_accept_obj_hist.clear(); dca_accept_obj_hist.push_back(ref_obj);
          dca_inner = 0; dca_model_start = std::numeric_limits<double>::infinity();
          Residuals Rg = compute_residuals(P, w, mu);
          if (Rg.ncomp > 0) mu = std::min(1.0, std::max(mu, 1e-2 * Rg.gap / (double)Rg.ncomp));
          mu_last_change_iter = info.iters;
          if (st.verbose >= 1)
            std::printf("[LazyDCA-rho] re-ancrage post-restauration (x_ref <- x, mu recale)\n");
        }
      } else if (reg_dyn && st.verbose >= 2) {
        std::printf("[LazyDCA-rho DYN] rho_x=%.3e est numerique seulement; rho modele reste %.3e\n",
                    rho_x, rho);
      }
    }
    if (evt_pending_probe && !atom_growth_frozen && info.iters >= evt_pending_since + 15 &&
        atoms.size() < max_atoms_eff) {
      // timeout : aucun accept depuis 15 iterations (deadlock) — le remede ne
      // peut plus attendre, insertion mi-course assumee.
      idx_t added_e = 0;
      if (evt_has_direction) {
        atoms.idx.push_back(std::move(evt_ui)); atoms.val.push_back(std::move(evt_uv));
        atoms.beta.push_back(evt_beta); added_e = 1;
        // ADAPTIVE_MARGINAL_RETURN_V21: direct emergency commit has no oracle
        // lift score; count it and treat it as strong reactivation evidence.
        if (st.adaptive_marginal_return_enable) {
          const auto amr21_direct = adaptive_return.observe_unscored(
              (std::size_t)atoms.size(), AdaptiveReturnSource::EventDirect, true);
          (void)amr21_direct;
          sync_adaptive_return_info();
        }
        atoms_dirty = true; info.n_atoms = atoms.size(); info.atoms_nnz = atoms.nnz();
        rho_cap = -1.0; hint_valid = false;
      } else added_e = try_add_atoms(1);
      if (added_e > 0) {
        // RE-ANCRAGE SIMULTANE : sans lui, x_ref reste celui d'AVANT le
        // deadlock (fige depuis 15+ iterations, x a derive loin) — le terme
        // beta*(a'(x-x_ref))^2 du nouvel atome n'est alors PAS nul en x,
        // violant la doctrine "insertion sans couture" (0.15.3) et injectant
        // une discontinuite de merite qui corrompt la factorisation suivante
        // (observe : explosion inf_du 3e6 + coherence-restore a l'iteration
        // IMMEDIATEMENT suivante, systematiquement). Re-ancrer restaure la
        // doctrine meme dans le cas deadlock.
        reset_momentum_epoch("insertion atomique deadlock / re-ancrage");
        x_ref = w.x; ref_obj = objective(P_in, w);
        dca_accept_obj_hist.clear(); dca_accept_obj_hist.push_back(ref_obj);
        dca_inner = 0; dca_model_start = std::numeric_limits<double>::infinity();
        Residuals Rd = compute_residuals(P, w, mu);
        if (Rd.ncomp > 0) mu = std::min(1.0, std::max(mu, 1e-2 * Rd.gap / (double)Rd.ncomp));
        mu_last_change_iter = info.iters;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-atomEvt %c] timeout deadlock : atome #%lld insere ET x_ref reancre "
                      "(merite sans discontinuite)\n", evt_pending_tag, (long long)atoms.size());
      }
      evt_pending_probe = false; evt_has_direction = false;
    }
    if (info.n_inertia_corrections > prev_inertia_corr) inertia_corr_streak++;
    else inertia_corr_streak = 0;
    prev_inertia_corr = info.n_inertia_corrections;
    if (inertia_corr_streak >= 3 && amr21_atom_event_budget_ok()) {
      // [A] 3 iterations consecutives corrigees : Theta+rho+H ne couvre pas
      // la courbure sur la face courante — rho_x rustinait isotropiquement
      // ce qu'un atome cible.
      last_event_atom_iter = info.iters; inertia_corr_streak = 0;
      if (!evt_pending_probe) {
        evt_pending_probe = true; evt_pending_tag = 'A'; evt_pending_since = info.iters;
        evt_has_direction = false;
        if (st.verbose >= 1)
          std::printf("[LazyDCA-atomEvt A] strie de corrections d'inertie : sonde en attente d'accept\n");
      }
    }
    last_apr = aP; last_adu = aD; last_dnorm = norm_inf(w.dx);
    last_clean_full = (did_reuse || clean_first_try) && aP >= st.kkt_reuse_alpha_min;
    (void)last_dnorm;
    for (idx_t i = 0; i < n; ++i) w.x[i] += aP * w.dx[i];
    for (idx_t i = 0; i < m; ++i) w.s[i] += aP * w.ds[i];
    // Convention IPOPT : y avance avec le pas PRIMAL. Le pas plein
    // inconditionnel (0.5.3) empoisonnait les duals quand Armijo rejette
    // (dy calcule pour un deplacement qui n'a pas eu lieu -> inf_du explose).
    // Le deadlock dual-domine est couvert par le LS-refresh (streak 2).
    for (idx_t i = 0; i < p; ++i) w.y[i] += aP * w.dy[i];
    // Garde-fou de coherence primal-dual : dvl/dvu linearisent la
    // complementarite EN SUPPOSANT le pas primal dx. Sur rejet Armijo
    // (aP ~ 1e-8) un pas dual PLEIN rend (vl+dvl)*sl ~ mu - vl*dx : faux de
    // plusieurs ordres -> les spikes inf_du 1e6 des queues. Plafond :
    // aD <= max(10*aP, 1e-2) — la reparation duale reste possible (plancher),
    // le pas plein sur primal gele est tue.
    double aDb = std::min(aD, std::max(10.0 * aP, 1e-2));
    // Region de confiance DUALE : sur rejet Armijo, dvl vient d'un systeme
    // resolu contre un modele potentiellement corrompu (|dvl| ~ 1e8 observe) ;
    // meme plafonne a 1e-2 le pas multiplie les duals par 1e6 (cycle limite
    // photographie sur NCVXQP2/7, periode ~18 : explosion 3e6 -> re-centrage
    // mu -> decrue -> rebelote). Borne : un multiplicateur ne peut pas etre
    // multiplie par plus de ~10 en une iteration.
    {
      double vmax = 1.0, dvmax = 0.0;
      for (idx_t i = 0; i < n; ++i) {
        vmax = std::max({vmax, w.vl[i], w.vu[i]});
        dvmax = std::max({dvmax, std::fabs(w.dvl[i]), std::fabs(w.dvu[i])});
      }
      for (idx_t i = 0; i < m; ++i) {
        vmax = std::max({vmax, w.wl[i], w.wu[i]});
        dvmax = std::max({dvmax, std::fabs(w.dwl[i]), std::fabs(w.dwu[i])});
      }
      if (dvmax > 0.0) aDb = std::min(aDb, 10.0 * vmax / dvmax);
    }
    for (idx_t i = 0; i < n; ++i) { w.vl[i] = std::max(0.0, w.vl[i] + aDb * w.dvl[i]); w.vu[i] = std::max(0.0, w.vu[i] + aDb * w.dvu[i]); }
    for (idx_t i = 0; i < m; ++i) {
      w.wl[i] = std::max(0.0, w.wl[i] + aDb * w.dwl[i]);
      w.wu[i] = std::max(0.0, w.wu[i] + aDb * w.dwu[i]);
      w.z[i] = w.wl[i] - w.wu[i];
    }
    if (sigma > 1e14) {
      // v0.16.2 : meme classificateur que la soupape/plafond rho — un point
      // FAISABLE ne recoit jamais "echec numerique" brut, quelle que soit la
      // ressource epuisee (sigma ici, rho ailleurs).
      classify_budget_exhausted("sigma_max (penalty escalation)", "penalty parameter blow-up");
      if (info.status == Status::NumericalError) info.status = Status::PrimalInfeasibleLike;
      break;
    }
  }

finish:
  // TERMINAL_QUALIFICATION_V261: an external resource exit during the explicit
  // recovery cannot silently return a partially recovered state.
  if (terminal_recovery_active && terminal_recovery_entry_valid &&
      info.status != Status::Solved && info.status != Status::SolvedSOC) {
    w = terminal_recovery_entry_work;
    mu = terminal_recovery_entry_mu;
    rho = terminal_recovery_entry_rho;
    rho_x = terminal_recovery_entry_rho_x;
    rho_pending = terminal_recovery_entry_rho_pending;
    rho_cut = terminal_recovery_entry_rho_cut;
    terminal_recovery_active = false;
    terminal_polish_active = false;
    fact_mu = -1.0; reuse_streak = 0;
    if (st.verbose >= 1)
      std::printf("[LazyDCA-terminal-v262] external exit during recovery; restored entry iterate\n");
  }
  // Ne jamais sortir en erreur (ou en timeout/max-iter) en jetant un
  // incumbent faisable : restaurer INCONDITIONNELLEMENT le meilleur point
  // faisable connu des que best_feas_valid, sur TOUS les statuts non
  // "propres" -- y compris MaxIter et TimeLimit. bx_/bs_/... n'a ete
  // snapshote QUE sur un pas ACCEPTE (feasibilite deja garantie par le
  // critere d'acceptation DCA a ce moment-la, cf. sd_cv_abs/sd_cv_rel) ;
  // feas_rel est une fonction deterministe de (x,s), donc recalculer sur le
  // meme bx_/bs_ ne peut pas "degrader" la faisabilite entre le moment du
  // snapshot et la restauration -- l'ancien garde-fou comparatif (revert si
  // stat_rel du point courant etait meilleur de plus de 10x) donnait la
  // priorite a la stationnarite du point BRUT au detriment de la garantie
  // de faisabilite explicitement demandee : retire.
  bool restored_best_feas = false;
  if (!converged &&
      (info.status == Status::NumericalError || info.status == Status::MaxIter ||
       info.status == Status::TimeLimit || info.status == Status::DualDegenerate ||
       info.status == Status::Unbounded ||
       info.status == Status::FactorizationBudget) &&
      best_feas_valid) {
    std::swap(w.x, bx_); std::swap(w.s, bs_); std::swap(w.y, by_); std::swap(w.z, bz_);
    std::swap(w.vl, bvl_); std::swap(w.vu, bvu_); std::swap(w.wl, bwl_); std::swap(w.wu, bwu_);
    restored_best_feas = true;
    if (st.verbose >= 1)
      std::printf("[LazyDCA] restauration du meilleur point faisable (obj %.10e)\n", best_feas_obj);
  }
  if (!converged && !restored_best_feas && terminal_best_valid) {
    Residuals Rend2 = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
    if (terminal_best_stat < Rend2.stat_rel) { w = terminal_best_work;
      if (st.verbose >= 1) std::printf("[LazyDCA-polish] restauration du meilleur itere (stat %.3e)\n", terminal_best_stat); }
  }
  R = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
  bool final_strict_kkt =
      R.feas_rel <= st.eps_feas && R.stat_rel <= st.eps_stat && R.comp_rel <= st.eps_comp;

  // One fixed-primal equality-multiplier refresh is allowed before declaring
  // a feasible terminal point uncertified. The pre-existing refresh routine is
  // itself guarded against worsening stationarity; this wrapper additionally
  // checks the original-coordinate residuals transactionally. No primal iterate
  // and no objective value is changed by this qualification step.
  if (!final_strict_kkt && R.feas_rel <= 100.0 * st.eps_feas) {
    const Work terminal_dual_refresh_entry = w;
    const Residuals terminal_dual_before = R;
    ls_multiplier_refresh();
    Residuals terminal_dual_after = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
    const bool terminal_dual_admissible =
        std::isfinite(terminal_dual_after.feas_rel) &&
        std::isfinite(terminal_dual_after.stat_rel) &&
        std::isfinite(terminal_dual_after.comp_rel) &&
        terminal_dual_after.feas_rel <= terminal_dual_before.feas_rel * (1.0 + 1e-9) + 1e-14 &&
        terminal_dual_after.stat_rel <= terminal_dual_before.stat_rel * (1.0 + 1e-9) + 1e-15 &&
        terminal_dual_after.comp_rel <= terminal_dual_before.comp_rel * (1.0 + 1e-9) + 1e-15;
    if (terminal_dual_admissible) {
      R = terminal_dual_after;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v262] fixed-primal dual refresh: stat %.3e -> %.3e, comp %.3e -> %.3e\n",
                    terminal_dual_before.stat_rel, R.stat_rel,
                    terminal_dual_before.comp_rel, R.comp_rel);
    } else {
      w = terminal_dual_refresh_entry;
      R = terminal_dual_before;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v262] fixed-primal dual refresh rejected by original-residual guard\n");
    }
    final_strict_kkt =
        R.feas_rel <= st.eps_feas && R.stat_rel <= st.eps_stat && R.comp_rel <= st.eps_comp;
  }

  // TERMINAL_ACTIVE_DUAL_V262: if the primal point is feasible but the
  // equality-only refresh is insufficient, reconstruct all multipliers at
  // fixed (x,s).  This is a terminal qualification step only: it never moves
  // the primal point and is committed only when ORIGINAL-coordinate
  // stationarity and complementarity are both non-worse.
  if (!final_strict_kkt && R.feas_rel <= 100.0 * st.eps_feas) {
    const Work dual_recon_entry = w;
    const Residuals dual_recon_before = R;
    Work dual_recon_best = w;
    Residuals dual_recon_best_R = R;
    double dual_recon_best_merit = std::max(
        R.stat_rel / std::max(st.eps_stat, 1e-30),
        R.comp_rel / std::max(st.eps_comp, 1e-30));
    double dual_recon_best_tol = -1.0;
    idx_t dual_recon_best_active_i = 0, dual_recon_best_active_x = 0;

    std::vector<double> c_row_norm2(m, 0.0);
    for (idx_t j = 0; j < P.C.cols; ++j)
      for (idx_t pp = P.C.colptr[j]; pp < P.C.colptr[j + 1]; ++pp) {
        const idx_t i = P.C.rowind[pp];
        const double v = P.C.values[pp];
        if (i >= 0 && i < m) c_row_norm2[i] += v * v;
      }

    const double active_tols[] = {1e-12, 1e-10, 1e-8, 1e-6};
    for (double act_tol : active_tols) {
      w = dual_recon_entry;
      std::vector<char> ilow(m, 0), iup(m, 0), xlow(n, 0), xup(n, 0);
      idx_t n_active_i = 0, n_active_x = 0;

      auto close_to = [&](double gap, double bnd) -> bool {
        if (!std::isfinite(bnd) || !std::isfinite(gap)) return false;
        return gap <= act_tol * std::max(1.0, std::fabs(bnd));
      };

      for (idx_t i = 0; i < m; ++i) {
        ilow[i] = w.hsl[i] && close_to(w.s[i] - P.cl[i], P.cl[i]);
        iup[i]  = w.hsu[i] && close_to(P.cu[i] - w.s[i], P.cu[i]);
        if (ilow[i] || iup[i]) ++n_active_i;
      }
      for (idx_t i = 0; i < n; ++i) {
        xlow[i] = w.hl[i] && close_to(w.x[i] - P.l[i], P.l[i]);
        xup[i]  = w.hu[i] && close_to(P.u[i] - w.x[i], P.u[i]);
        if (xlow[i] || xup[i]) ++n_active_x;
      }

      // Start the inequality-normal reconstruction from zero. This removes
      // stale interior-point multipliers before solving the active-face dual
      // least-squares problem.
      std::fill(w.z.begin(), w.z.end(), 0.0);
      std::fill(w.wl.begin(), w.wl.end(), 0.0);
      std::fill(w.wu.begin(), w.wu.end(), 0.0);
      std::fill(w.vl.begin(), w.vl.end(), 0.0);
      std::fill(w.vu.begin(), w.vu.end(), 0.0);

      auto sync_slack_duals = [&]() {
        for (idx_t i = 0; i < m; ++i) {
          if (ilow[i] && iup[i]) {
            w.wl[i] = std::max(0.0, w.z[i]);
            w.wu[i] = std::max(0.0, -w.z[i]);
          } else if (ilow[i]) {
            w.z[i] = std::max(0.0, w.z[i]);
            w.wl[i] = w.z[i]; w.wu[i] = 0.0;
          } else if (iup[i]) {
            w.z[i] = std::min(0.0, w.z[i]);
            w.wl[i] = 0.0; w.wu[i] = -w.z[i];
          } else {
            w.z[i] = 0.0; w.wl[i] = 0.0; w.wu[i] = 0.0;
          }
        }
      };

      auto rebuild_x_bound_duals = [&]() {
        std::vector<double> gx(n, 0.0), aty(n, 0.0), ctz(n, 0.0);
        P.Q.gaxpy_sym(w.x.data(), gx.data());
        if (p) P.A.gaxpy_t(w.y.data(), aty.data());
        if (m) P.C.gaxpy_t(w.z.data(), ctz.data());
        for (idx_t i = 0; i < n; ++i) {
          const double base = gx[i] + P.q[i] - aty[i] - ctz[i];
          double nu = 0.0; // nu = vl-vu; stationarity contains -nu.
          if (xlow[i] && xup[i]) nu = base;
          else if (xlow[i]) nu = std::max(0.0, base);
          else if (xup[i]) nu = std::min(0.0, base);
          w.vl[i] = std::max(0.0, nu);
          w.vu[i] = std::max(0.0, -nu);
        }
      };

      auto x_stationarity_vector = [&]() {
        std::vector<double> gx(n, 0.0), aty(n, 0.0), ctz(n, 0.0), rr(n, 0.0);
        P.Q.gaxpy_sym(w.x.data(), gx.data());
        if (p) P.A.gaxpy_t(w.y.data(), aty.data());
        if (m) P.C.gaxpy_t(w.z.data(), ctz.data());
        for (idx_t i = 0; i < n; ++i)
          rr[i] = gx[i] + P.q[i] - aty[i] - ctz[i] - w.vl[i] + w.vu[i];
        return rr;
      };

      sync_slack_duals();
      ls_multiplier_refresh();
      rebuild_x_bound_duals();

      const int max_sweeps = (m <= 1000 ? 24 : 12);
      double omega = 0.35;
      Residuals last_scan_R = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
      double last_scan_merit = std::max(
          last_scan_R.stat_rel / std::max(st.eps_stat, 1e-30),
          last_scan_R.comp_rel / std::max(st.eps_comp, 1e-30));

      for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        if (m > 0 && n_active_i > 0) {
          std::vector<double> rr = x_stationarity_vector();
          std::vector<double> crr(m, 0.0);
          P.C.gaxpy(rr.data(), crr.data());
          for (idx_t i = 0; i < m; ++i) {
            if (!(ilow[i] || iup[i]) || c_row_norm2[i] <= 1e-30) continue;
            double zi = w.z[i] + omega * crr[i] / c_row_norm2[i];
            if (ilow[i] && !iup[i]) zi = std::max(0.0, zi);
            else if (iup[i] && !ilow[i]) zi = std::min(0.0, zi);
            w.z[i] = zi;
          }
          sync_slack_duals();
        }

        if (sweep % 4 == 3 || sweep + 1 == max_sweeps) ls_multiplier_refresh();
        rebuild_x_bound_duals();
        Residuals Rs = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
        const double merit = std::max(
            Rs.stat_rel / std::max(st.eps_stat, 1e-30),
            Rs.comp_rel / std::max(st.eps_comp, 1e-30));
        if (std::isfinite(merit) && merit > last_scan_merit * 1.05) omega *= 0.5;
        last_scan_merit = std::min(last_scan_merit, merit);
      }

      ls_multiplier_refresh();
      rebuild_x_bound_duals();
      Residuals Rscan = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
      const bool finite_scan = std::isfinite(Rscan.feas_rel) &&
                               std::isfinite(Rscan.stat_rel) &&
                               std::isfinite(Rscan.comp_rel);
      const bool primal_same = finite_scan &&
          Rscan.feas_rel <= dual_recon_before.feas_rel * (1.0 + 1e-9) + 1e-14;
      const bool dual_nonworse = finite_scan &&
          Rscan.stat_rel <= dual_recon_before.stat_rel * (1.0 + 1e-8) + 1e-14 &&
          Rscan.comp_rel <= dual_recon_before.comp_rel * (1.0 + 1e-8) + 1e-14;
      const double merit = finite_scan ? std::max(
          Rscan.stat_rel / std::max(st.eps_stat, 1e-30),
          Rscan.comp_rel / std::max(st.eps_comp, 1e-30)) :
          std::numeric_limits<double>::infinity();

      if (st.verbose >= 2)
        std::printf("[LazyDCA-terminal-v262] dual scan tol=%.1e activeI=%lld activeX=%lld stat=%.3e comp=%.3e merit=%.3e\n",
                    act_tol, (long long)n_active_i, (long long)n_active_x,
                    Rscan.stat_rel, Rscan.comp_rel, merit);

      if (primal_same && dual_nonworse && merit < dual_recon_best_merit) {
        dual_recon_best = w;
        dual_recon_best_R = Rscan;
        dual_recon_best_merit = merit;
        dual_recon_best_tol = act_tol;
        dual_recon_best_active_i = n_active_i;
        dual_recon_best_active_x = n_active_x;
      }
    }

    if (dual_recon_best_tol > 0.0) {
      w = dual_recon_best;
      R = dual_recon_best_R;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v262] active-set dual reconstruction: tol=%.1e activeI=%lld activeX=%lld stat %.3e -> %.3e, comp %.3e -> %.3e\n",
                    dual_recon_best_tol, (long long)dual_recon_best_active_i,
                    (long long)dual_recon_best_active_x,
                    dual_recon_before.stat_rel, R.stat_rel,
                    dual_recon_before.comp_rel, R.comp_rel);
    } else {
      w = dual_recon_entry;
      R = dual_recon_before;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v262] active-set dual reconstruction: no componentwise-admissible improvement; original dual state restored\n");
    }

    final_strict_kkt =
        R.feas_rel <= st.eps_feas && R.stat_rel <= st.eps_stat && R.comp_rel <= st.eps_comp;
  }

  // TERMINAL_ACTIVE_FACE_V264 runtime/status-gate correction.
  // TERMINAL_ACTIVE_FACE_V263: guarded original-QP active-face KKT rescue.
  // Small-problem only (n <= 512). It runs after the fixed-primal dual
  // reconstruction and commits a new primal point ONLY when the ORIGINAL-
  // coordinate first-order KKT certificate is strict and the objective is
  // not worsened beyond 1e-8 relative. The main DCA/IPM trajectory and its
  // publication iteration budget are untouched.
  // TERMINAL_ACTIVE_FACE_V264: the v2.6.3 rescue was guarded by the
  // pre-finalization status.  That can suppress a mathematically valid
  // postsolve when the main loop has already set NumericalError even though
  // the retained point is finite and feasible; the later max-iteration
  // precedence can then hide that status as MaxIter.  Qualification is now
  // driven by the retained point.  Resource/structural impossibility exits
  // remain excluded, but a finite feasible point may be rescued after an
  // internal numerical exit.  Failure is transactional and preserves the
  // underlying status.
  const bool terminal_face_finite_v264 =
      std::isfinite(R.feas_rel) && std::isfinite(R.stat_rel) && std::isfinite(R.comp_rel);
  const bool terminal_face_status_allowed_v264 =
      info.status != Status::TimeLimit &&
      info.status != Status::Unbounded &&
      info.status != Status::PrimalInfeasibleLike;
  auto terminal_status_name_v264 = [&](Status s) -> const char* {
    switch (s) {
      case Status::Solved: return "solved";
      case Status::SolvedSOC: return "solved_soc";
      case Status::MaxIter: return "max_iter";
      case Status::TimeLimit: return "time_limit";
      case Status::PrimalInfeasibleLike: return "primal_infeasible_like";
      case Status::NumericalError: return "numerical_error";
      case Status::Unbounded: return "unbounded";
      case Status::DCAConverged: return "dca_converged";
      case Status::DualDegenerate: return "dual_degenerate";
      default: return "other";
    }
  };
  if (st.verbose >= 1 && !final_strict_kkt && n <= 512)
    std::printf("[LazyDCA-terminal-v264] active-face entry prestatus=%s finite=%s allowed=%s feas=%.3e stat=%.3e comp=%.3e n=%lld\\n",
                terminal_status_name_v264(info.status), terminal_face_finite_v264 ? "yes" : "no",
                terminal_face_status_allowed_v264 ? "yes" : "no",
                R.feas_rel, R.stat_rel, R.comp_rel, (long long)n);

  if (!final_strict_kkt && terminal_face_finite_v264 &&
      R.feas_rel <= 100.0 * st.eps_feas && n <= 512 &&
      terminal_face_status_allowed_v264) {
    const Work face_entry = w;
    const Residuals face_before = R;
    const double face_obj_before = objective(P_in, w);
    const double face_obj_scale = std::max(1.0, std::fabs(face_obj_before));

    double max_comp_pressure = 0.0;
    for (idx_t i = 0; i < m; ++i) {
      if (w.hsl[i]) max_comp_pressure = std::max(max_comp_pressure,
          std::max(0.0, w.wl[i]) * std::max(0.0, w.s[i] - P.cl[i]));
      if (w.hsu[i]) max_comp_pressure = std::max(max_comp_pressure,
          std::max(0.0, w.wu[i]) * std::max(0.0, P.cu[i] - w.s[i]));
    }
    for (idx_t i = 0; i < n; ++i) {
      if (w.hl[i]) max_comp_pressure = std::max(max_comp_pressure,
          std::max(0.0, w.vl[i]) * std::max(0.0, w.x[i] - P.l[i]));
      if (w.hu[i]) max_comp_pressure = std::max(max_comp_pressure,
          std::max(0.0, w.vu[i]) * std::max(0.0, P.u[i] - w.x[i]));
    }

    struct FaceRowV263 {
      // 0=A equality; 1=C lower; 2=C upper represented by -C;
      // 3=C fixed/free; 4=x lower; 5=x upper represented by -e;
      // 6=x fixed/free.
      int kind = 0;
      idx_t idx = 0;
      double rhs = 0.0;
    };

    Work face_best = face_entry;
    Residuals face_best_R = face_before;
    double face_best_obj = face_obj_before;
    double face_best_obj_rel = 0.0;
    double face_best_step = 0.0;
    double face_best_linres = std::numeric_limits<double>::infinity();
    int face_best_variant = -1;
    idx_t face_best_rows = 0, face_best_rank = 0;
    bool face_best_strict = false;

    const double prox_tols[] = {1e-10, 1e-8, 1e-6, 1e-4, 1e-2};
    const double pressure_fracs[] = {0.50, 0.10, 0.01, 0.001};
    const int nvariants = 9;

    for (int variant = 0; variant < nvariants; ++variant) {
      const bool pressure_mode = variant >= 5;
      const double prox_tol = pressure_mode ? 1e-4 : prox_tols[variant];
      const double pressure_frac = pressure_mode ? pressure_fracs[variant - 5] : -1.0;
      std::vector<FaceRowV263> rows;
      rows.reserve((size_t)p + (size_t)m + (size_t)n);
      for (idx_t i = 0; i < p; ++i) rows.push_back({0, i, P.b[i]});

      auto close_gap = [&](double gap, double bound, double value) -> bool {
        if (!std::isfinite(gap) || !std::isfinite(bound) || gap < -1e-10) return false;
        const double sc = std::max({1.0, std::fabs(bound), std::fabs(value)});
        return gap <= prox_tol * sc;
      };
      auto pressure_hit = [&](double dual, double gap) -> bool {
        if (!pressure_mode || !(max_comp_pressure > 0.0) || !std::isfinite(gap)) return false;
        const double prod = std::max(0.0, dual) * std::max(0.0, gap);
        return prod >= pressure_frac * max_comp_pressure;
      };

      for (idx_t i = 0; i < m; ++i) {
        const bool finite_l = w.hsl[i];
        const bool finite_u = w.hsu[i];
        const double width_sc = (finite_l && finite_u)
            ? std::max({1.0, std::fabs(P.cl[i]), std::fabs(P.cu[i])}) : 1.0;
        const bool fixed = finite_l && finite_u &&
            std::fabs(P.cu[i] - P.cl[i]) <= 1e-12 * width_sc;
        if (fixed) { rows.push_back({3, i, P.cl[i]}); continue; }
        const double gl = finite_l ? w.s[i] - P.cl[i] : std::numeric_limits<double>::infinity();
        const double gu = finite_u ? P.cu[i] - w.s[i] : std::numeric_limits<double>::infinity();
        bool al = finite_l && (close_gap(gl, P.cl[i], w.s[i]) || pressure_hit(w.wl[i], gl));
        bool au = finite_u && (close_gap(gu, P.cu[i], w.s[i]) || pressure_hit(w.wu[i], gu));
        if (al && au) {
          const double pl = std::max(0.0, w.wl[i]) * std::max(0.0, gl);
          const double pu = std::max(0.0, w.wu[i]) * std::max(0.0, gu);
          if (pressure_mode && std::fabs(pl - pu) > 1e-30) { al = pl >= pu; au = !al; }
          else { al = gl <= gu; au = !al; }
        }
        if (al) rows.push_back({1, i, P.cl[i]});
        else if (au) rows.push_back({2, i, -P.cu[i]});
      }

      for (idx_t i = 0; i < n; ++i) {
        const bool finite_l = w.hl[i];
        const bool finite_u = w.hu[i];
        const double width_sc = (finite_l && finite_u)
            ? std::max({1.0, std::fabs(P.l[i]), std::fabs(P.u[i])}) : 1.0;
        const bool fixed = finite_l && finite_u &&
            std::fabs(P.u[i] - P.l[i]) <= 1e-12 * width_sc;
        if (fixed) { rows.push_back({6, i, P.l[i]}); continue; }
        const double gl = finite_l ? w.x[i] - P.l[i] : std::numeric_limits<double>::infinity();
        const double gu = finite_u ? P.u[i] - w.x[i] : std::numeric_limits<double>::infinity();
        bool al = finite_l && (close_gap(gl, P.l[i], w.x[i]) || pressure_hit(w.vl[i], gl));
        bool au = finite_u && (close_gap(gu, P.u[i], w.x[i]) || pressure_hit(w.vu[i], gu));
        if (al && au) {
          const double pl = std::max(0.0, w.vl[i]) * std::max(0.0, gl);
          const double pu = std::max(0.0, w.vu[i]) * std::max(0.0, gu);
          if (pressure_mode && std::fabs(pl - pu) > 1e-30) { al = pl >= pu; au = !al; }
          else { al = gl <= gu; au = !al; }
        }
        if (al) rows.push_back({4, i, P.l[i]});
        else if (au) rows.push_back({5, i, -P.u[i]});
      }

      const idx_t nr = (idx_t)rows.size();
      if (nr <= 0) continue;
      const Eigen::Index N = (Eigen::Index)n;
      const Eigen::Index NR = (Eigen::Index)nr;
      Eigen::MatrixXd B = Eigen::MatrixXd::Zero(NR, N);
      Eigen::VectorXd h = Eigen::VectorXd::Zero(NR);
      std::vector<idx_t> cpos(m, -1), xpos(n, -1);
      std::vector<double> csign(m, 0.0), xsign(n, 0.0);
      for (idx_t r = 0; r < nr; ++r) {
        h[(Eigen::Index)r] = rows[(size_t)r].rhs;
        const auto &fr = rows[(size_t)r];
        if (fr.kind >= 1 && fr.kind <= 3) {
          cpos[fr.idx] = r;
          csign[fr.idx] = (fr.kind == 2 ? -1.0 : 1.0);
        } else if (fr.kind >= 4 && fr.kind <= 6) {
          xpos[fr.idx] = r;
          xsign[fr.idx] = (fr.kind == 5 ? -1.0 : 1.0);
        }
      }
      for (idx_t j = 0; j < P.A.cols; ++j)
        for (idx_t pp = P.A.colptr[j]; pp < P.A.colptr[j + 1]; ++pp) {
          const idx_t i = P.A.rowind[pp];
          if (i >= 0 && i < p) B((Eigen::Index)i, (Eigen::Index)j) += P.A.values[pp];
        }
      for (idx_t j = 0; j < P.C.cols; ++j)
        for (idx_t pp = P.C.colptr[j]; pp < P.C.colptr[j + 1]; ++pp) {
          const idx_t i = P.C.rowind[pp];
          if (i >= 0 && i < m && cpos[i] >= 0)
            B((Eigen::Index)cpos[i], (Eigen::Index)j) += csign[i] * P.C.values[pp];
        }
      for (idx_t i = 0; i < n; ++i)
        if (xpos[i] >= 0) B((Eigen::Index)xpos[i], (Eigen::Index)i) = xsign[i];

      Eigen::MatrixXd K = Eigen::MatrixXd::Zero(N + NR, N + NR);
      for (idx_t j = 0; j < P.Q.U.cols; ++j)
        for (idx_t pp = P.Q.U.colptr[j]; pp < P.Q.U.colptr[j + 1]; ++pp) {
          const idx_t i = P.Q.U.rowind[pp];
          const double v = P.Q.U.values[pp];
          if (i < 0 || i >= n || j < 0 || j >= n) continue;
          K((Eigen::Index)i, (Eigen::Index)j) += v;
          if (i != j) K((Eigen::Index)j, (Eigen::Index)i) += v;
        }
      K.block(0, N, N, NR) = -B.transpose();
      K.block(N, 0, NR, N) = B;
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(N + NR);
      for (idx_t i = 0; i < n; ++i) rhs[(Eigen::Index)i] = -P.q[i];
      rhs.tail(NR) = h;

      Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(K);
      cod.setThreshold(1e-11);
      Eigen::VectorXd sol = cod.solve(rhs);
      if (!sol.allFinite()) continue;
      const double linres = (K * sol - rhs).lpNorm<Eigen::Infinity>();
      const double rhssc = 1.0 + rhs.lpNorm<Eigen::Infinity>();
      if (!(std::isfinite(linres) && linres <= 1e-7 * rhssc)) {
        if (st.verbose >= 2)
          std::printf("[LazyDCA-terminal-v264] face variant=%d rejected: linear residual %.3e\n", variant, linres / rhssc);
        continue;
      }

      Work wc = face_entry;
      for (idx_t i = 0; i < n; ++i) wc.x[i] = sol[(Eigen::Index)i];
      if (m) {
        std::fill(wc.s.begin(), wc.s.end(), 0.0);
        P.C.gaxpy(wc.x.data(), wc.s.data());
      }
      std::fill(wc.y.begin(), wc.y.end(), 0.0);
      std::fill(wc.z.begin(), wc.z.end(), 0.0);
      std::fill(wc.vl.begin(), wc.vl.end(), 0.0);
      std::fill(wc.vu.begin(), wc.vu.end(), 0.0);
      std::fill(wc.wl.begin(), wc.wl.end(), 0.0);
      std::fill(wc.wu.begin(), wc.wu.end(), 0.0);

      double lam_inf = 0.0;
      for (idx_t r = 0; r < nr; ++r)
        lam_inf = std::max(lam_inf, std::fabs(sol[N + (Eigen::Index)r]));
      const double sign_tol = 1e-8 * (1.0 + lam_inf);
      bool sign_ok = true;
      for (idx_t r = 0; r < nr; ++r) {
        const auto &fr = rows[(size_t)r];
        double lam = sol[N + (Eigen::Index)r];
        if (fr.kind == 0) wc.y[fr.idx] = lam;
        else if (fr.kind == 3) {
          wc.z[fr.idx] = lam;
          wc.wl[fr.idx] = std::max(0.0, lam);
          wc.wu[fr.idx] = std::max(0.0, -lam);
        } else if (fr.kind == 6) {
          wc.vl[fr.idx] = std::max(0.0, lam);
          wc.vu[fr.idx] = std::max(0.0, -lam);
        } else {
          if (lam < -sign_tol) { sign_ok = false; break; }
          lam = std::max(0.0, lam);
          if (fr.kind == 1) { wc.z[fr.idx] = lam; wc.wl[fr.idx] = lam; }
          else if (fr.kind == 2) { wc.z[fr.idx] = -lam; wc.wu[fr.idx] = lam; }
          else if (fr.kind == 4) wc.vl[fr.idx] = lam;
          else if (fr.kind == 5) wc.vu[fr.idx] = lam;
        }
      }
      if (!sign_ok) {
        if (st.verbose >= 2)
          std::printf("[LazyDCA-terminal-v264] face variant=%d rejected: multiplier sign violation\n", variant);
        continue;
      }

      w = wc;
      Residuals Rc = do_scale ? residuals_original(mu) : compute_residuals(P, w, mu);
      const double fc = objective(P_in, w);
      double step_inf = 0.0;
      for (idx_t i = 0; i < n; ++i)
        step_inf = std::max(step_inf, std::fabs(w.x[i] - face_entry.x[i]));
      const double obj_rel = (fc - face_obj_before) / face_obj_scale;
      const bool strict = std::isfinite(Rc.feas_rel) && std::isfinite(Rc.stat_rel) && std::isfinite(Rc.comp_rel) &&
          Rc.feas_rel <= st.eps_feas && Rc.stat_rel <= st.eps_stat && Rc.comp_rel <= st.eps_comp;
      const bool obj_guard = std::isfinite(fc) && obj_rel <= 1e-8;

      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v264] face variant=%d mode=%s rows=%lld rank=%lld step=%.3e lin=%.3e objRel=%+.3e feas=%.3e stat=%.3e comp=%.3e strict=%s\n",
                    variant, pressure_mode ? "pressure" : "proximity",
                    (long long)nr, (long long)cod.rank(), step_inf, linres / rhssc,
                    obj_rel, Rc.feas_rel, Rc.stat_rel, Rc.comp_rel, strict ? "yes" : "no");

      if (strict && obj_guard) {
        const bool better = !face_best_strict || fc < face_best_obj - 1e-12 * face_obj_scale ||
            (std::fabs(fc - face_best_obj) <= 1e-12 * face_obj_scale &&
             Rc.stat_rel + Rc.comp_rel < face_best_R.stat_rel + face_best_R.comp_rel);
        if (better) {
          face_best = w;
          face_best_R = Rc;
          face_best_obj = fc;
          face_best_obj_rel = obj_rel;
          face_best_step = step_inf;
          face_best_linres = linres / rhssc;
          face_best_variant = variant;
          face_best_rows = nr;
          face_best_rank = (idx_t)cod.rank();
          face_best_strict = true;
        }
      }
      w = face_entry;
    }

    if (face_best_strict) {
      w = face_best;
      R = face_best_R;
      final_strict_kkt = true;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v264] active-face KKT rescue COMMIT variant=%d rows=%lld rank=%lld step=%.3e lin=%.3e objRel=%+.3e; stat %.3e->%.3e comp %.3e->%.3e\n",
                    face_best_variant, (long long)face_best_rows, (long long)face_best_rank,
                    face_best_step, face_best_linres, face_best_obj_rel,
                    face_before.stat_rel, R.stat_rel, face_before.comp_rel, R.comp_rel);
    } else {
      w = face_entry;
      R = face_before;
      if (st.verbose >= 1)
        std::printf("[LazyDCA-terminal-v264] active-face KKT rescue: no strict objective-safe candidate; original terminal point restored\n");
    }
  }

  if (final_strict_kkt && info.status != Status::SolvedSOC) {
    const std::string prior_terminal_message = info.message;
    info.status = Status::Solved;
    converged = true;
    info.message = "original-QP KKT certificate satisfied at finalization";
    if (!prior_terminal_message.empty())
      info.message += "; prior termination: " + prior_terminal_message;
    if (st.verbose >= 1)
      std::printf("[LazyDCA-terminal-v262] final strict KKT: solved (feas=%.3e stat=%.3e comp=%.3e)\n",
                  R.feas_rel, R.stat_rel, R.comp_rel);
  } else if (st.verbose >= 1 && info.iters >= st.max_iter) {
    std::printf("[LazyDCA-terminal-v262] primary iteration budget exhausted without strict KKT (feas=%.3e stat=%.3e comp=%.3e)\n",
                R.feas_rel, R.stat_rel, R.comp_rel);
  }
  info.res_stat_abs = R.stat_abs; info.res_stat_rel = R.stat_rel;
  info.res_eq_abs = R.eq_abs; info.res_ineq_abs = R.ineq_abs; info.res_feas_rel = R.feas_rel;
  info.res_comp_abs = R.comp_abs; info.res_comp_rel = R.comp_rel;
  info.mu_final = mu;
  info.rho_final = rho;
  info.rho_stability_final = rho_x;
  info.obj = objective(P_in, w);
  info.obj_scale = s_f;
  info.solve_time = elapsed();
  if (converged) {
    info.status = Status::Solved;
    info.message = "first-order KKT certified in original coordinates";
    if (st.soc_certificate) {
      // Second-order certificate on the CRITICAL CONE: measured reduced
      // curvature of Q on ker(Aeq) intersected with the active bounds and
      // active inequality rows (approximated by the active subspace).
      const double xsc = 1.0 + norm_inf(w.x);
      const double act_tol = std::sqrt(st.eps_comp) * xsc;
      std::vector<idx_t> AI, AJ; std::vector<double> AV;
      idx_t nrow = 0;
      for (idx_t j = 0; j < P.A.cols; ++j)
        for (idx_t pp = P.A.colptr[j]; pp < P.A.colptr[j + 1]; ++pp) {
          AI.push_back(P.A.rowind[pp]); AJ.push_back(j); AV.push_back(P.A.values[pp]);
        }
      nrow = p;
      if (m) { setzero(w.Cx); P.C.gaxpy(w.x.data(), w.Cx.data()); }
      // SOC_ACTIVE_C_LINEAR_V1
      // Historical code rescanned the whole sparse C matrix once per active
      // inequality: O(n_active * nnz(C)).  First assign active-row indices,
      // then scan the CSC matrix once: O(m + nnz(C)), same active subspace.
      std::vector<char> active_c_mask((size_t)m, 0);
      std::vector<idx_t> active_c_row((size_t)m, 0);

      for (idx_t i = 0; i < m; ++i) {
        const bool act =
            (w.hsl[i] && w.Cx[i] - P.cl[i] <= act_tol) ||
            (w.hsu[i] && P.cu[i] - w.Cx[i] <= act_tol);

        if (act) {
          active_c_mask[(size_t)i] = 1;
          active_c_row[(size_t)i] = nrow++;
        }
      }

      for (idx_t j = 0; j < P.C.cols; ++j) {
        for (idx_t pp = P.C.colptr[j]; pp < P.C.colptr[j + 1]; ++pp) {
          const idx_t i = P.C.rowind[pp];
          if (active_c_mask[(size_t)i]) {
            AI.push_back(active_c_row[(size_t)i]);
            AJ.push_back(j);
            AV.push_back(P.C.values[pp]);
          }
        }
      }
      for (idx_t i = 0; i < n; ++i) {
        bool act = (w.hl[i] && w.x[i] - P.l[i] <= act_tol) ||
                   (w.hu[i] && P.u[i] - w.x[i] <= act_tol);
        if (act) { AI.push_back(nrow); AJ.push_back(i); AV.push_back(1.0); nrow++; }
      }
      if (nrow >= n) {  // trivial critical subspace
        info.lambda_min_reduced = 0.0; info.soc_certified = true;
      } else {
        try {
          SparseCSC Aact = csc_from_coo(nrow, n, AI, AJ, AV);
          KerProjector pact(n, Aact, st.ordering);
          if (pact.ok()) {
            info.n_lanczos_calls++;
            auto rep = projected_lanczos_lmin(P.Q, &pact, st.lanczos_maxit, st.lanczos_tol);
            info.lambda_min_reduced = rep.lmin;
            double qn = 1.0 + P.Q.U.inf_norm();
            info.soc_certified = (rep.lmin >= -1e-7 * qn);
          }
        } catch (...) { /* certificate unavailable; keep first-order status */ }
      }
      if (info.soc_certified) { info.status = Status::SolvedSOC; info.message += " + measured critical-cone curvature certificate"; }
    }
  } else if (info.status == Status::NumericalError && info.message.empty()) {
    info.status = (info.iters >= st.max_iter) ? Status::MaxIter : Status::NumericalError;
  } else if (info.iters >= st.max_iter && info.status != Status::TimeLimit &&
             info.status != Status::Unbounded && info.status != Status::PrimalInfeasibleLike &&
             info.status != Status::FactorizationBudget &&
             info.status != Status::Solved &&
             info.status != Status::SolvedSOC &&
             info.status != Status::DCAConverged &&
             info.status != Status::DualDegenerate &&
             info.status != Status::NumericalError) {
    info.status = Status::MaxIter;
  }
  res.x = w.x; res.y = w.y; res.z = w.z;
  res.zl = w.vl; res.zu = w.vu;
  if (st.verbose) {
    static const char* exit_msg[] = {
        "Optimal solution found (first-order KKT).",
        "Optimal solution found (KKT + measured second-order certificate).",
        "Maximum number of iterations exceeded.",
        "Maximum wall-clock time exceeded.",
        "Converged to a point of local infeasibility.",
        "Numerical error encountered.",
        "Iterates diverging; problem might be unbounded.",
        "DCA objective sequence converged (stationary, dual test not required).",
        "Primal converged, dual DEGENERATE (unbounded multipliers, LICQ likely "
        "deficient) -- best feasible point returned, not a numerical failure.",
        "KKT factorization budget reached; best feasible incumbent returned."};
    std::printf("\nNumber of Iterations....: %lld\n", (long long)info.iters);
    std::printf("\n                                  (relative)               (absolute)\n");
    std::printf("Objective...............:  %+.16e   %+.16e\n", info.obj, info.obj);
    std::printf("Dual infeasibility......:   %.16e    %.16e\n", info.res_stat_rel, info.res_stat_abs);
    std::printf("Constraint violation....:   %.16e    %.16e\n", info.res_feas_rel,
                std::max(info.res_eq_abs, info.res_ineq_abs));
    std::printf("Complementarity.........:   %.16e    %.16e\n", info.res_comp_rel, info.res_comp_abs);
    std::printf("Overall NLP error.......:   %.16e    %.16e\n",
                std::max({info.res_stat_rel, info.res_feas_rel, info.res_comp_rel}),
                std::max({info.res_stat_abs, std::max(info.res_eq_abs, info.res_ineq_abs), info.res_comp_abs}));
    std::printf("\nNumber of KKT factorizations.................= %lld\n", (long long)info.n_factorizations);
    std::printf("Number of inertia corrections................= %lld\n", (long long)info.n_inertia_corrections);
    std::printf("Number of Lanczos curvature calls............= %lld\n", (long long)info.n_lanczos_calls);
    std::printf("Number of KKT factorization reuses...........= %lld\n", (long long)info.n_kkt_reuses);
    std::printf("Number of convexification atoms (nnz)........= %lld (%lld)\n",
                (long long)info.n_atoms, (long long)info.atoms_nnz);
    if (st.atom_oracle_enable) {
      std::printf("Atom oracle candidates / rejects / scalar....= %lld / %lld / %lld\n",
                  (long long)info.n_atom_candidates, (long long)info.n_atom_oracle_rejects,
                  (long long)info.n_atom_scalar_fallbacks);
      std::printf("Atom oracle retunes / rollbacks..............= %lld / %lld\n",
                  (long long)info.n_atom_retunes, (long long)info.n_atom_rollbacks);
      std::printf("Atom oracle last score/activity/lift.........= %.3e / %.3e / %.3e\n",
                  info.atom_last_score, info.atom_last_activity, info.atom_last_lift_rel);
      std::printf("Atom oracle last energy/support...............= %.6f / %.6f\n",
                  info.atom_last_energy, info.atom_last_support_frac);
    }
    std::printf("Number of DCA accepts / rejects..............= %lld / %lld\n",
                (long long)info.n_accepts, (long long)info.n_rejects);
    if (std::isfinite(info.best_feasible_obj))
      std::printf("Best feasible objective......................= %.10e\n", info.best_feasible_obj);
    std::printf("Final DCA model rho..........................= %.6e\n", info.rho_final);
    std::printf("Final factorization stability rho_x..........= %.6e\n", rho_x);
    if (std::isfinite(info.lambda_min_reduced))
      { double ymx = 0.0, vmx = 0.0, y1s = 0.0, v1s = 0.0;
        for (idx_t i = 0; i < p; ++i) { ymx = std::max(ymx, std::fabs(w.y[i])); y1s += std::fabs(w.y[i]); }
        for (idx_t i = 0; i < n; ++i) { vmx = std::max({vmx, w.vl[i], w.vu[i]}); v1s += w.vl[i] + w.vu[i]; }
        std::printf("Multiplier norms |y|_inf / |v|_inf...........= %.3e / %.3e\n", ymx, vmx);
        std::printf("Multiplier means |y|_1/p / |v|_1/n...........= %.3e / %.3e\n",
                    p ? y1s / (double)p : 0.0, v1s / (double)n); }
      std::printf("Measured critical-cone curvature.............= %.6e%s\n",
                  info.lambda_min_reduced, info.soc_certified ? "  (certified)" : "");
    std::printf("Total seconds (wall clock)...................= %.3f\n", info.solve_time);
    std::printf("\nEXIT: %s\n", exit_msg[(int)info.status]);
  }
  return res;
}

}  // namespace lazyipdca
