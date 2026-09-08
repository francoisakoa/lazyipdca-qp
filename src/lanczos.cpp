// lazyipdca — projected Lanczos: estimate lambda_min of Z'QZ, Z = ker(A) basis,
// matrix-free.  Projection P = I - A'(AA')^+A applied via one MUMPS solve on
// the (regularized) augmented projector  [ I  A' ; A  -delta I ].
#include "lazyipdca/sparse.hpp"
#include "lazyipdca/mumps_solver.hpp"
#include "lazyipdca/lanczos.hpp"
#include <random>
#include <cstring>

extern "C" {
// LAPACK: eigenvalues/vectors of symmetric tridiagonal
void dstev_(const char* jobz, const int* n, double* d, double* e, double* z,
            const int* ldz, double* work, int* info);
}

namespace lazyipdca {

SparseCSC csc_from_coo(idx_t rows, idx_t cols, const std::vector<idx_t>& I,
                       const std::vector<idx_t>& J, const std::vector<double>& V) {
  SparseCSC M(rows, cols);
  const idx_t nz = (idx_t)I.size();
  std::vector<idx_t> count(cols, 0);
  for (idx_t k = 0; k < nz; ++k) count[J[k]]++;
  M.colptr[0] = 0;
  for (idx_t j = 0; j < cols; ++j) M.colptr[j + 1] = M.colptr[j] + count[j];
  M.rowind.assign(nz, 0); M.values.assign(nz, 0.0);
  std::vector<idx_t> next(M.colptr.begin(), M.colptr.end() - 1);
  for (idx_t k = 0; k < nz; ++k) {
    idx_t p = next[J[k]]++;
    M.rowind[p] = I[k]; M.values[p] = V[k];
  }
  // sort rows within columns and sum duplicates
  std::vector<idx_t> ri; std::vector<double> vv; std::vector<idx_t> newptr(cols + 1, 0);
  ri.reserve(nz); vv.reserve(nz);
  for (idx_t j = 0; j < cols; ++j) {
    idx_t lo = M.colptr[j], hi = M.colptr[j + 1];
    std::vector<std::pair<idx_t, double>> col;
    col.reserve(hi - lo);
    for (idx_t p = lo; p < hi; ++p) col.emplace_back(M.rowind[p], M.values[p]);
    std::sort(col.begin(), col.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t t = 0; t < col.size(); ++t) {
      if (!ri.empty() && (idx_t)ri.size() > newptr[j] && ri.back() == col[t].first)
        vv.back() += col[t].second;
      else { ri.push_back(col[t].first); vv.push_back(col[t].second); }
    }
    newptr[j + 1] = (idx_t)ri.size();
  }
  M.colptr = std::move(newptr); M.rowind = std::move(ri); M.values = std::move(vv);
  return M;
}

// Build & factorize the projector matrix once; apply P via solve.
KerProjector::KerProjector(idx_t n, const SparseCSC& A, int ordering, double delta)
    : n_(n), p_(A.rows) {
  if (p_ == 0) return;  // identity projection
  std::vector<int> irn, jcn;
  const idx_t N = n_ + p_;
  irn.reserve(n_ + A.nnz() + p_); jcn.reserve(n_ + A.nnz() + p_);
  vals_.reserve(n_ + A.nnz() + p_);
  for (idx_t i = 0; i < n_; ++i) { irn.push_back((int)i + 1); jcn.push_back((int)i + 1); vals_.push_back(1.0); }
  for (idx_t j = 0; j < A.cols; ++j)
    for (idx_t pp = A.colptr[j]; pp < A.colptr[j + 1]; ++pp) {
      // A' block: entry (row=j in x-block, col=n+A.rowind[pp]) upper triangle
      irn.push_back((int)j + 1);
      jcn.push_back((int)(n_ + A.rowind[pp]) + 1);
      vals_.push_back(A.values[pp]);
    }
  for (idx_t i = 0; i < p_; ++i) { irn.push_back((int)(n_ + i) + 1); jcn.push_back((int)(n_ + i) + 1); vals_.push_back(-delta); }
  mumps_ = std::make_unique<MumpsSym>();
  mumps_->set_matrix(N, std::move(irn), std::move(jcn), (idx_t)vals_.size(), ordering);
  mumps_->analyze();
  ok_ = mumps_->factorize(vals_.data());
}
KerProjector::~KerProjector() = default;

void KerProjector::apply(const double* v, double* out) const {
  if (p_ == 0) { std::memcpy(out, v, sizeof(double) * (size_t)n_); return; }
  std::vector<double> rhs(n_ + p_, 0.0);
  std::memcpy(rhs.data(), v, sizeof(double) * (size_t)n_);
  mumps_->solve(rhs);
  std::memcpy(out, rhs.data(), sizeof(double) * (size_t)n_);
}

// Lanczos with full reorthogonalization on op(v) = P Q P v.
LanczosReport projected_lanczos_lmin(const SparseSymUpper& Q, const KerProjector* Pp,
                                     idx_t maxit, double tol, unsigned seed,
                                     const AtomSet* atoms,
                                     std::vector<std::vector<double>>* ritz_out, idx_t ritz_k) {
  LanczosReport rep;
  const idx_t n = Q.n();
  if (n == 0) { rep.lmin = 0.0; rep.converged = true; return rep; }
  maxit = std::min<idx_t>(maxit, n);
  std::mt19937 gen(seed);
  std::normal_distribution<double> nd(0.0, 1.0);

  std::vector<std::vector<double>> V;
  std::vector<double> alpha, beta;
  std::vector<double> v(n), w(n), t(n);
  for (auto& x : v) x = nd(gen);
  if (Pp) { Pp->apply(v.data(), w.data()); v = w; }  // start in ker(A)
  double nv = norm2(v);
  if (nv < 1e-30) { rep.lmin = 0.0; rep.converged = true; return rep; }
  scal(1.0 / nv, v);
  V.push_back(v);

  double lmin_prev = INF;
  for (idx_t k = 0; k < maxit; ++k) {
    setzero(t); Q.gaxpy_sym(V[k].data(), t.data());
    if (atoms) atoms->apply(V[k].data(), t.data());  // model operator Q + H_k
    if (Pp) Pp->apply(t.data(), w.data());         // w = P (Q+H) v_k (v_k in ker)
    else w = t;                                    // espace complet (defaut zip)
    double a = dot(w, V[k]); alpha.push_back(a);
    axpy(-a, V[k], w);
    if (k > 0) axpy(-beta[k - 1], V[k - 1], w);
    for (const auto& vi : V) { double c = dot(w, vi); if (std::fabs(c) > 0) axpy(-c, vi, w); } // full reorth
    double bnext = norm2(w);

    // eigen of T_k
    int kk = (int)alpha.size(), info = 0, ldz = kk;
    std::vector<double> d(alpha), e(beta), Zm((size_t)kk * kk), work(std::max(1, 2 * kk - 2));
    dstev_("V", &kk, d.data(), e.data(), Zm.data(), &ldz, work.data(), &info);
    if (info == 0) {
      rep.lmin = d[0];
      rep.ritz_low.assign(d.begin(), d.begin() + std::min<size_t>(10, d.size()));
      double resid = bnext * std::fabs(Zm[(size_t)(kk - 1)]);  // |beta * last comp of eigvec 0|
      rep.residual = resid;
      if (resid <= tol * std::max(1.0, std::fabs(rep.lmin)) ||
          std::fabs(rep.lmin - lmin_prev) <= tol * std::max(1.0, std::fabs(rep.lmin))) {
        rep.converged = true; rep.iters = k + 1;
        if (ritz_out) {
          const idx_t nk = std::min<idx_t>(ritz_k, kk);
          ritz_out->assign(nk, std::vector<double>(n, 0.0));
          for (idx_t jr = 0; jr < nk; ++jr)
            for (int i = 0; i <= (int)k; ++i)
              axpy(Zm[(size_t)jr * kk + i], V[i], (*ritz_out)[jr]);
        }
        return rep;
      }
      lmin_prev = rep.lmin;
    }
    if (bnext < 1e-14) {
      rep.converged = true; rep.iters = k + 1;
      if (ritz_out && !alpha.empty()) {
        int kk2 = (int)alpha.size(), info = 0, ldz = kk2;
        std::vector<double> d(alpha), e(beta), Zm2((size_t)kk2 * kk2), work(std::max(1, 2 * kk2 - 2));
        dstev_("V", &kk2, d.data(), e.data(), Zm2.data(), &ldz, work.data(), &info);
        if (info == 0) {
          const idx_t nk = std::min<idx_t>(ritz_k, kk2);
          ritz_out->assign(nk, std::vector<double>(n, 0.0));
          for (idx_t jr = 0; jr < nk; ++jr)
            for (int i = 0; i < kk2; ++i) axpy(Zm2[(size_t)jr * kk2 + i], V[i], (*ritz_out)[jr]);
        }
      }
      return rep;
    }
    beta.push_back(bnext);
    scal(1.0 / bnext, w);
    V.push_back(w);
  }
  rep.iters = maxit;
  if (ritz_out && !alpha.empty()) {
    int kk2 = (int)alpha.size(), info = 0, ldz = kk2;
    std::vector<double> d(alpha), e(beta); e.resize(std::max(0, kk2 - 1));
    std::vector<double> Zm2((size_t)kk2 * kk2), work(std::max(1, 2 * kk2 - 2));
    dstev_("V", &kk2, d.data(), e.data(), Zm2.data(), &ldz, work.data(), &info);
    if (info == 0) {
      const idx_t nk = std::min<idx_t>(ritz_k, kk2);
      ritz_out->assign(nk, std::vector<double>(n, 0.0));
      for (idx_t jr = 0; jr < nk; ++jr)
        for (int i = 0; i < kk2; ++i) axpy(Zm2[(size_t)jr * kk2 + i], V[i], (*ritz_out)[jr]);
    }
  }
  return rep;
}

}  // namespace lazyipdca
