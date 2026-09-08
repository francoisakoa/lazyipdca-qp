#pragma once
// Backend QDLDL : le noyau LDL^T C nu d'OSQP (Stellato et al., Apache-2.0),
// dont derive le LDL de PIQP (lignee Timothy A. Davis). Sans pivotage
// (quasi-defini, Vanderbei), inertie exacte par les signes de D, permutation
// METIS calculee par nous. Interface identique a MumpsSym/LdlSym.
#include "qdldl.h"
#include <metis.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>

using lazyipdca_midx = ::idx_t;   // idx_t de metis.h, avant tout shadowing

namespace lazyipdca {

class QdldlSym {
public:
  using ll = long long;
  void set_matrix(ll N, std::vector<int>&& irn, std::vector<int>&& jcn,
                  ll nnz, int, double) {
    N_ = (QDLDL_int)N; irn_ = std::move(irn); jcn_ = std::move(jcn); nnz_ = (size_t)nnz;
  }
  bool analyze() {
    const int n = (int)N_;
    perm_.assign(n, 0); iperm_.assign(n, 0);
    { std::vector<std::vector<int>> adj(n);
      for (size_t k = 0; k < nnz_; ++k) {
        int i = irn_[k] - 1, j = jcn_[k] - 1;
        if (i != j) { adj[i].push_back(j); adj[j].push_back(i); }
      }
      std::vector<lazyipdca_midx> xadj(n + 1, 0), adjncy;
      for (int i = 0; i < n; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        xadj[i + 1] = xadj[i] + (lazyipdca_midx)adj[i].size();
      }
      adjncy.reserve(xadj[n]);
      for (int i = 0; i < n; ++i) for (int v : adj[i]) adjncy.push_back(v);
      std::vector<lazyipdca_midx> p(n), ip(n); lazyipdca_midx nv = n;
      if (xadj[n] == 0 || METIS_NodeND(&nv, xadj.data(), adjncy.data(), nullptr,
                                       nullptr, p.data(), ip.data()) != METIS_OK)
        for (int i = 0; i < n; ++i) { perm_[i] = i; iperm_[i] = i; }
      else for (int i = 0; i < n; ++i) { perm_[i] = (int)p[i]; iperm_[i] = (int)ip[i]; }
    }
    // CSC superieur permute, doublons fusionnes, mapping COO->CSC
    std::vector<std::pair<std::pair<int,int>, int>> ent(nnz_);
    for (size_t k = 0; k < nnz_; ++k) {
      int i = iperm_[irn_[k] - 1], j = iperm_[jcn_[k] - 1];
      if (i > j) std::swap(i, j);
      ent[k] = {{j, i}, (int)k};
    }
    std::sort(ent.begin(), ent.end());
    Ap_.assign(N_ + 1, 0); Ai_.clear(); map_.assign(nnz_, -1);
    int pj = -1, pi = -1;
    for (auto& e : ent) {
      int j = e.first.first, i = e.first.second;
      if (j != pj || i != pi) { Ai_.push_back(i); Ap_[j + 1]++; pj = j; pi = i; }
      map_[e.second] = (int)Ai_.size() - 1;
    }
    for (QDLDL_int j = 0; j < N_; ++j) Ap_[j + 1] += Ap_[j];
    Ax_.assign(Ai_.size(), 0.0);
    // etree + tailles
    etree_.assign(N_, 0); Lnz_.assign(N_, 0); iwork_.assign(3 * (size_t)N_, 0);
    QDLDL_int sum = QDLDL_etree(N_, Ap_.data(), Ai_.data(), iwork_.data(),
                                Lnz_.data(), etree_.data());
    if (sum < 0) return false;
    Lp_.assign(N_ + 1, 0); Li_.assign(sum, 0); Lx_.assign(sum, 0.0);
    D_.assign(N_, 0.0); Dinv_.assign(N_, 0.0);
    bwork_.assign(N_, 0); fwork_.assign(N_, 0.0);
    return true;
  }
  bool factorize(const double* vals) {
    std::fill(Ax_.begin(), Ax_.end(), 0.0);
    for (size_t k = 0; k < nnz_; ++k) Ax_[map_[k]] += vals[k];
    QDLDL_int pos = QDLDL_factor(N_, Ap_.data(), Ai_.data(), Ax_.data(),
                                 Lp_.data(), Li_.data(), Lx_.data(),
                                 D_.data(), Dinv_.data(), Lnz_.data(),
                                 etree_.data(), bwork_.data(), iwork_.data(),
                                 fwork_.data());
    if (pos < 0) { neg_ = -1; return false; }   // zero pivot QDLDL
    double dmax = 0.0;
    for (QDLDL_int i = 0; i < N_; ++i) dmax = std::max(dmax, std::fabs(D_[i]));
    for (QDLDL_int i = 0; i < N_; ++i)
      if (!std::isfinite(D_[i]) || D_[i] == 0.0) { neg_ = -1; return false; }
    neg_ = (int)(N_ - pos);
    return true;
  }
  void solve(double* rhs, int nrhs = 1) {
    std::vector<QDLDL_float> bp(N_);
    for (int r = 0; r < nrhs; ++r) {
      double* b = rhs + (size_t)r * N_;
      for (QDLDL_int i = 0; i < N_; ++i) bp[iperm_[i]] = b[i];
      QDLDL_solve(N_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(), bp.data());
      for (QDLDL_int i = 0; i < N_; ++i) b[i] = bp[iperm_[i]];
    }
  }
  int neg_pivots() const { return neg_; }
private:
  QDLDL_int N_ = 0; size_t nnz_ = 0;
  std::vector<int> irn_, jcn_, perm_, iperm_, map_;
  std::vector<QDLDL_int> Ap_, Ai_, Lp_, Li_, etree_, Lnz_, iwork_;
  std::vector<QDLDL_float> Ax_, Lx_, D_, Dinv_, fwork_;
  std::vector<QDLDL_bool> bwork_;
  int neg_ = -1;
};

} // namespace lazyipdca
