#pragma once
// Backend LDL^T mince (Eigen SimplicialLDLT, simplicial, sans pivotage) pour
// les KKT quasi-definis de taille petite/moyenne, ou les frais fixes
// multifrontaux de MUMPS dominent. Theoreme de Vanderbei : une matrice
// quasi-definie est fortement factorisable en LDL^T sans pivotage, et les
// signes de D donnent l'inertie exacte — notre chaine de certification
// (escalade rho_x, decroissance certifiee du rho modele) est inchangee.
// Interface identique a MumpsSym : set_matrix (COO 1-base, triangle
// superieur, doublons sommes) / analyze / factorize(vals) / solve / neg_pivots.

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <metis.h>
#include <algorithm>

namespace lazyipdca {

class LdlSym {
public:
  using idx_t = long long;

  void set_matrix(idx_t N, std::vector<int>&& irn, std::vector<int>&& jcn,
                  idx_t nnz, int /*ordering*/, double /*pivot_tol*/) {
    N_ = (int)N; irn_ = std::move(irn); jcn_ = std::move(jcn); nnz_ = (size_t)nnz;
    analyzed_ = false;
  }

  bool analyze() {
    // permutation METIS (nested dissection) : l'AMD d'Eigen explose le
    // remplissage sur les graphes de couplage type Gould-Toint ; on permute
    // nous-memes et on factorise en ordre NATUREL.
    perm_.assign(N_, 0); iperm_.assign(N_, 0);
    {
      std::vector<std::vector<int>> adj(N_);
      for (size_t k = 0; k < nnz_; ++k) {
        int i = irn_[k] - 1, j = jcn_[k] - 1;
        if (i != j) { adj[i].push_back(j); adj[j].push_back(i); }
      }
      std::vector<idx_t_m> xadj(N_ + 1, 0), adjncy;
      for (int i = 0; i < N_; ++i) {
        std::sort(adj[i].begin(), adj[i].end());
        adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
        xadj[i + 1] = xadj[i] + (idx_t_m)adj[i].size();
      }
      adjncy.reserve(xadj[N_]);
      for (int i = 0; i < N_; ++i) for (int v2 : adj[i]) adjncy.push_back(v2);
      std::vector<idx_t_m> p(N_), ip(N_);
      idx_t_m nv = N_;
      if (xadj[N_] == 0 ||
          METIS_NodeND(&nv, xadj.data(), adjncy.data(), nullptr, nullptr,
                       p.data(), ip.data()) != METIS_OK) {
        for (int i = 0; i < N_; ++i) { perm_[i] = i; iperm_[i] = i; }
      } else {
        for (int i = 0; i < N_; ++i) { perm_[i] = (int)p[i]; iperm_[i] = (int)ip[i]; }
      }
    }
    // pattern permute avec valeurs factices (doublons sommes par Eigen)
    std::vector<Eigen::Triplet<double>> T; T.reserve(nnz_);
    for (size_t k = 0; k < nnz_; ++k) {
      int i = iperm_[irn_[k] - 1], j = iperm_[jcn_[k] - 1];
      if (i > j) std::swap(i, j);                // rester triangle superieur
      T.emplace_back(i, j, 1.0);
    }
    M_.resize(N_, N_); M_.setFromTriplets(T.begin(), T.end());
    M_.makeCompressed();
    // mapping COO k -> position dans le stockage compresse (pour factorize)
    map_.assign(nnz_, -1);
    for (size_t k = 0; k < nnz_; ++k) {
      int i = iperm_[irn_[k] - 1], j = iperm_[jcn_[k] - 1];
      if (i > j) std::swap(i, j);
      for (int p = (int)M_.outerIndexPtr()[j]; p < (int)M_.outerIndexPtr()[j + 1]; ++p)
        if (M_.innerIndexPtr()[p] == i) { map_[k] = p; break; }
    }
    ldl_.analyzePattern(M_);
    analyzed_ = true;
    return true;
  }

  bool factorize(const double* vals) {
    if (!analyzed_) analyze();
    double* v = M_.valuePtr();
    std::fill(v, v + M_.nonZeros(), 0.0);
    for (size_t k = 0; k < nnz_; ++k) {
      if (map_[k] < 0) { std::fprintf(stderr, "[LdlSym] entree %zu (%d,%d) hors pattern\n", k, irn_[k], jcn_[k]); neg_ = -1; return false; }
      v[map_[k]] += vals[k];  // doublons sommes
    }
    ldl_.factorize(M_);
    if (ldl_.info() != Eigen::Success) { neg_ = -1; return false; }
    const auto& D = ldl_.vectorD();
    neg_ = 0; double dmax = 0.0;
    for (int i = 0; i < D.size(); ++i) dmax = std::max(dmax, std::fabs(D[i]));
    bool bad = !(dmax > 0.0);
    for (int i = 0; i < D.size() && !bad; ++i) {
      if (!std::isfinite(D[i]) || D[i] == 0.0) bad = true;
      else if (D[i] < 0.0) neg_++;
    }
    if (bad) { neg_ = -1; return false; }   // breakdown : signal de courbure
    return true;
  }

  void solve(double* rhs, int nrhs = 1) {
    Eigen::VectorXd bp(N_);
    for (int r = 0; r < nrhs; ++r) {
      double* b = rhs + (size_t)r * N_;
      for (int i = 0; i < N_; ++i) bp[iperm_[i]] = b[i];
      Eigen::VectorXd x = ldl_.solve(bp);
      for (int i = 0; i < N_; ++i) b[i] = x[iperm_[i]];
    }
  }

  int neg_pivots() const { return neg_; }

private:
  int N_ = 0; size_t nnz_ = 0;
  std::vector<int> irn_, jcn_;
  std::vector<int> map_;
  Eigen::SparseMatrix<double> M_;   // triangle superieur
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper,
                        Eigen::NaturalOrdering<int>> ldl_;
  using idx_t_m = ::idx_t;   // idx_t de metis.h
  std::vector<int> perm_, iperm_;
  bool analyzed_ = false;
  int neg_ = -1;
};

} // namespace lazyipdca
