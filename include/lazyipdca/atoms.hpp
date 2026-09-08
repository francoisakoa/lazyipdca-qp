// lazyipdca — atoms.hpp
// Low-rank lazy convexification H_k = sum_j beta_j u_j u_j^T  (PSD), ported
// from the reference LazyDCA (piqp-lazydca-19av): atoms are sparsified Ritz
// directions of negative reduced curvature; they convexify ONLY the offending
// modes so that the uniform rho_eff can stay small (bigger Newton steps).
// Zip defaults honoured: single Ritz candidate per addition (legacy path),
// post-sparsification Rayleigh acceptance, orthogonality filter.
#pragma once
#include "sparse.hpp"

namespace lazyipdca {

struct AtomSet {
  std::vector<std::vector<idx_t>> idx;   // support of each atom (sorted)
  std::vector<std::vector<double>> val;  // values (unit 2-norm after sparsify)
  std::vector<double> beta;              // weights (> 0)

  idx_t size() const { return (idx_t)beta.size(); }
  idx_t nnz() const {
    idx_t s = 0;
    for (const auto& v : idx) s += (idx_t)v.size();
    return s;
  }
  double dot_atom(idx_t j, const double* x) const {
    double s = 0.0;
    for (size_t k = 0; k < idx[j].size(); ++k) s += val[j][k] * x[idx[j][k]];
    return s;
  }
  // y += (sum_j beta_j u_j u_j^T) x
  void apply(const double* x, double* y) const {
    for (idx_t j = 0; j < size(); ++j) {
      const double s = beta[j] * dot_atom(j, x);
      if (s == 0.0) continue;
      for (size_t k = 0; k < idx[j].size(); ++k) y[idx[j][k]] += val[j][k] * s;
    }
  }
  // quadratic form d^T H d
  double quad(const double* d) const {
    double s = 0.0;
    for (idx_t j = 0; j < size(); ++j) { double t = dot_atom(j, d); s += beta[j] * t * t; }
    return s;
  }
  // |cos| of a dense candidate (unit norm) against atom j
  double abs_cos(idx_t j, const double* v_unit) const { return std::fabs(dot_atom(j, v_unit)); }

  // Sparsify a dense unit direction: keep the max_nnz largest-|.| entries,
  // renormalise.  Returns false if degenerate.
  static bool sparsify(const std::vector<double>& v, idx_t max_nnz,
                       std::vector<idx_t>& out_idx, std::vector<double>& out_val) {
    const idx_t n = (idx_t)v.size();
    std::vector<idx_t> ord(n);
    for (idx_t i = 0; i < n; ++i) ord[i] = i;
    const idx_t keep = std::min(max_nnz, n);
    std::partial_sort(ord.begin(), ord.begin() + keep, ord.end(),
                      [&](idx_t a, idx_t b) { return std::fabs(v[a]) > std::fabs(v[b]); });
    out_idx.assign(ord.begin(), ord.begin() + keep);
    std::sort(out_idx.begin(), out_idx.end());
    out_val.resize(keep);
    double nrm = 0.0;
    for (idx_t k = 0; k < keep; ++k) { out_val[k] = v[out_idx[k]]; nrm += out_val[k] * out_val[k]; }
    nrm = std::sqrt(nrm);
    if (nrm < 1e-14) return false;
    for (double& x : out_val) x /= nrm;
    return true;
  }

  // Energy-aware sparsification used by the adaptive atom oracle. Keep the
  // SMALLEST Top-|v| support whose squared-norm energy reaches target_energy,
  // subject to [min_nnz,max_nnz], then renormalise.  This removes numerically
  // dense Ritz tails that add H-couplings without material spectral content.
  static bool sparsify_energy(const std::vector<double>& v, idx_t max_nnz,
                              double target_energy, idx_t min_nnz,
                              std::vector<idx_t>& out_idx,
                              std::vector<double>& out_val,
                              double* retained_energy = nullptr) {
    const idx_t n = (idx_t)v.size();
    if (n <= 0 || max_nnz <= 0) return false;
    const idx_t cap = std::min(std::max<idx_t>(1, max_nnz), n);
    const idx_t mink = std::min(std::max<idx_t>(1, min_nnz), cap);
    const double tgt = std::clamp(target_energy, 0.0, 1.0);
    std::vector<idx_t> ord(n);
    double total = 0.0;
    for (idx_t i = 0; i < n; ++i) { ord[i] = i; total += v[i] * v[i]; }
    if (!(total > 1e-28)) return false;
    std::partial_sort(ord.begin(), ord.begin() + cap, ord.end(),
                      [&](idx_t a, idx_t b) { return std::fabs(v[a]) > std::fabs(v[b]); });
    double kept = 0.0; idx_t keep = 0;
    for (; keep < cap; ++keep) {
      kept += v[ord[keep]] * v[ord[keep]];
      const idx_t k1 = keep + 1;
      if (k1 >= mink && kept / total >= tgt) { keep = k1; break; }
    }
    if (keep <= 0) keep = mink;
    if (keep > cap) keep = cap;
    // If the loop ended naturally at cap, `keep` is cap-1 before increment.
    if (kept / total < tgt) keep = cap;
    out_idx.assign(ord.begin(), ord.begin() + keep);
    std::sort(out_idx.begin(), out_idx.end());
    out_val.resize(keep);
    double nrm2 = 0.0;
    for (idx_t k = 0; k < keep; ++k) {
      out_val[k] = v[out_idx[k]]; nrm2 += out_val[k] * out_val[k];
    }
    const double nrm = std::sqrt(nrm2);
    if (!(nrm > 1e-14)) return false;
    for (double& x : out_val) x /= nrm;
    if (retained_energy) *retained_energy = nrm2 / total;
    return true;
  }
};

}  // namespace lazyipdca
