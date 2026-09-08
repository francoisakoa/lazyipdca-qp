// lazyipdca — sparse.hpp
// Minimal, fast CSC sparse matrix with OpenMP kernels.
// From-scratch implementation (no Eigen dependency in the core).
#pragma once
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lazyipdca {

using idx_t = int64_t;
constexpr double INF = std::numeric_limits<double>::infinity();

// Compressed Sparse Column matrix (0-based).
struct SparseCSC {
  idx_t rows = 0, cols = 0;
  std::vector<idx_t> colptr;   // size cols+1
  std::vector<idx_t> rowind;   // size nnz
  std::vector<double> values;  // size nnz

  SparseCSC() = default;
  SparseCSC(idx_t r, idx_t c) : rows(r), cols(c), colptr(c + 1, 0) {}
  idx_t nnz() const { return (idx_t)rowind.size(); }
  bool empty() const { return rows == 0 || cols == 0 || nnz() == 0; }

  // y += alpha * M * x   (parallel over columns with atomic-free row split is
  // costly; we parallelize the transpose product instead and keep A*x serial
  // per column but SIMD.  For large m,n both kernels below are provided.)
  void gaxpy(const double* x, double* y, double alpha = 1.0) const {
    for (idx_t j = 0; j < cols; ++j) {
      const double xj = alpha * x[j];
      if (xj == 0.0) continue;
      for (idx_t p = colptr[j]; p < colptr[j + 1]; ++p) y[rowind[p]] += values[p] * xj;
    }
  }

  // y += alpha * M^T * x  (embarrassingly parallel over columns)
  void gaxpy_t(const double* x, double* y, double alpha = 1.0) const {
#pragma omp parallel for schedule(static)
    for (idx_t j = 0; j < cols; ++j) {
      double acc = 0.0;
      for (idx_t p = colptr[j]; p < colptr[j + 1]; ++p) acc += values[p] * x[rowind[p]];
      y[j] += alpha * acc;
    }
  }

  double inf_norm() const {
    double v = 0.0;
    for (double a : values) v = std::max(v, std::fabs(a));
    return v;
  }
};

// Build CSC from COO triplets (duplicates are summed).
SparseCSC csc_from_coo(idx_t rows, idx_t cols,
                       const std::vector<idx_t>& I,
                       const std::vector<idx_t>& J,
                       const std::vector<double>& V);

// Symmetric matrix stored as upper triangle in CSC (col-major, i<=j).
// gaxpy_sym computes y += alpha*(U + U^T - diag(U))*x.
struct SparseSymUpper {
  SparseCSC U;  // upper triangle including diagonal
  idx_t n() const { return U.cols; }

  void gaxpy_sym(const double* x, double* y, double alpha = 1.0) const {
    const idx_t nn = U.cols;
    for (idx_t j = 0; j < nn; ++j) {
      const double xj = x[j];
      double acc = 0.0;
      for (idx_t p = U.colptr[j]; p < U.colptr[j + 1]; ++p) {
        const idx_t i = U.rowind[p];
        const double v = U.values[p];
        if (i == j) { acc += v * xj; }
        else { y[i] += alpha * v * xj; acc += v * x[i]; }
      }
      y[j] += alpha * acc;
    }
  }

  // Gershgorin lower bound on lambda_min (cheap, no factorization).
  double gershgorin_lmin() const {
    const idx_t nn = U.cols;
    std::vector<double> diag(nn, 0.0), radius(nn, 0.0);
    for (idx_t j = 0; j < nn; ++j)
      for (idx_t p = U.colptr[j]; p < U.colptr[j + 1]; ++p) {
        const idx_t i = U.rowind[p];
        if (i == j) diag[j] += U.values[p];
        else { const double a = std::fabs(U.values[p]); radius[i] += a; radius[j] += a; }
      }
    double lmin = INF;
    for (idx_t j = 0; j < nn; ++j) lmin = std::min(lmin, diag[j] - radius[j]);
    return (nn == 0) ? 0.0 : lmin;
  }
};

// Dense vector helpers (OpenMP where it pays off).
inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  const idx_t n = (idx_t)a.size();
#pragma omp parallel for reduction(+ : s) schedule(static)
  for (idx_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
inline double norm_inf(const std::vector<double>& a) {
  double s = 0.0;
  for (double v : a) s = std::max(s, std::fabs(v));
  return s;
}
inline double norm2(const std::vector<double>& a) { return std::sqrt(std::max(0.0, dot(a, a))); }
inline void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
  const idx_t n = (idx_t)x.size();
#pragma omp parallel for schedule(static)
  for (idx_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}
inline void scal(double alpha, std::vector<double>& x) { for (double& v : x) v *= alpha; }
inline void setzero(std::vector<double>& x) { std::fill(x.begin(), x.end(), 0.0); }

}  // namespace lazyipdca
