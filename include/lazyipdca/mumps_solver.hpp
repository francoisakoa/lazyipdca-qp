// lazyipdca — mumps_solver.hpp
// RAII wrapper around sequential MUMPS (dmumps) for symmetric indefinite
// systems, with METIS ordering and inertia extraction (INFOG(12)).
#pragma once
#include "sparse.hpp"
#include <dmumps_c.h>
#include <vector>
#include <stdexcept>

namespace lazyipdca {

class MumpsSym {
 public:
  MumpsSym();
  ~MumpsSym();
  MumpsSym(const MumpsSym&) = delete;
  MumpsSym& operator=(const MumpsSym&) = delete;

  // One triangle (upper) of the symmetric matrix in 0-based COO.
  // Duplicate entries are summed by MUMPS (used for diagonal updates).
  // ordering: MUMPS ICNTL(7) (5 = METIS).
  void set_matrix(idx_t n, std::vector<int>&& irn, std::vector<int>&& jcn,
                  idx_t nnz, int ordering, double pivot_tol = 1e-6);
  void analyze();                       // symbolic (once per pattern)
  // values must have the same length/order as the COO pattern
  bool factorize(const double* values); // numeric; false on failure
  void solve(std::vector<double>& rhs); // in-place solve

  int neg_pivots() const { return id_.infog[11]; }   // INFOG(12)
  int null_pivots() const { return id_.infog[27]; }  // INFOG(28)
  int info1() const { return infog1_; }
  idx_t n() const { return n_; }

 private:
  DMUMPS_STRUC_C id_;
  idx_t n_ = 0, nnz_ = 0;
  std::vector<int> irn_, jcn_;
  std::vector<double> a_;
  bool analyzed_ = false;
  int infog1_ = 0;
};

}  // namespace lazyipdca
