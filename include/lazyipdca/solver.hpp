// lazyipdca — solver.hpp
// IPDCA: interior-point DC algorithm for large-scale nonconvex sparse QP.
// The DC-model metric (rho I + H_k) is kept conceptually separate from
// numerical KKT stabilization rho_x.  Controlled CONST/DYN/lazy variants
// support the OMS ablation; DYN checks full-space curvature before committing
// smaller model shifts, while KKT inertia remains an independent numerical
// diagnostic.  An optional inertial DCA layer can be combined with
// monotone, windowed non-monotone, or Lyapunov acceptance.  The default
// (original + monotone) is the historical algorithm.  KKT residuals are
// measured in ORIGINAL coordinates.
#pragma once
#include "problem.hpp"

namespace lazyipdca {

Result solve(const Problem& prob, const Settings& settings);

}  // namespace lazyipdca
