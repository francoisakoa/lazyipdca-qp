#pragma once

#include "atoms.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace lazyipdca {

enum class DcaMomentumMode : int {
  Original = 0,
  InertialEuclidean = 1,
  InertialGlobal = 2,
  InertialAtomic = 3
};

enum class DcaAcceptanceMode : int {
  Monotone = 0,
  Window = 1,
  Lyapunov = 2
};

enum class DcaEuclideanGammaMode : int {
  Fixed = 0,
  MetricRayleigh = 1
};

inline const char* momentum_mode_name(DcaMomentumMode m) {
  switch (m) {
    case DcaMomentumMode::Original: return "original";
    case DcaMomentumMode::InertialEuclidean: return "inertiel_euclidien";
    case DcaMomentumMode::InertialGlobal: return "inertiel_global";
    case DcaMomentumMode::InertialAtomic: return "inertiel_atomique";
  }
  return "original";
}

inline const char* euclidean_gamma_mode_name(DcaEuclideanGammaMode m) {
  switch (m) {
    case DcaEuclideanGammaMode::Fixed: return "fixed";
    case DcaEuclideanGammaMode::MetricRayleigh: return "rayleigh";
  }
  return "fixed";
}

inline const char* acceptance_mode_name(DcaAcceptanceMode m) {
  switch (m) {
    case DcaAcceptanceMode::Monotone: return "monotone";
    case DcaAcceptanceMode::Window: return "window";
    case DcaAcceptanceMode::Lyapunov: return "lyapunov";
  }
  return "monotone";
}

inline double dca_metric_quad(const std::vector<double>& v, double rho,
                              const AtomSet& atoms) {
  double q = std::max(0.0, rho) * dot(v, v);
  q += atoms.quad(v.data());
  return std::max(0.0, q);
}

// Build p_k = J_k v_k.  H_k = sum_j beta_j u_j u_j^T and
// M_k = rho_k I + H_k.
//
// Original:          p = 0
// Euclidean:         p = theta * gamma * v
// Global metric:     p = theta * M_k v
// Atom-wise:         p = theta*rho_factor*rho*v
//                       + sum_j theta_j beta_j u_j(u_j^T v),
//                    theta_j = theta [a0 + (1-a0)|cos(u_j,v)|^power].
//
// With 0<=rho_factor<=1, 0<=a0<=1 and theta<=theta_bar, the atom-wise
// operator satisfies 0 <= J_k <= theta_bar M_k without requiring the atoms
// to be orthogonal.
inline void build_dca_momentum_force(
    DcaMomentumMode mode,
    const std::vector<double>& v,
    double rho,
    const AtomSet& atoms,
    double theta,
    double gamma,
    double rho_factor,
    double atom_align_floor,
    double atom_align_power,
    std::vector<double>& p) {
  const idx_t n = static_cast<idx_t>(v.size());
  p.assign(n, 0.0);
  theta = std::max(0.0, theta);
  if (mode == DcaMomentumMode::Original || theta == 0.0 || n == 0) return;

  if (mode == DcaMomentumMode::InertialEuclidean) {
    const double c = theta * std::max(0.0, gamma);
    for (idx_t i = 0; i < n; ++i) p[i] = c * v[i];
    return;
  }

  if (mode == DcaMomentumMode::InertialGlobal) {
    const double c = theta * std::max(0.0, rho);
    for (idx_t i = 0; i < n; ++i) p[i] = c * v[i];
    std::vector<double> Hv(n, 0.0);
    atoms.apply(v.data(), Hv.data());
    for (idx_t i = 0; i < n; ++i) p[i] += theta * Hv[i];
    return;
  }

  const double rf = std::clamp(rho_factor, 0.0, 1.0);
  const double af = std::clamp(atom_align_floor, 0.0, 1.0);
  const double ap = std::max(0.0, atom_align_power);
  const double nv = std::sqrt(std::max(0.0, dot(v, v)));
  const double cr = theta * rf * std::max(0.0, rho);
  for (idx_t i = 0; i < n; ++i) p[i] = cr * v[i];
  if (nv <= 1e-300) return;

  for (idx_t j = 0; j < atoms.size(); ++j) {
    const double uv = atoms.dot_atom(j, v.data());
    const double align = std::clamp(std::fabs(uv) / nv, 0.0, 1.0);
    const double weight = af + (1.0 - af) * std::pow(align, ap);
    const double c = theta * weight * atoms.beta[j] * uv;
    if (c == 0.0) continue;
    for (size_t k = 0; k < atoms.idx[j].size(); ++k)
      p[atoms.idx[j][k]] += atoms.val[j][k] * c;
  }
}

}  // namespace lazyipdca
