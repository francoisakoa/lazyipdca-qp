#pragma once
// INERTIA_COMPLETE_AMRS_V24
// Second-generation inertial helpers for the legacy atom-oracle integration.
// This header deliberately layers on top of inertia.hpp so the historical
// original+monotone and first-generation inertial paths remain available.

#include "lazyipdca/inertia.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace lazyipdca {

inline const char* momentum_atom_weighting_name(DcaMomentumAtomWeighting p) {
  switch (p) {
    case DcaMomentumAtomWeighting::Uniform: return "uniform";
    case DcaMomentumAtomWeighting::Cosine:  return "cosine";
    case DcaMomentumAtomWeighting::Energy:  return "energy";
  }
  return "cosine";
}

inline const char* momentum_rho_policy_name(DcaMomentumRhoPolicy p) {
  switch (p) {
    case DcaMomentumRhoPolicy::Restart:   return "restart";
    case DcaMomentumRhoPolicy::Keep:      return "keep";
    case DcaMomentumRhoPolicy::Transport: return "transport";
  }
  return "restart";
}

inline double complete_theta_cap(const Settings& st) {
  double cap = std::max(0.0, st.dca_momentum_theta_max);
  if (st.dca_acceptance == DcaAcceptanceMode::Lyapunov &&
      st.dca_lyapunov_cap_momentum_theta) {
    const double tb = st.dca_lyapunov_theta_bar > 0.0
        ? st.dca_lyapunov_theta_bar : st.dca_momentum_theta;
    cap = std::min(cap, std::max(0.0, tb));
  }
  return cap;
}

// Correct Lyapunov domination for p = theta * gamma * v:
// theta*gamma I <= theta_bar M, M >= rho I => gamma <= theta_bar*rho/theta.
inline double complete_euclidean_gamma_cap(double gamma, double theta,
                                           double rho, const Settings& st,
                                           bool* capped = nullptr) {
  if (capped) *capped = false;
  const bool enforce = st.dca_lyapunov_enforce_euclidean_bound ||
      (st.dca_acceptance == DcaAcceptanceMode::Lyapunov &&
       !st.dca_lyapunov_unsafe_euclidean);
  if (!enforce || theta <= 0.0) return std::max(0.0, gamma);
  const double tb = st.dca_lyapunov_theta_bar > 0.0
      ? st.dca_lyapunov_theta_bar : st.dca_momentum_theta;
  const double gcap = std::max(0.0, tb) * std::max(0.0, rho) / theta;
  if (gamma > gcap) {
    if (capped) *capped = true;
    return gcap;
  }
  return std::max(0.0, gamma);
}

inline double complete_transport_theta(double theta, double rho_old,
                                       double rho_new, const Settings& st,
                                       bool* capped = nullptr) {
  if (capped) *capped = false;
  if (!(rho_old > 0.0 && rho_new > 0.0)) return std::min(theta, complete_theta_cap(st));
  const double raw = theta * rho_old / rho_new;
  const double cap = complete_theta_cap(st);
  if (raw > cap && capped) *capped = true;
  return std::min(cap, std::max(0.0, raw));
}

struct CompleteAtomicWeights {
  double rho_weight = 1.0;
  std::vector<double> atom;
  double energy_rho_share = 0.0;
  double energy_atom_max_share = 0.0;
  double energy_atom_effective_count = 0.0;
  double align_mean = 0.0;
  double align_max = 0.0;
};

inline CompleteAtomicWeights complete_atomic_weights(
    DcaMomentumAtomWeighting policy,
    const std::vector<double>& v,
    const AtomSet& atoms,
    double rho,
    bool energy_weight_rho) {
  CompleteAtomicWeights out;
  out.atom.assign((size_t)atoms.size(), 1.0);
  double v2 = 0.0;
  for (double x : v) v2 += x*x;
  const double vn = std::sqrt(v2);

  if (policy == DcaMomentumAtomWeighting::Uniform) return out;

  if (policy == DcaMomentumAtomWeighting::Cosine) {
    double s = 0.0;
    for (idx_t j=0; j<atoms.size(); ++j) {
      double un2 = 0.0;
      for (double a : atoms.val[(size_t)j]) un2 += a*a;
      const double un = std::sqrt(un2);
      double w = 0.0;
      if (vn > 0.0 && un > 0.0)
        w = std::min(1.0, std::fabs(atoms.dot_atom(j, v.data()))/(un*vn));
      out.atom[(size_t)j] = w;
      s += w; out.align_max = std::max(out.align_max, w);
    }
    out.align_mean = atoms.size() ? s/(double)atoms.size() : 0.0;
    return out;
  }

  const double atom_energy = atoms.quad(v.data());
  const double total = std::max(0.0, rho)*v2 + atom_energy;
  if (!(total > 0.0)) return out;
  out.energy_rho_share = std::max(0.0, rho)*v2/total;
  out.rho_weight = energy_weight_rho ? std::sqrt(out.energy_rho_share) : 1.0;
  double sum_e2 = 0.0;
  for (idx_t j=0; j<atoms.size(); ++j) {
    const double uv = atoms.dot_atom(j, v.data());
    const double ej = std::max(0.0, atoms.beta[(size_t)j])*uv*uv;
    const double frac = ej/total;
    out.atom[(size_t)j] = std::sqrt(std::max(0.0, frac));
    out.energy_atom_max_share = std::max(out.energy_atom_max_share, frac);
    sum_e2 += ej*ej;
  }
  if (sum_e2 > 0.0 && atom_energy > 0.0)
    out.energy_atom_effective_count = atom_energy*atom_energy/sum_e2;
  return out;
}

inline void complete_atomic_force(const std::vector<double>& v,
                                  const AtomSet& atoms, double rho,
                                  double theta_rho,
                                  const std::vector<double>& theta_atom,
                                  std::vector<double>& p) {
  p.assign(v.size(), 0.0);
  for (size_t i=0; i<v.size(); ++i) p[i] = theta_rho*std::max(0.0,rho)*v[i];
  // Deliberately avoid AtomSet::apply_weighted: the AMRS v2.3 baseline is
  // first-generation inertia and does not guarantee that later helper exists.
  const idx_t na = std::min<idx_t>(atoms.size(), (idx_t)theta_atom.size());
  for (idx_t j=0; j<na; ++j) {
    const double th = theta_atom[(size_t)j];
    if (th == 0.0) continue;
    const double uv = atoms.dot_atom(j, v.data());
    const double c = th * atoms.beta[(size_t)j] * uv;
    if (c == 0.0) continue;
    for (size_t k=0; k<atoms.idx[(size_t)j].size(); ++k)
      p[(size_t)atoms.idx[(size_t)j][k]] += atoms.val[(size_t)j][k] * c;
  }
}

inline double complete_metric_force_inf(const std::vector<double>& v,
                                        const AtomSet& atoms, double rho) {
  std::vector<double> mv(v.size(), 0.0);
  for (size_t i=0; i<v.size(); ++i) mv[i] = std::max(0.0,rho)*v[i];
  atoms.apply(v.data(), mv.data());
  double z=0.0; for (double x: mv) z=std::max(z,std::fabs(x)); return z;
}

inline double complete_cap_force(std::vector<double>& p,
                                 const std::vector<double>& v,
                                 const AtomSet& atoms, double rho,
                                 double rel_max, bool* capped=nullptr) {
  if (capped) *capped=false;
  double pi=0.0; for (double x:p) pi=std::max(pi,std::fabs(x));
  if (!(rel_max > 0.0) || !(pi > 0.0)) return pi;
  const double base = complete_metric_force_inf(v, atoms, rho);
  const double cap = rel_max*std::max(1e-16, base);
  if (pi > cap) {
    const double a=cap/pi; for(double& x:p) x*=a;
    if(capped) *capped=true;
    return cap;
  }
  return pi;
}

} // namespace lazyipdca
