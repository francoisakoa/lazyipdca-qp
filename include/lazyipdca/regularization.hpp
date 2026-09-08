#pragma once

#include <algorithm>
#include <cmath>

namespace lazyipdca {

// Controlled regularization variants used in the OMS ablation.
// Legacy preserves the historical v0.17.1 behaviour and independent toggles.
// Const/Dyn/Lazy correspond to the article's CONST/DYN/lazy variants.
enum class DcRegularizationMode : int {
  Legacy = 0,
  Const = 1,   // H_k == 0, spectral rho_0, model rho frozen for the whole run
  Dyn = 2,     // H_k == 0, spectral rho_0, model rho may adapt
  Lazy = 3     // adaptive scalar rho + low-rank H_k
};

inline const char* regularization_mode_name(DcRegularizationMode mode) {
  switch (mode) {
    case DcRegularizationMode::Legacy: return "legacy";
    case DcRegularizationMode::Const: return "const";
    case DcRegularizationMode::Dyn: return "dyn";
    case DcRegularizationMode::Lazy: return "lazy";
  }
  return "legacy";
}

// Source used to compute the initial scalar shift in CONST/DYN.
enum class RhoInitMethod : int {
  Auto = 0,        // external hint if available, else Lanczos, else Gershgorin
  User = 1,        // use rho0 directly
  Lanczos = 2,     // full-space Lanczos on Q, fall back to Gershgorin
  Gershgorin = 3,  // guaranteed full-space lower bound of Q
  Hint = 4         // lambda_min_hint, fall back to Auto when absent
};

inline const char* rho_init_method_name(RhoInitMethod method) {
  switch (method) {
    case RhoInitMethod::Auto: return "auto";
    case RhoInitMethod::User: return "user";
    case RhoInitMethod::Lanczos: return "lanczos";
    case RhoInitMethod::Gershgorin: return "gershgorin";
    case RhoInitMethod::Hint: return "hint";
  }
  return "auto";
}


// Numerical guard used by DYN around a Lanczos leftmost-Ritz estimate.
// If the Ritz data are unusable, return the caller's fallback lower estimate
// (typically Gershgorin).  This is a numerical safeguard; it is not advertised
// as a rigorous global eigenvalue lower bound.
inline double guarded_lanczos_lmin(double lambda_hat, double residual,
                                   double residual_safety,
                                   double fallback_lmin) {
  if (!std::isfinite(lambda_hat)) return fallback_lmin;
  if (!std::isfinite(residual)) return lambda_hat;
  return lambda_hat - std::max(0.0, residual_safety) * std::fabs(residual);
}

inline bool rho_candidate_passes_curvature(double guarded_lmin, double rho,
                                            double tol = 1e-8) {
  return std::isfinite(guarded_lmin) && std::isfinite(rho) && rho >= 0.0 &&
         guarded_lmin + rho >= -std::max(0.0, tol);
}

// Formula used in the OMS fixed-shift baseline:
// rho_0 = max(0,-lambda_hat) * (1 + rel_margin) + abs_margin.
inline double spectral_shift_from_lmin(double lambda_hat,
                                       double rel_margin = 1e-6,
                                       double abs_margin = 1e-8) {
  const double neg = std::max(0.0, -lambda_hat);
  return neg * (1.0 + std::max(0.0, rel_margin)) + std::max(0.0, abs_margin);
}

// DYN adapts only the excess scalar regularization above a certified DC floor.
inline double rho_dyn_decay_candidate(double rho, double dc_floor, double decay) {
  const double d = std::clamp(decay, 0.0, 1.0);
  return std::max(std::max(0.0, dc_floor), std::max(0.0, rho) * d);
}

inline double rho_dyn_growth_candidate(double rho, double dc_floor,
                                       double growth, double abs_max) {
  const double g = std::max(1.0, growth);
  const double hi = std::max(0.0, abs_max);
  const double cand = std::max(std::max(0.0, dc_floor), std::max(0.0, rho) * g);
  return std::min(cand, hi);
}

}  // namespace lazyipdca
