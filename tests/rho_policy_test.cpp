#include "lazyipdca/regularization.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace lazyipdca;

int main() {
  {
    const double r = spectral_shift_from_lmin(-2.0, 1e-6, 1e-8);
    const double expected = 2.0 * (1.0 + 1e-6) + 1e-8;
    assert(std::fabs(r - expected) <= 1e-14 * std::max(1.0, expected));
  }
  assert(std::fabs(spectral_shift_from_lmin(3.0, 1e-6, 1e-8) - 1e-8) < 1e-18);
  assert(std::strcmp(regularization_mode_name(DcRegularizationMode::Const), "const") == 0);
  assert(std::strcmp(regularization_mode_name(DcRegularizationMode::Dyn), "dyn") == 0);
  assert(std::strcmp(regularization_mode_name(DcRegularizationMode::Lazy), "lazy") == 0);
  assert(std::strcmp(rho_init_method_name(RhoInitMethod::Gershgorin), "gershgorin") == 0);
  {
    const double g = guarded_lanczos_lmin(-2.0, 0.05, 2.0, -10.0);
    assert(std::fabs(g + 2.1) < 1e-15);
    assert(rho_candidate_passes_curvature(g, 2.10000001, 1e-8));
    assert(!rho_candidate_passes_curvature(g, 2.0, 1e-8));
  }
  assert(guarded_lanczos_lmin(INFINITY, 1.0, 1.0, -7.0) == -7.0);
  // DYN controller: decay/growth act only above the certified floor.
  assert(std::fabs(rho_dyn_decay_candidate(100.0, 30.0, 0.5) - 50.0) < 1e-12);
  assert(std::fabs(rho_dyn_decay_candidate(40.0, 30.0, 0.5) - 30.0) < 1e-12);
  assert(std::fabs(rho_dyn_growth_candidate(30.0, 30.0, 2.0, 1e6) - 60.0) < 1e-12);
  assert(std::fabs(rho_dyn_growth_candidate(60.0, 30.0, 2.0, 100.0) - 100.0) < 1e-12);

  std::cout << "rho_policy_test: PASS\n";
  return 0;
}
