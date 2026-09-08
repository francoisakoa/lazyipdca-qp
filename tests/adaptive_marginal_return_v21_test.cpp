#include "adaptive_marginal_return_v21.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace lazyipdca;

int main() {
  AdaptiveMarginalReturnController::Params p;
  p.warmup = 4;
  p.window = 3;
  p.confirmations = 2;
  p.alpha_lift = 0.20;
  p.alpha_effective = 0.20;
  p.lift_min = 1e-5;
  p.lift_max = 1e-3;
  p.effective_min = 1e-5;
  p.effective_max = 2.5e-3;
  p.cooldown_factor = 2.0;
  p.cooldown_max_factor = 8.0;
  p.strong_factor = 4.0;

  // Seed: 4 useful warm-up commits, then a sustained low-return tail.
  AdaptiveMarginalReturnController c(p);
  for (std::size_t k = 1; k <= 4; ++k) {
    auto d = c.observe_scored(0.05, 0.04, k, AdaptiveReturnSource::Seed);
    assert(d.action == AdaptiveReturnAction::None);
  }
  assert(c.stats().references_ready);
  // Thresholds clip at the configured maxima.
  assert(std::fabs(c.stats().lift_threshold - 1e-3) < 1e-15);
  assert(std::fabs(c.stats().effective_threshold - 2.5e-3) < 1e-15);

  AdaptiveReturnAction a = AdaptiveReturnAction::None;
  for (std::size_t k = 5; k <= 9; ++k) {
    auto d = c.observe_scored(2e-4, 1e-4, k, AdaptiveReturnSource::Seed);
    a = d.action;
    if (a == AdaptiveReturnAction::StopSeed) break;
  }
  assert(a == AdaptiveReturnAction::StopSeed);
  assert(c.stats().seed_stop_requests == 1);

  // Post-seed low return -> reversible cooldown x2.
  AdaptiveMarginalReturnController e(p);
  for (std::size_t k = 1; k <= 4; ++k)
    e.observe_scored(0.05, 0.04, k, AdaptiveReturnSource::EventOracle);
  bool cooled = false;
  for (std::size_t k = 5; k <= 10; ++k) {
    auto d = e.observe_scored(2e-4, 1e-4, k, AdaptiveReturnSource::EventOracle);
    if (d.action == AdaptiveReturnAction::Cooldown) { cooled = true; break; }
  }
  assert(cooled);
  assert(e.stats().cooldown_multiplier == 2.0);
  assert(e.event_spacing(50) == 100);

  // A direct emergency event is unscored and reactivates one level.
  auto r = e.observe_unscored(11, AdaptiveReturnSource::EventDirect, true);
  assert(r.action == AdaptiveReturnAction::Reactivate);
  assert(e.stats().cooldown_multiplier == 1.0);
  assert(e.stats().event_direct_commits == 1);

  // Strong scored return also vetoes low-return history.
  auto s = e.observe_scored(0.02, 0.01, 12, AdaptiveReturnSource::EventOracle);
  assert(!s.low_return);

  // Model reset preserves cumulative audit counters but clears learned refs.
  const auto commits_before = e.stats().commits_total;
  e.reset_model();
  assert(e.stats().commits_total == commits_before);
  assert(!e.stats().references_ready);
  assert(e.stats().cooldown_multiplier == 1.0);
  assert(e.stats().model_resets == 1);

  std::cout << "adaptive_marginal_return_v21_test PASS\n";
  return 0;
}
