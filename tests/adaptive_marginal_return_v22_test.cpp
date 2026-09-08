#include "adaptive_marginal_return_v22.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace lazyipdca;

static void commit_seed(AdaptiveMarginalReturnController& c, std::size_t k,
                        double lift=9e-4) {
  auto d=c.observe_scored(lift, 0.5*lift, k, AdaptiveReturnSource::Seed);
  assert(d.observed && !d.scored && d.action==AdaptiveReturnAction::None);
}

int main() {
  {
    AdaptiveMarginalReturnController c;
    commit_seed(c,1); commit_seed(c,2);
    auto s=c.stats();
    assert(s.seed_commits==2 && s.seed_scored_rounds==0);
  }
  {
    // Relative seed branch: warm-up with large gains, then persistently tiny
    // gain AND tiny lift.  Relative stop must eventually fire.
    AdaptiveMarginalReturnController c;
    double cov=0.0;
    for (std::size_t k=1;k<=4;k++) {
      commit_seed(c,k,2e-2); cov += 5e-2;
      c.observe_seed_round(cov,2e-2,k);
    }
    assert(c.stats().seed_references_ready);
    AdaptiveReturnAction a=AdaptiveReturnAction::None;
    for (std::size_t k=5;k<=10;k++) {
      commit_seed(c,k,1e-4); cov += 1e-4;
      auto d=c.observe_seed_round(cov,1e-4,k); a=d.action;
      if (a==AdaptiveReturnAction::StopSeed) {
        assert(d.seed_stop_reason==AdaptiveSeedStopReason::Relative);
        break;
      }
    }
    assert(a==AdaptiveReturnAction::StopSeed);
  }
  {
    // Absolute branch: QPNBAND-like regime. Lift stays near its warm-up
    // reference (therefore NOT relatively low) but absolute dC and lift are
    // persistently small.  The longer 5-sample/3-confirmation branch stops it.
    AdaptiveMarginalReturnController::Params p;
    AdaptiveMarginalReturnController c(p);
    double cov=0.0;
    for (std::size_t k=1;k<=4;k++) {
      commit_seed(c,k,9e-4); cov += 1e-2;
      c.observe_seed_round(cov,9e-4,k);
    }
    auto s=c.stats();
    assert(s.seed_lift_ref>8e-4 && s.seed_lift_threshold<3e-4);
    AdaptiveSeedStopReason reason=AdaptiveSeedStopReason::None;
    for (std::size_t k=5;k<=20;k++) {
      commit_seed(c,k,9e-4); cov += 1e-4;
      auto d=c.observe_seed_round(cov,9e-4,k);
      if (d.action==AdaptiveReturnAction::StopSeed) {
        reason=d.seed_stop_reason; break;
      }
    }
    assert(reason==AdaptiveSeedStopReason::Absolute);
    assert(c.stats().seed_low_relative_rounds==0);
    assert(c.stats().seed_low_absolute_rounds>=3);
  }
  {
    // An unreliable seed round breaks absolute persistence and clears rolling
    // windows, preventing separated plateaus from being concatenated.
    AdaptiveMarginalReturnController c;
    double cov=0.0;
    for (std::size_t k=1;k<=4;k++) {
      commit_seed(c,k,9e-4); cov += 1e-2;
      c.observe_seed_round(cov,9e-4,k);
    }
    for (std::size_t k=5;k<=9;k++) {
      commit_seed(c,k,9e-4); cov += 1e-4;
      c.observe_seed_round(cov,9e-4,k);
    }
    commit_seed(c,10,9e-4);
    c.observe_seed_unreliable(10);
    auto s=c.stats();
    assert(s.seed_unreliable_rounds==1);
    assert(s.seed_relative_confirmations==0);
    assert(s.seed_absolute_confirmations==0);
    commit_seed(c,11,9e-4); cov += 1e-4;
    auto d=c.observe_seed_round(cov,9e-4,11);
    assert(d.action==AdaptiveReturnAction::None);
  }
  {
    // Event controller remains reversible: build references, enter cooldown,
    // then a strong event relaxes it.
    AdaptiveMarginalReturnController c;
    for (std::size_t k=1;k<=4;k++)
      c.observe_scored(1e-1,5e-2,k,AdaptiveReturnSource::EventOracle);
    AdaptiveReturnAction act=AdaptiveReturnAction::None;
    for (std::size_t k=5;k<=12;k++) {
      auto d=c.observe_scored(1e-5,1e-5,k,AdaptiveReturnSource::EventOracle);
      if (d.action==AdaptiveReturnAction::Cooldown) { act=d.action; break; }
    }
    assert(act==AdaptiveReturnAction::Cooldown);
    assert(c.stats().cooldown_multiplier>1.0);
    auto d=c.observe_scored(1.0,1.0,20,AdaptiveReturnSource::EventOracle);
    assert(d.action==AdaptiveReturnAction::Reactivate);
  }
  std::cout << "adaptive_marginal_return_v22_test PASS\n";
  return 0;
}
