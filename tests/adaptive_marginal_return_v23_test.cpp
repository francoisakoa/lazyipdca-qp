#include "adaptive_marginal_return_v23.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace lazyipdca;

static void seed_commit(AdaptiveMarginalReturnController& c,
                        std::size_t k,
                        double coverage,
                        double lift,
                        bool reliable,
                        AdaptiveMarginalReturnController::Decision* out=nullptr) {
  auto commit = c.observe_scored(lift, 0.5*lift, k, AdaptiveReturnSource::Seed);
  assert(commit.observed && !commit.scored);
  auto d = c.observe_seed_progress(coverage, lift, k, reliable);
  if (out) *out = d;
}

int main() {
  // 1) QPNBAND-like plateau: many noisy rounds must count toward rank/time.
  {
    AdaptiveMarginalReturnController::Params p;
    p.seed_plateau_enable = true;
    p.seed_plateau_horizon = 8;
    p.seed_plateau_min_commits = 12;
    p.seed_plateau_confirmations = 2;
    p.seed_plateau_coverage_per_commit = 2.5e-3;
    p.seed_plateau_lift = 1e-3;
    p.seed_absolute_enable = false;  // retired legacy branch
    AdaptiveMarginalReturnController c(p);

    AdaptiveMarginalReturnController::Decision last;
    bool stopped=false;
    for (std::size_t k=1;k<=30;k++) {
      // Small early progress, then essentially flat best-certified coverage.
      const double cov = (k<=4) ? 2.0e-5*k : 8.0e-5 + 2.0e-6*(k-4);
      const bool reliable = (k==1 || k==7 || k==19); // mostly noisy
      seed_commit(c,k,cov,9.0e-4,reliable,&last);
      if (last.action == AdaptiveReturnAction::StopSeed) { stopped=true; break; }
    }
    assert(stopped);
    assert(last.seed_stop_reason == AdaptiveSeedStopReason::Plateau);
    assert(c.stats().seed_progress_rounds >= 13);
    assert(c.stats().seed_unreliable_rounds > c.stats().seed_scored_rounds);
    assert(c.stats().seed_plateau_evaluations >= 2);
    assert(c.stats().seed_plateau_gain_per_commit <= 2.5e-3);
    assert(c.stats().seed_plateau_lift_median <= 1e-3);
  }

  // 2) Minimum-rank guard: no plateau stop before min_commits.
  {
    AdaptiveMarginalReturnController::Params p;
    p.seed_plateau_horizon=4;
    p.seed_plateau_min_commits=12;
    p.seed_plateau_confirmations=1;
    p.seed_plateau_coverage_per_commit=1e-2;
    p.seed_plateau_lift=1e-2;
    AdaptiveMarginalReturnController c(p);
    for (std::size_t k=1;k<12;k++) {
      AdaptiveMarginalReturnController::Decision d;
      seed_commit(c,k,0.0,1e-5,false,&d);
      assert(d.action != AdaptiveReturnAction::StopSeed);
    }
  }

  // 3) A fresh strong lift cancels plateau persistence.
  {
    AdaptiveMarginalReturnController::Params p;
    p.seed_plateau_horizon=4;
    p.seed_plateau_min_commits=6;
    p.seed_plateau_confirmations=2;
    p.seed_plateau_coverage_per_commit=1e-3;
    p.seed_plateau_lift=1e-3;
    AdaptiveMarginalReturnController c(p);
    AdaptiveMarginalReturnController::Decision d;
    for (std::size_t k=1;k<=6;k++) seed_commit(c,k,1e-5*k,5e-4,false,&d);
    assert(c.stats().seed_plateau_confirmations == 1);
    seed_commit(c,7,7e-5,5e-3,false,&d);
    assert(d.action != AdaptiveReturnAction::StopSeed);
    assert(c.stats().seed_plateau_confirmations == 0);
  }

  // 4) A fresh coverage jump cancels plateau persistence even if horizon avg is small.
  {
    AdaptiveMarginalReturnController::Params p;
    p.seed_plateau_horizon=4;
    p.seed_plateau_min_commits=6;
    p.seed_plateau_confirmations=2;
    p.seed_plateau_coverage_per_commit=1e-3;
    p.seed_plateau_lift=1e-3;
    AdaptiveMarginalReturnController c(p);
    AdaptiveMarginalReturnController::Decision d;
    for (std::size_t k=1;k<=6;k++) seed_commit(c,k,1e-5*k,5e-4,false,&d);
    assert(c.stats().seed_plateau_confirmations == 1);
    seed_commit(c,7,1e-2,5e-4,true,&d);
    assert(d.action != AdaptiveReturnAction::StopSeed);
    assert(c.stats().seed_plateau_confirmations == 0);
  }

  // 5) Relative reliable-only branch still works; noisy rounds do not reset it.
  {
    AdaptiveMarginalReturnController::Params p;
    p.warmup=2; p.window=2; p.confirmations=2;
    p.seed_alpha_coverage=0.2; p.seed_alpha_lift=0.2;
    p.seed_coverage_min=1e-6; p.seed_coverage_max=1e-2;
    p.seed_lift_min=1e-6; p.seed_lift_max=1e-2;
    p.seed_plateau_enable=false;
    AdaptiveMarginalReturnController c(p);
    AdaptiveMarginalReturnController::Decision d;
    seed_commit(c,1,0.10,0.10,true,&d);
    seed_commit(c,2,0.20,0.10,true,&d); // refs ready
    seed_commit(c,3,0.20,1e-4,false,&d); // noisy, no reset
    seed_commit(c,4,0.2001,1e-4,true,&d);
    seed_commit(c,5,0.2002,1e-4,true,&d); // first relative low
    seed_commit(c,6,0.2003,1e-4,true,&d); // second relative low => stop
    assert(d.action==AdaptiveReturnAction::StopSeed);
    assert(d.seed_stop_reason==AdaptiveSeedStopReason::Relative);
  }

  // 6) Event cooldown/reactivation unchanged from v2.1.
  {
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

  std::cout << "adaptive_marginal_return_v23_test PASS\n";
  return 0;
}
