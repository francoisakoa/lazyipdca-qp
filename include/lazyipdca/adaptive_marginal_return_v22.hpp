#pragma once
// ADAPTIVE_MARGINAL_RETURN_V22
// AMRS v2.2 for LazyIPDCA C5.
//
// v2.1 established the correct global commit instrumentation.  v2.2 keeps
// that instrumentation and changes the SEED decision signal:
//   * event/update commits keep the v2.1 lift/effective-return cooldown;
//   * seed commits are counted at commit time but are classified only after a
//     trustworthy post-commit Lanczos round exposes the actual spectral
//     coverage C_k = max(cover_best, cover_k);
//   * the seed controller uses true marginal coverage gain dC plus lift;
//   * stopping has two independent, auditable branches:
//       RELATIVE: diminishing return versus frozen warm-up medians;
//       ABSOLUTE: persistently small physical dC and lift;
//   * unreliable/regressed Lanczos seed rounds are never scored and reset the
//     persistence windows, guarding degenerate eigenspace plateaus;
//   * no candidate ranking is changed and no atom-growth freeze is introduced.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <vector>

namespace lazyipdca {

enum class AdaptiveReturnSource : int {
  Seed = 1,
  EventOracle = 2,
  EventDirect = 3,
  Update = 4
};

enum class AdaptiveReturnAction : int {
  None = 0,
  StopSeed = 1,
  Cooldown = 2,
  Reactivate = 3
};

enum class AdaptiveSeedStopReason : int {
  None = 0,
  Relative = 1,
  Absolute = 2
};

class AdaptiveMarginalReturnController {
 public:
  struct Params {
    // Event/update controller (v2.1 semantics, retained).
    std::size_t warmup = 4;
    std::size_t window = 3;
    std::size_t confirmations = 2;
    double alpha_lift = 0.20;
    double alpha_effective = 0.20;
    double lift_min = 1e-5;
    double lift_max = 1e-3;
    double effective_min = 1e-5;
    double effective_max = 2.5e-3;

    // Seed controller (v2.2): true marginal spectral coverage + lift.
    double seed_alpha_coverage = 0.20;
    double seed_alpha_lift = 0.20;
    double seed_coverage_min = 1e-5;
    double seed_coverage_max = 2.5e-3;
    double seed_lift_min = 1e-5;
    double seed_lift_max = 1e-3;

    // Absolute saturation is deliberately more persistent than the relative
    // branch.  This protects continuum/degenerate spectra from a short local
    // plateau.  These are physical gates, not learned threshold caps.
    bool seed_absolute_enable = true;
    std::size_t seed_absolute_window = 5;
    std::size_t seed_absolute_confirmations = 3;
    double seed_absolute_coverage = 2.5e-3;
    double seed_absolute_lift = 1e-3;

    // Reversible post-seed event throttling, unchanged from v2.1.
    double cooldown_factor = 2.0;
    double cooldown_max_factor = 8.0;
    double strong_factor = 4.0;
  };

  struct Decision {
    bool observed = false;
    bool scored = false;
    bool references_ready = false;       // event/update references
    bool low_return = false;             // event/update or any seed branch
    bool strong_return = false;
    AdaptiveReturnAction action = AdaptiveReturnAction::None;
    AdaptiveReturnSource source = AdaptiveReturnSource::Update;
    std::size_t atom_count = 0;
    std::size_t confirmations = 0;
    double lift_rel = 0.0;
    double effective_rel = 0.0;
    double lift_ref = 0.0;
    double effective_ref = 0.0;
    double lift_threshold = 0.0;
    double effective_threshold = 0.0;
    double recent_lift_median = 0.0;
    double recent_effective_median = 0.0;
    double cooldown_multiplier = 1.0;

    // v2.2 seed decision fields.
    bool seed_round = false;
    bool seed_reliable = false;
    bool seed_references_ready = false;
    bool seed_low_relative = false;
    bool seed_low_absolute = false;
    AdaptiveSeedStopReason seed_stop_reason = AdaptiveSeedStopReason::None;
    std::size_t seed_relative_confirmations = 0;
    std::size_t seed_absolute_confirmations = 0;
    double seed_coverage = 0.0;
    double seed_coverage_gain = 0.0;
    double seed_lift_rel = 0.0;
    double seed_coverage_ref = 0.0;
    double seed_lift_ref = 0.0;
    double seed_coverage_threshold = 0.0;
    double seed_lift_threshold = 0.0;
    double seed_recent_coverage_median = 0.0;
    double seed_recent_lift_median = 0.0;
    double seed_absolute_recent_coverage_median = 0.0;
    double seed_absolute_recent_lift_median = 0.0;
  };

  struct Stats {
    std::size_t commits_total = 0;
    std::size_t scored_observations = 0;
    std::size_t seed_commits = 0;
    std::size_t event_oracle_commits = 0;
    std::size_t event_direct_commits = 0;
    std::size_t update_commits = 0;
    std::size_t low_return_rounds = 0;
    std::size_t seed_stop_requests = 0;
    std::size_t cooldowns = 0;
    std::size_t reactivations = 0;
    std::size_t model_resets = 0;
    std::size_t last_atom_count = 0;
    int last_source_code = 0;

    // Event/update v2.1 telemetry.
    bool references_ready = false;
    double lift_ref = 0.0;
    double effective_ref = 0.0;
    double lift_threshold = 0.0;
    double effective_threshold = 0.0;
    double last_lift_rel = 0.0;
    double last_effective_rel = 0.0;
    double recent_lift_median = 0.0;
    double recent_effective_median = 0.0;
    double cooldown_multiplier = 1.0;

    // Seed v2.2 telemetry.
    std::size_t seed_scored_rounds = 0;
    std::size_t seed_unreliable_rounds = 0;
    std::size_t seed_low_relative_rounds = 0;
    std::size_t seed_low_absolute_rounds = 0;
    std::size_t seed_relative_confirmations = 0;
    std::size_t seed_absolute_confirmations = 0;
    int seed_last_stop_reason = 0;
    bool seed_references_ready = false;
    double seed_last_coverage = 0.0;
    double seed_last_coverage_gain = 0.0;
    double seed_last_lift_rel = 0.0;
    double seed_coverage_ref = 0.0;
    double seed_lift_ref = 0.0;
    double seed_coverage_threshold = 0.0;
    double seed_lift_threshold = 0.0;
    double seed_recent_coverage_median = 0.0;
    double seed_recent_lift_median = 0.0;
    double seed_absolute_recent_coverage_median = 0.0;
    double seed_absolute_recent_lift_median = 0.0;
    double seed_stop_coverage = 0.0;
    double seed_stop_coverage_gain = 0.0;
    double seed_stop_lift_rel = 0.0;
  };

  AdaptiveMarginalReturnController() : p_(sanitize(Params{})) {}
  explicit AdaptiveMarginalReturnController(Params p) : p_(sanitize(p)) {}

  void clear() {
    st_ = Stats{};
    event_confirmations_ = 0;
    seed_relative_confirmations_ = 0;
    seed_absolute_confirmations_ = 0;
    warm_event_lift_.clear();
    warm_event_effective_.clear();
    recent_event_lift_.clear();
    recent_event_effective_.clear();
    warm_seed_coverage_.clear();
    warm_seed_lift_.clear();
    recent_seed_coverage_.clear();
    recent_seed_lift_.clear();
    absolute_seed_coverage_.clear();
    absolute_seed_lift_.clear();
  }

  // Reset learned model-relative state after a complete atom rollback while
  // preserving cumulative audit counters.
  void reset_model() {
    st_.model_resets++;

    st_.references_ready = false;
    st_.lift_ref = st_.effective_ref = 0.0;
    st_.lift_threshold = st_.effective_threshold = 0.0;
    st_.last_lift_rel = st_.last_effective_rel = 0.0;
    st_.recent_lift_median = st_.recent_effective_median = 0.0;
    st_.cooldown_multiplier = 1.0;
    event_confirmations_ = 0;
    warm_event_lift_.clear();
    warm_event_effective_.clear();
    recent_event_lift_.clear();
    recent_event_effective_.clear();

    st_.seed_references_ready = false;
    st_.seed_last_coverage = st_.seed_last_coverage_gain = 0.0;
    st_.seed_last_lift_rel = 0.0;
    st_.seed_coverage_ref = st_.seed_lift_ref = 0.0;
    st_.seed_coverage_threshold = st_.seed_lift_threshold = 0.0;
    st_.seed_recent_coverage_median = st_.seed_recent_lift_median = 0.0;
    st_.seed_absolute_recent_coverage_median = 0.0;
    st_.seed_absolute_recent_lift_median = 0.0;
    st_.seed_relative_confirmations = 0;
    st_.seed_absolute_confirmations = 0;
    st_.seed_last_stop_reason = 0;
    st_.seed_stop_coverage = st_.seed_stop_coverage_gain = 0.0;
    st_.seed_stop_lift_rel = 0.0;
    seed_relative_confirmations_ = 0;
    seed_absolute_confirmations_ = 0;
    warm_seed_coverage_.clear();
    warm_seed_lift_.clear();
    recent_seed_coverage_.clear();
    recent_seed_lift_.clear();
    absolute_seed_coverage_.clear();
    absolute_seed_lift_.clear();
  }

  const Stats& stats() const { return st_; }

  // Global commit hook.  Seed commits are COUNTED here, but deliberately not
  // classified: true coverage is only trustworthy in the later seed Lanczos
  // round.  Event/update commits retain v2.1 behavior.
  Decision observe_scored(double lift_rel,
                          double effective_rel,
                          std::size_t atom_count,
                          AdaptiveReturnSource source) {
    Decision d;
    d.observed = true;
    d.source = source;
    d.atom_count = atom_count;
    count_commit(source, atom_count);

    lift_rel = finite_nonneg(lift_rel);
    effective_rel = std::min(finite_nonneg(effective_rel), lift_rel);
    st_.last_lift_rel = lift_rel;
    st_.last_effective_rel = effective_rel;
    st_.last_source_code = static_cast<int>(source);

    if (is_seed(source)) {
      // Await observe_seed_round().
      d.scored = false;
      return fill(d);
    }

    d.scored = true;
    st_.scored_observations++;

    if (!st_.references_ready) {
      warm_event_lift_.push_back(lift_rel);
      warm_event_effective_.push_back(effective_rel);
      if (warm_event_lift_.size() >= p_.warmup) finalize_event_refs();
      return fill(d);
    }

    push(recent_event_lift_, lift_rel, p_.window);
    push(recent_event_effective_, effective_rel, p_.window);
    if (recent_event_lift_.size() < p_.window) return fill(d);

    st_.recent_lift_median = median(recent_event_lift_);
    st_.recent_effective_median = median(recent_event_effective_);

    d.strong_return =
        lift_rel > p_.strong_factor * st_.lift_threshold ||
        effective_rel > p_.strong_factor * st_.effective_threshold;

    d.low_return =
        !d.strong_return &&
        lift_rel <= st_.lift_threshold &&
        effective_rel <= st_.effective_threshold &&
        st_.recent_lift_median <= st_.lift_threshold &&
        st_.recent_effective_median <= st_.effective_threshold;

    if (d.strong_return) {
      event_confirmations_ = 0;
      if (st_.cooldown_multiplier > 1.0 + 1e-12) {
        st_.cooldown_multiplier =
            std::max(1.0, st_.cooldown_multiplier / p_.cooldown_factor);
        st_.reactivations++;
        d.action = AdaptiveReturnAction::Reactivate;
      }
      return fill(d);
    }

    if (!d.low_return) {
      event_confirmations_ = 0;
      return fill(d);
    }

    st_.low_return_rounds++;
    event_confirmations_++;
    if (event_confirmations_ < p_.confirmations) return fill(d);

    event_confirmations_ = 0;
    const double old = st_.cooldown_multiplier;
    st_.cooldown_multiplier = std::min(
        p_.cooldown_max_factor,
        std::max(p_.cooldown_factor, old * p_.cooldown_factor));
    if (st_.cooldown_multiplier > old + 1e-12) st_.cooldowns++;
    d.action = AdaptiveReturnAction::Cooldown;
    return fill(d);
  }

  // Trustworthy seed round.  `coverage` is the native C5 spectral coverage
  // after the committed atom and post-commit Lanczos recertification.
  Decision observe_seed_round(double coverage,
                              double lift_rel,
                              std::size_t atom_count) {
    Decision d;
    d.observed = true;
    d.scored = true;
    d.seed_round = true;
    d.seed_reliable = true;
    d.source = AdaptiveReturnSource::Seed;
    d.atom_count = atom_count;

    st_.scored_observations++;
    st_.seed_scored_rounds++;
    st_.last_atom_count = atom_count;
    st_.last_source_code = static_cast<int>(AdaptiveReturnSource::Seed);

    coverage = std::min(1.0, finite_nonneg(coverage));
    // Coverage is a progress variable: do not let noisy small backtracking
    // create a negative marginal return.
    coverage = std::max(coverage, st_.seed_last_coverage);
    const double dcov = std::max(0.0, coverage - st_.seed_last_coverage);
    lift_rel = finite_nonneg(lift_rel);

    st_.seed_last_coverage = coverage;
    st_.seed_last_coverage_gain = dcov;
    st_.seed_last_lift_rel = lift_rel;
    st_.last_lift_rel = lift_rel;

    if (!st_.seed_references_ready) {
      warm_seed_coverage_.push_back(dcov);
      warm_seed_lift_.push_back(lift_rel);
      if (warm_seed_coverage_.size() >= p_.warmup) finalize_seed_refs();
      return fill(d);
    }

    push(recent_seed_coverage_, dcov, p_.window);
    push(recent_seed_lift_, lift_rel, p_.window);
    push(absolute_seed_coverage_, dcov, p_.seed_absolute_window);
    push(absolute_seed_lift_, lift_rel, p_.seed_absolute_window);

    const bool rel_ready = recent_seed_coverage_.size() >= p_.window;
    const bool abs_ready = p_.seed_absolute_enable &&
        absolute_seed_coverage_.size() >= p_.seed_absolute_window;

    if (rel_ready) {
      st_.seed_recent_coverage_median = median(recent_seed_coverage_);
      st_.seed_recent_lift_median = median(recent_seed_lift_);
    }
    if (abs_ready) {
      st_.seed_absolute_recent_coverage_median = median(absolute_seed_coverage_);
      st_.seed_absolute_recent_lift_median = median(absolute_seed_lift_);
    }

    d.seed_low_relative = rel_ready &&
        dcov <= st_.seed_coverage_threshold &&
        lift_rel <= st_.seed_lift_threshold &&
        st_.seed_recent_coverage_median <= st_.seed_coverage_threshold &&
        st_.seed_recent_lift_median <= st_.seed_lift_threshold;

    d.seed_low_absolute = abs_ready &&
        dcov <= p_.seed_absolute_coverage &&
        lift_rel <= p_.seed_absolute_lift &&
        st_.seed_absolute_recent_coverage_median <= p_.seed_absolute_coverage &&
        st_.seed_absolute_recent_lift_median <= p_.seed_absolute_lift;

    if (d.seed_low_relative) {
      st_.seed_low_relative_rounds++;
      seed_relative_confirmations_++;
    } else {
      seed_relative_confirmations_ = 0;
    }

    if (d.seed_low_absolute) {
      st_.seed_low_absolute_rounds++;
      seed_absolute_confirmations_++;
    } else {
      seed_absolute_confirmations_ = 0;
    }

    st_.seed_relative_confirmations = seed_relative_confirmations_;
    st_.seed_absolute_confirmations = seed_absolute_confirmations_;
    d.low_return = d.seed_low_relative || d.seed_low_absolute;
    if (d.low_return) st_.low_return_rounds++;

    AdaptiveSeedStopReason reason = AdaptiveSeedStopReason::None;
    if (seed_relative_confirmations_ >= p_.confirmations)
      reason = AdaptiveSeedStopReason::Relative;
    else if (p_.seed_absolute_enable &&
             seed_absolute_confirmations_ >= p_.seed_absolute_confirmations)
      reason = AdaptiveSeedStopReason::Absolute;

    if (reason != AdaptiveSeedStopReason::None) {
      st_.seed_stop_requests++;
      st_.seed_last_stop_reason = static_cast<int>(reason);
      st_.seed_stop_coverage = coverage;
      st_.seed_stop_coverage_gain = dcov;
      st_.seed_stop_lift_rel = lift_rel;
      d.seed_stop_reason = reason;
      d.action = AdaptiveReturnAction::StopSeed;
    }

    return fill(d);
  }

  // A regressed/noisy seed Lanczos reading is explicitly non-evidence.  It
  // breaks persistence so short plateaus separated by unreliable rounds do
  // not masquerade as sustained saturation.
  Decision observe_seed_unreliable(std::size_t atom_count) {
    Decision d;
    d.observed = true;
    d.scored = false;
    d.seed_round = true;
    d.seed_reliable = false;
    d.source = AdaptiveReturnSource::Seed;
    d.atom_count = atom_count;
    st_.seed_unreliable_rounds++;
    st_.last_atom_count = atom_count;
    st_.last_source_code = static_cast<int>(AdaptiveReturnSource::Seed);
    seed_relative_confirmations_ = 0;
    seed_absolute_confirmations_ = 0;
    st_.seed_relative_confirmations = 0;
    st_.seed_absolute_confirmations = 0;
    recent_seed_coverage_.clear();
    recent_seed_lift_.clear();
    absolute_seed_coverage_.clear();
    absolute_seed_lift_.clear();
    return fill(d);
  }

  // Direct A/B/C event directions have no oracle post-commit score.  Count
  // them, never call them low-return, and relax one cooldown level.
  Decision observe_unscored(std::size_t atom_count,
                            AdaptiveReturnSource source,
                            bool strong_evidence = true) {
    Decision d;
    d.observed = true;
    d.scored = false;
    d.source = source;
    d.atom_count = atom_count;
    count_commit(source, atom_count);
    st_.last_source_code = static_cast<int>(source);
    event_confirmations_ = 0;
    if (strong_evidence && !is_seed(source) &&
        st_.cooldown_multiplier > 1.0 + 1e-12) {
      st_.cooldown_multiplier =
          std::max(1.0, st_.cooldown_multiplier / p_.cooldown_factor);
      st_.reactivations++;
      d.strong_return = true;
      d.action = AdaptiveReturnAction::Reactivate;
    }
    return fill(d);
  }

  std::size_t event_spacing(std::size_t base) const {
    base = std::max<std::size_t>(1, base);
    const double raw = std::ceil(static_cast<double>(base) *
                                 std::max(1.0, st_.cooldown_multiplier));
    const double cap =
        static_cast<double>(std::numeric_limits<std::size_t>::max());
    return static_cast<std::size_t>(std::min(raw, cap));
  }

 private:
  Params p_;
  Stats st_;
  std::size_t event_confirmations_ = 0;
  std::size_t seed_relative_confirmations_ = 0;
  std::size_t seed_absolute_confirmations_ = 0;

  std::vector<double> warm_event_lift_, warm_event_effective_;
  std::deque<double> recent_event_lift_, recent_event_effective_;
  std::vector<double> warm_seed_coverage_, warm_seed_lift_;
  std::deque<double> recent_seed_coverage_, recent_seed_lift_;
  std::deque<double> absolute_seed_coverage_, absolute_seed_lift_;

  static bool is_seed(AdaptiveReturnSource s) {
    return s == AdaptiveReturnSource::Seed;
  }

  void count_commit(AdaptiveReturnSource source, std::size_t atom_count) {
    st_.commits_total++;
    st_.last_atom_count = atom_count;
    switch (source) {
      case AdaptiveReturnSource::Seed: st_.seed_commits++; break;
      case AdaptiveReturnSource::EventOracle: st_.event_oracle_commits++; break;
      case AdaptiveReturnSource::EventDirect: st_.event_direct_commits++; break;
      case AdaptiveReturnSource::Update: st_.update_commits++; break;
    }
  }

  static double finite_nonneg(double x) {
    return std::isfinite(x) ? std::max(0.0, x) : 0.0;
  }
  static double clampv(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }
  static double median(std::vector<double> x) {
    if (x.empty()) return 0.0;
    const std::size_t n = x.size(), m = n / 2;
    std::nth_element(x.begin(), x.begin() + m, x.end());
    const double hi = x[m];
    if (n & 1u) return hi;
    std::nth_element(x.begin(), x.begin() + m - 1, x.begin() + m);
    return 0.5 * (x[m - 1] + hi);
  }
  static double median(const std::deque<double>& q) {
    return median(std::vector<double>(q.begin(), q.end()));
  }
  static void push(std::deque<double>& q, double x, std::size_t w) {
    q.push_back(x);
    while (q.size() > w) q.pop_front();
  }

  static Params sanitize(Params p) {
    p.warmup = std::max<std::size_t>(1, p.warmup);
    p.window = std::max<std::size_t>(1, p.window);
    p.confirmations = std::max<std::size_t>(1, p.confirmations);

    p.alpha_lift = clampv(finite_nonneg(p.alpha_lift), 0.0, 1.0);
    p.alpha_effective = clampv(finite_nonneg(p.alpha_effective), 0.0, 1.0);
    p.lift_min = finite_nonneg(p.lift_min);
    p.lift_max = finite_nonneg(p.lift_max);
    p.effective_min = finite_nonneg(p.effective_min);
    p.effective_max = finite_nonneg(p.effective_max);
    if (p.lift_max < p.lift_min) std::swap(p.lift_max, p.lift_min);
    if (p.effective_max < p.effective_min)
      std::swap(p.effective_max, p.effective_min);

    p.seed_alpha_coverage =
        clampv(finite_nonneg(p.seed_alpha_coverage), 0.0, 1.0);
    p.seed_alpha_lift = clampv(finite_nonneg(p.seed_alpha_lift), 0.0, 1.0);
    p.seed_coverage_min = finite_nonneg(p.seed_coverage_min);
    p.seed_coverage_max = finite_nonneg(p.seed_coverage_max);
    p.seed_lift_min = finite_nonneg(p.seed_lift_min);
    p.seed_lift_max = finite_nonneg(p.seed_lift_max);
    if (p.seed_coverage_max < p.seed_coverage_min)
      std::swap(p.seed_coverage_max, p.seed_coverage_min);
    if (p.seed_lift_max < p.seed_lift_min)
      std::swap(p.seed_lift_max, p.seed_lift_min);
    p.seed_absolute_window = std::max<std::size_t>(1, p.seed_absolute_window);
    p.seed_absolute_confirmations =
        std::max<std::size_t>(1, p.seed_absolute_confirmations);
    p.seed_absolute_coverage = finite_nonneg(p.seed_absolute_coverage);
    p.seed_absolute_lift = finite_nonneg(p.seed_absolute_lift);

    p.cooldown_factor = std::max(1.0, finite_nonneg(p.cooldown_factor));
    p.cooldown_max_factor =
        std::max(p.cooldown_factor, finite_nonneg(p.cooldown_max_factor));
    p.strong_factor = std::max(1.0, finite_nonneg(p.strong_factor));
    return p;
  }

  void finalize_event_refs() {
    st_.lift_ref = median(warm_event_lift_);
    st_.effective_ref = median(warm_event_effective_);
    st_.lift_threshold =
        clampv(p_.alpha_lift * st_.lift_ref, p_.lift_min, p_.lift_max);
    st_.effective_threshold = clampv(
        p_.alpha_effective * st_.effective_ref,
        p_.effective_min, p_.effective_max);
    st_.references_ready = true;
  }

  void finalize_seed_refs() {
    st_.seed_coverage_ref = median(warm_seed_coverage_);
    st_.seed_lift_ref = median(warm_seed_lift_);
    st_.seed_coverage_threshold = clampv(
        p_.seed_alpha_coverage * st_.seed_coverage_ref,
        p_.seed_coverage_min, p_.seed_coverage_max);
    st_.seed_lift_threshold = clampv(
        p_.seed_alpha_lift * st_.seed_lift_ref,
        p_.seed_lift_min, p_.seed_lift_max);
    st_.seed_references_ready = true;
  }

  Decision fill(Decision d) const {
    d.references_ready = st_.references_ready;
    d.confirmations = event_confirmations_;
    d.lift_rel = st_.last_lift_rel;
    d.effective_rel = st_.last_effective_rel;
    d.lift_ref = st_.lift_ref;
    d.effective_ref = st_.effective_ref;
    d.lift_threshold = st_.lift_threshold;
    d.effective_threshold = st_.effective_threshold;
    d.recent_lift_median = st_.recent_lift_median;
    d.recent_effective_median = st_.recent_effective_median;
    d.cooldown_multiplier = st_.cooldown_multiplier;

    d.seed_references_ready = st_.seed_references_ready;
    d.seed_relative_confirmations = seed_relative_confirmations_;
    d.seed_absolute_confirmations = seed_absolute_confirmations_;
    d.seed_coverage = st_.seed_last_coverage;
    d.seed_coverage_gain = st_.seed_last_coverage_gain;
    d.seed_lift_rel = st_.seed_last_lift_rel;
    d.seed_coverage_ref = st_.seed_coverage_ref;
    d.seed_lift_ref = st_.seed_lift_ref;
    d.seed_coverage_threshold = st_.seed_coverage_threshold;
    d.seed_lift_threshold = st_.seed_lift_threshold;
    d.seed_recent_coverage_median = st_.seed_recent_coverage_median;
    d.seed_recent_lift_median = st_.seed_recent_lift_median;
    d.seed_absolute_recent_coverage_median =
        st_.seed_absolute_recent_coverage_median;
    d.seed_absolute_recent_lift_median =
        st_.seed_absolute_recent_lift_median;
    return d;
  }
};

}  // namespace lazyipdca
