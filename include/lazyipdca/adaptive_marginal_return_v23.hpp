#pragma once
// ADAPTIVE_MARGINAL_RETURN_V23
// AMRS v2.3 for LazyIPDCA C5.
//
// v2.3 keeps the qualified v2.1 event controller and the v2.2 true-coverage
// seed sensor, but replaces the fragile instantaneous absolute seed branch by
// a robust plateau detector over ALL committed seed rounds:
//   * reliable rounds update the relative dCoverage/lift detector;
//   * noisy/regressed rounds are retained as elapsed committed rank for the
//     plateau detector, while the monotone best certified coverage is held;
//   * over horizon H, progress is (C_k-C_{k-H})/H and lift is the median of
//     the last H committed seed atoms;
//   * a plateau can stop seeding only after min_commits and consecutive
//     confirmations, with both low coverage progress and low lift;
//   * a fresh large coverage jump or high lift immediately resets plateau
//     persistence;
//   * v2.2 instantaneous absolute saturation is retired (legacy fields remain
//     only for backward compatibility);
//   * native C5 coverage remains sovereign and no global atom freeze is added.


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
  Absolute = 2,  // legacy v2.2 reason (not emitted by v2.3)
  Plateau = 3
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

    // v2.3 robust committed-rank plateau. Overall AMRS remains opt-in.
    bool seed_plateau_enable = true;
    std::size_t seed_plateau_horizon = 8;
    std::size_t seed_plateau_min_commits = 12;
    std::size_t seed_plateau_confirmations = 2;
    double seed_plateau_coverage_per_commit = 2.5e-3;
    double seed_plateau_lift = 1e-3;

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

    // v2.3 plateau decision fields.
    bool seed_plateau_ready = false;
    bool seed_low_plateau = false;
    std::size_t seed_plateau_confirmations = 0;
    std::size_t seed_progress_rounds = 0;
    double seed_plateau_horizon_gain = 0.0;
    double seed_plateau_gain_per_commit = 0.0;
    double seed_plateau_lift_median = 0.0;
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

    // Seed v2.3 plateau telemetry.
    std::size_t seed_progress_rounds = 0;
    std::size_t seed_plateau_evaluations = 0;
    std::size_t seed_low_plateau_rounds = 0;
    std::size_t seed_plateau_confirmations = 0;
    bool seed_plateau_ready = false;
    double seed_last_reliable_coverage = 0.0;
    double seed_plateau_horizon_gain = 0.0;
    double seed_plateau_gain_per_commit = 0.0;
    double seed_plateau_lift_median = 0.0;
    double seed_stop_plateau_gain_per_commit = 0.0;
    double seed_stop_plateau_lift_median = 0.0;
  };

  AdaptiveMarginalReturnController() : p_(sanitize(Params{})) {}
  explicit AdaptiveMarginalReturnController(Params p) : p_(sanitize(p)) {}

  void clear() {
    st_ = Stats{};
    event_confirmations_ = 0;
    seed_relative_confirmations_ = 0;
    seed_absolute_confirmations_ = 0;
    seed_plateau_confirmations_ = 0;
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
    seed_plateau_coverage_.clear();
    seed_plateau_lift_.clear();
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
    st_.seed_progress_rounds = 0;
    st_.seed_plateau_evaluations = 0;
    st_.seed_low_plateau_rounds = 0;
    st_.seed_plateau_confirmations = 0;
    st_.seed_plateau_ready = false;
    st_.seed_last_reliable_coverage = 0.0;
    st_.seed_plateau_horizon_gain = 0.0;
    st_.seed_plateau_gain_per_commit = 0.0;
    st_.seed_plateau_lift_median = 0.0;
    st_.seed_stop_plateau_gain_per_commit = 0.0;
    st_.seed_stop_plateau_lift_median = 0.0;
    seed_relative_confirmations_ = 0;
    seed_absolute_confirmations_ = 0;
    seed_plateau_confirmations_ = 0;
    warm_seed_coverage_.clear();
    warm_seed_lift_.clear();
    recent_seed_coverage_.clear();
    recent_seed_lift_.clear();
    absolute_seed_coverage_.clear();
    absolute_seed_lift_.clear();
    seed_plateau_coverage_.clear();
    seed_plateau_lift_.clear();
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

  // v2.3 seed progress hook. Call exactly once after each committed seed
  // atom and its post-commit Lanczos reading. `coverage_best` is the monotone
  // best certified C5 coverage. A noisy round keeps that coverage unchanged
  // but still counts as committed rank in the plateau horizon.
  Decision observe_seed_progress(double coverage_best,
                                 double lift_rel,
                                 std::size_t atom_count,
                                 bool reliable) {
    Decision d;
    d.observed = true;
    d.scored = reliable;
    d.seed_round = true;
    d.seed_reliable = reliable;
    d.source = AdaptiveReturnSource::Seed;
    d.atom_count = atom_count;

    st_.seed_progress_rounds++;
    st_.last_atom_count = atom_count;
    st_.last_source_code = static_cast<int>(AdaptiveReturnSource::Seed);

    coverage_best = std::min(1.0, finite_nonneg(coverage_best));
    coverage_best = std::max(coverage_best, st_.seed_last_coverage);
    const double dcov_all =
        std::max(0.0, coverage_best - st_.seed_last_coverage);
    lift_rel = finite_nonneg(lift_rel);

    st_.seed_last_coverage = coverage_best;
    st_.seed_last_coverage_gain = dcov_all;
    st_.seed_last_lift_rel = lift_rel;
    st_.last_lift_rel = lift_rel;

    // Relative detector: only trustworthy recertifications update learned
    // references/windows. Noisy rounds neither fabricate zero dCoverage nor
    // clear relative persistence.
    if (reliable) {
      st_.scored_observations++;
      st_.seed_scored_rounds++;
      const double dcov_rel = std::max(
          0.0, coverage_best - st_.seed_last_reliable_coverage);
      st_.seed_last_reliable_coverage = coverage_best;

      if (!st_.seed_references_ready) {
        warm_seed_coverage_.push_back(dcov_rel);
        warm_seed_lift_.push_back(lift_rel);
        if (warm_seed_coverage_.size() >= p_.warmup) finalize_seed_refs();
      } else {
        push(recent_seed_coverage_, dcov_rel, p_.window);
        push(recent_seed_lift_, lift_rel, p_.window);
        const bool rel_ready = recent_seed_coverage_.size() >= p_.window;
        if (rel_ready) {
          st_.seed_recent_coverage_median = median(recent_seed_coverage_);
          st_.seed_recent_lift_median = median(recent_seed_lift_);
        }
        d.seed_low_relative = rel_ready &&
            dcov_rel <= st_.seed_coverage_threshold &&
            lift_rel <= st_.seed_lift_threshold &&
            st_.seed_recent_coverage_median <= st_.seed_coverage_threshold &&
            st_.seed_recent_lift_median <= st_.seed_lift_threshold;
        if (d.seed_low_relative) {
          st_.seed_low_relative_rounds++;
          seed_relative_confirmations_++;
        } else {
          seed_relative_confirmations_ = 0;
        }
      }
    } else {
      st_.seed_unreliable_rounds++;
      // Crucially: do NOT clear relative or plateau state.
    }

    // Robust plateau detector: every committed seed round contributes one
    // unit of rank/time. Keep H+1 coverage values and H lift values.
    if (p_.seed_plateau_enable) {
      seed_plateau_coverage_.push_back(coverage_best);
      while (seed_plateau_coverage_.size() > p_.seed_plateau_horizon + 1)
        seed_plateau_coverage_.pop_front();
      push(seed_plateau_lift_, lift_rel, p_.seed_plateau_horizon);

      const bool horizon_ready =
          seed_plateau_coverage_.size() >= p_.seed_plateau_horizon + 1 &&
          seed_plateau_lift_.size() >= p_.seed_plateau_horizon;
      const bool min_ready =
          st_.seed_progress_rounds >= p_.seed_plateau_min_commits;
      st_.seed_plateau_ready = horizon_ready && min_ready;

      if (st_.seed_plateau_ready) {
        st_.seed_plateau_evaluations++;
        st_.seed_plateau_horizon_gain = std::max(
            0.0, seed_plateau_coverage_.back() -
                     seed_plateau_coverage_.front());
        st_.seed_plateau_gain_per_commit =
            st_.seed_plateau_horizon_gain /
            static_cast<double>(p_.seed_plateau_horizon);
        st_.seed_plateau_lift_median = median(seed_plateau_lift_);

        d.seed_low_plateau =
            dcov_all <= p_.seed_plateau_coverage_per_commit &&
            st_.seed_plateau_gain_per_commit <=
                p_.seed_plateau_coverage_per_commit &&
            lift_rel <= p_.seed_plateau_lift &&
            st_.seed_plateau_lift_median <= p_.seed_plateau_lift;

        if (d.seed_low_plateau) {
          st_.seed_low_plateau_rounds++;
          seed_plateau_confirmations_++;
        } else {
          seed_plateau_confirmations_ = 0;
        }
      } else {
        seed_plateau_confirmations_ = 0;
      }
    } else {
      seed_plateau_confirmations_ = 0;
      st_.seed_plateau_ready = false;
    }

    st_.seed_relative_confirmations = seed_relative_confirmations_;
    // v2.2 instantaneous absolute branch is retired in v2.3.
    seed_absolute_confirmations_ = 0;
    st_.seed_absolute_confirmations = 0;
    st_.seed_plateau_confirmations = seed_plateau_confirmations_;
    d.low_return = d.seed_low_relative || d.seed_low_plateau;
    if (d.low_return) st_.low_return_rounds++;

    AdaptiveSeedStopReason reason = AdaptiveSeedStopReason::None;
    if (seed_relative_confirmations_ >= p_.confirmations)
      reason = AdaptiveSeedStopReason::Relative;
    else if (p_.seed_plateau_enable &&
             seed_plateau_confirmations_ >= p_.seed_plateau_confirmations)
      reason = AdaptiveSeedStopReason::Plateau;

    if (reason != AdaptiveSeedStopReason::None) {
      st_.seed_stop_requests++;
      st_.seed_last_stop_reason = static_cast<int>(reason);
      st_.seed_stop_coverage = coverage_best;
      st_.seed_stop_coverage_gain = dcov_all;
      st_.seed_stop_lift_rel = lift_rel;
      st_.seed_stop_plateau_gain_per_commit =
          st_.seed_plateau_gain_per_commit;
      st_.seed_stop_plateau_lift_median = st_.seed_plateau_lift_median;
      d.seed_stop_reason = reason;
      d.action = AdaptiveReturnAction::StopSeed;
    }

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
  std::size_t seed_plateau_confirmations_ = 0;

  std::vector<double> warm_event_lift_, warm_event_effective_;
  std::deque<double> recent_event_lift_, recent_event_effective_;
  std::vector<double> warm_seed_coverage_, warm_seed_lift_;
  std::deque<double> recent_seed_coverage_, recent_seed_lift_;
  std::deque<double> absolute_seed_coverage_, absolute_seed_lift_;
  std::deque<double> seed_plateau_coverage_, seed_plateau_lift_;

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
    p.seed_plateau_horizon = std::max<std::size_t>(1, p.seed_plateau_horizon);
    p.seed_plateau_min_commits = std::max<std::size_t>(
        p.seed_plateau_horizon + 1, p.seed_plateau_min_commits);
    p.seed_plateau_confirmations =
        std::max<std::size_t>(1, p.seed_plateau_confirmations);
    p.seed_plateau_coverage_per_commit =
        finite_nonneg(p.seed_plateau_coverage_per_commit);
    p.seed_plateau_lift = finite_nonneg(p.seed_plateau_lift);

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
    d.seed_plateau_ready = st_.seed_plateau_ready;
    d.seed_low_plateau = st_.seed_plateau_ready &&
        st_.seed_plateau_confirmations > 0;
    d.seed_plateau_confirmations = seed_plateau_confirmations_;
    d.seed_progress_rounds = st_.seed_progress_rounds;
    d.seed_plateau_horizon_gain = st_.seed_plateau_horizon_gain;
    d.seed_plateau_gain_per_commit = st_.seed_plateau_gain_per_commit;
    d.seed_plateau_lift_median = st_.seed_plateau_lift_median;
    return d;
  }
};

}  // namespace lazyipdca
