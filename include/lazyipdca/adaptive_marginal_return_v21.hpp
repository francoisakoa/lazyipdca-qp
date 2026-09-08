#pragma once
// ADAPTIVE_MARGINAL_RETURN_V21
// Global, commit-time marginal-return controller for LazyIPDCA C5.
// - scores every oracle commit (including seed rounds later judged noisy),
// - never changes candidate ranking,
// - can request an early end of the current seed episode,
// - outside seeding, increases event spacing through a reversible cooldown,
// - direct emergency event commits are counted and treated as strong reactivation evidence.

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

class AdaptiveMarginalReturnController {
 public:
  struct Params {
    std::size_t warmup = 4;
    std::size_t window = 3;
    std::size_t confirmations = 2;

    // Two scale-free commit returns:
    //   lift_rel      = oracle spectral lift of the committed atom;
    //   effective_rel = lift_rel * clipped activity, i.e. spectral lift that
    //                   is also relevant to the current active geometry.
    double alpha_lift = 0.20;
    double alpha_effective = 0.20;
    double lift_min = 1e-5;
    double lift_max = 1e-3;
    double effective_min = 1e-5;
    double effective_max = 2.5e-3;

    // Reversible post-seed throttling.  A confirmed low-return regime does
    // not freeze atom growth; it only enlarges the minimum event spacing.
    double cooldown_factor = 2.0;
    double cooldown_max_factor = 8.0;
    double strong_factor = 4.0;
  };

  struct Decision {
    bool observed = false;
    bool scored = false;
    bool references_ready = false;
    bool low_return = false;
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
  };

  AdaptiveMarginalReturnController() : p_(sanitize(Params{})) {}
  explicit AdaptiveMarginalReturnController(Params p) : p_(sanitize(p)) {}

  void clear() {
    st_ = Stats{};
    warm_lift_.clear();
    warm_effective_.clear();
    recent_lift_.clear();
    recent_effective_.clear();
    confirmations_ = 0;
  }

  // Reset only model-relative learning after a complete atom rollback.  Keep
  // cumulative counters for auditability.
  void reset_model() {
    st_.model_resets++;
    st_.references_ready = false;
    st_.lift_ref = 0.0;
    st_.effective_ref = 0.0;
    st_.lift_threshold = 0.0;
    st_.effective_threshold = 0.0;
    st_.recent_lift_median = 0.0;
    st_.recent_effective_median = 0.0;
    st_.last_lift_rel = 0.0;
    st_.last_effective_rel = 0.0;
    st_.cooldown_multiplier = 1.0;
    warm_lift_.clear();
    warm_effective_.clear();
    recent_lift_.clear();
    recent_effective_.clear();
    confirmations_ = 0;
  }

  const Stats& stats() const { return st_; }

  Decision observe_scored(double lift_rel,
                          double effective_rel,
                          std::size_t atom_count,
                          AdaptiveReturnSource source) {
    Decision d;
    d.observed = true;
    d.scored = true;
    d.source = source;
    d.atom_count = atom_count;

    count_commit(source, atom_count);
    st_.scored_observations++;

    lift_rel = finite_nonneg(lift_rel);
    effective_rel = finite_nonneg(effective_rel);
    // The effective return must not exceed the underlying spectral lift.
    effective_rel = std::min(effective_rel, lift_rel);
    st_.last_lift_rel = lift_rel;
    st_.last_effective_rel = effective_rel;
    st_.last_source_code = static_cast<int>(source);

    if (!st_.references_ready) {
      warm_lift_.push_back(lift_rel);
      warm_effective_.push_back(effective_rel);
      if (warm_lift_.size() >= p_.warmup) finalize_refs();
      return fill(d);
    }

    push(recent_lift_, lift_rel, p_.window);
    push(recent_effective_, effective_rel, p_.window);
    if (recent_lift_.size() < p_.window) return fill(d);

    st_.recent_lift_median = median(recent_lift_);
    st_.recent_effective_median = median(recent_effective_);

    d.strong_return =
        lift_rel > p_.strong_factor * st_.lift_threshold ||
        effective_rel > p_.strong_factor * st_.effective_threshold;

    // A strong CURRENT commit vetoes a stale low rolling median.
    d.low_return =
        !d.strong_return &&
        lift_rel <= st_.lift_threshold &&
        effective_rel <= st_.effective_threshold &&
        st_.recent_lift_median <= st_.lift_threshold &&
        st_.recent_effective_median <= st_.effective_threshold;

    if (d.strong_return) {
      confirmations_ = 0;
      if (!is_seed(source) && st_.cooldown_multiplier > 1.0 + 1e-12) {
        st_.cooldown_multiplier =
            std::max(1.0, st_.cooldown_multiplier / p_.cooldown_factor);
        st_.reactivations++;
        d.action = AdaptiveReturnAction::Reactivate;
      }
      return fill(d);
    }

    if (!d.low_return) {
      confirmations_ = 0;
      return fill(d);
    }

    st_.low_return_rounds++;
    confirmations_++;
    if (confirmations_ < p_.confirmations) return fill(d);

    confirmations_ = 0;
    if (is_seed(source)) {
      st_.seed_stop_requests++;
      d.action = AdaptiveReturnAction::StopSeed;
    } else {
      const double old = st_.cooldown_multiplier;
      st_.cooldown_multiplier = std::min(
          p_.cooldown_max_factor,
          std::max(p_.cooldown_factor, old * p_.cooldown_factor));
      if (st_.cooldown_multiplier > old + 1e-12) st_.cooldowns++;
      d.action = AdaptiveReturnAction::Cooldown;
    }
    return fill(d);
  }

  // Direct A/B/C event directions do not have the oracle's post-commit lift
  // score.  Count them, never classify them as low-return, and treat them as
  // evidence strong enough to relax one cooldown level.
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
    confirmations_ = 0;
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
    const double cap = static_cast<double>(std::numeric_limits<std::size_t>::max());
    return static_cast<std::size_t>(std::min(raw, cap));
  }

 private:
  Params p_;
  Stats st_;
  std::size_t confirmations_ = 0;
  std::vector<double> warm_lift_, warm_effective_;
  std::deque<double> recent_lift_, recent_effective_;

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
    p.cooldown_factor = std::max(1.0, finite_nonneg(p.cooldown_factor));
    p.cooldown_max_factor =
        std::max(p.cooldown_factor, finite_nonneg(p.cooldown_max_factor));
    p.strong_factor = std::max(1.0, finite_nonneg(p.strong_factor));
    return p;
  }

  void finalize_refs() {
    st_.lift_ref = median(warm_lift_);
    st_.effective_ref = median(warm_effective_);
    st_.lift_threshold = clampv(p_.alpha_lift * st_.lift_ref,
                                p_.lift_min, p_.lift_max);
    st_.effective_threshold =
        clampv(p_.alpha_effective * st_.effective_ref,
               p_.effective_min, p_.effective_max);
    st_.references_ready = true;
  }

  Decision fill(Decision d) const {
    d.references_ready = st_.references_ready;
    d.confirmations = confirmations_;
    d.lift_rel = st_.last_lift_rel;
    d.effective_rel = st_.last_effective_rel;
    d.lift_ref = st_.lift_ref;
    d.effective_ref = st_.effective_ref;
    d.lift_threshold = st_.lift_threshold;
    d.effective_threshold = st_.effective_threshold;
    d.recent_lift_median = st_.recent_lift_median;
    d.recent_effective_median = st_.recent_effective_median;
    d.cooldown_multiplier = st_.cooldown_multiplier;
    return d;
  }
};

}  // namespace lazyipdca
