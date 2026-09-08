#pragma once
// ADAPTIVE_MARGINAL_SEED_V2
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

namespace lazyipdca {

class AdaptiveMarginalSeedController {
 public:
  struct Params {
    std::size_t warmup = 4;
    std::size_t window = 3;
    std::size_t confirmations = 2;
    double alpha_coverage = 0.20;
    double alpha_lift = 0.20;
    double coverage_min = 1e-5;
    double coverage_max = 2.5e-3;
    double lift_min = 1e-5;
    double lift_max = 1e-3;
  };

  struct Decision {
    bool observed = false;
    bool low_return = false;
    bool stop = false;
    std::size_t atom_count = 0;
    std::size_t atom_jump = 0;
    std::size_t observations = 0;
    std::size_t confirmations = 0;
    double coverage = 0.0;
    double coverage_gain = 0.0;
    double lift_rel = 0.0;
    double coverage_ref = 0.0;
    double lift_ref = 0.0;
    double coverage_threshold = 0.0;
    double lift_threshold = 0.0;
    double recent_coverage_median = 0.0;
    double recent_lift_median = 0.0;
  };

  struct Stats {
    std::size_t observations = 0;
    std::size_t low_return_rounds = 0;
    std::size_t multi_atom_jumps = 0;
    std::size_t stop_atom = 0;
    bool references_ready = false;
    bool stopped = false;
    double coverage_ref = 0.0;
    double lift_ref = 0.0;
    double coverage_threshold = 0.0;
    double lift_threshold = 0.0;
    double last_coverage = 0.0;
    double last_coverage_gain = 0.0;
    double last_lift_rel = 0.0;
    double recent_coverage_median = 0.0;
    double recent_lift_median = 0.0;
    double stop_coverage = 0.0;
    double stop_coverage_gain = 0.0;
    double stop_lift_rel = 0.0;
  };

  AdaptiveMarginalSeedController() : p_(sanitize(Params{})) {}
  explicit AdaptiveMarginalSeedController(Params p) : p_(sanitize(p)) {}

  void clear() {
    warm_cov_.clear(); warm_lift_.clear(); recent_cov_.clear(); recent_lift_.clear();
    last_atom_count_ = 0; confirmations_ = 0; st_ = Stats{};
  }

  const Stats& stats() const { return st_; }

  Decision observe(double coverage, double lift, std::size_t atom_count) {
    Decision d; d.atom_count = atom_count;
    if (st_.stopped || atom_count <= last_atom_count_) return fill(d);

    d.observed = true;
    d.atom_jump = atom_count - last_atom_count_;
    if (d.atom_jump > 1) st_.multi_atom_jumps++;

    coverage = std::min(1.0, finite_nonneg(coverage));
    lift = finite_nonneg(lift);
    const double dcov = std::max(0.0, coverage - st_.last_coverage);

    last_atom_count_ = atom_count;
    st_.last_coverage = coverage;
    st_.last_coverage_gain = dcov;
    st_.last_lift_rel = lift;
    st_.observations++;

    if (!st_.references_ready) {
      warm_cov_.push_back(dcov);
      warm_lift_.push_back(lift);
      if (warm_cov_.size() >= p_.warmup) finalize_refs();
      return fill(d);
    }

    push(recent_cov_, dcov, p_.window);
    push(recent_lift_, lift, p_.window);
    if (recent_cov_.size() < p_.window) return fill(d);

    st_.recent_coverage_median = median(recent_cov_);
    st_.recent_lift_median = median(recent_lift_);
    // A strong current commit vetoes a low-return decision even if the rolling
    // median is still low; this gives temporal hysteresis without lagging across
    // a renewed useful-curvature regime.
    d.low_return = dcov <= st_.coverage_threshold &&
                   lift <= st_.lift_threshold &&
                   st_.recent_coverage_median <= st_.coverage_threshold &&
                   st_.recent_lift_median <= st_.lift_threshold;
    if (d.low_return) { st_.low_return_rounds++; confirmations_++; }
    else confirmations_ = 0;

    if (confirmations_ >= p_.confirmations) {
      st_.stopped = true;
      st_.stop_atom = atom_count;
      st_.stop_coverage = coverage;
      st_.stop_coverage_gain = dcov;
      st_.stop_lift_rel = lift;
      d.stop = true;
    }
    return fill(d);
  }

 private:
  Params p_;
  Stats st_;
  std::size_t last_atom_count_ = 0;
  std::size_t confirmations_ = 0;
  std::vector<double> warm_cov_, warm_lift_;
  std::deque<double> recent_cov_, recent_lift_;

  static double finite_nonneg(double x) { return std::isfinite(x) ? std::max(0.0, x) : 0.0; }
  static double clampv(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }
  static double median(std::vector<double> x) {
    if (x.empty()) return 0.0;
    const std::size_t n=x.size(), m=n/2;
    std::nth_element(x.begin(), x.begin()+m, x.end());
    const double hi=x[m];
    if (n & 1u) return hi;
    std::nth_element(x.begin(), x.begin()+m-1, x.begin()+m);
    return 0.5*(x[m-1]+hi);
  }
  static double median(const std::deque<double>& q) { return median(std::vector<double>(q.begin(),q.end())); }
  static void push(std::deque<double>& q, double x, std::size_t w) { q.push_back(x); while(q.size()>w) q.pop_front(); }
  static Params sanitize(Params p) {
    p.warmup=std::max<std::size_t>(1,p.warmup); p.window=std::max<std::size_t>(1,p.window); p.confirmations=std::max<std::size_t>(1,p.confirmations);
    p.alpha_coverage=clampv(finite_nonneg(p.alpha_coverage),0.0,1.0); p.alpha_lift=clampv(finite_nonneg(p.alpha_lift),0.0,1.0);
    p.coverage_min=finite_nonneg(p.coverage_min); p.coverage_max=finite_nonneg(p.coverage_max); p.lift_min=finite_nonneg(p.lift_min); p.lift_max=finite_nonneg(p.lift_max);
    if(p.coverage_max<p.coverage_min) std::swap(p.coverage_max,p.coverage_min);
    if(p.lift_max<p.lift_min) std::swap(p.lift_max,p.lift_min);
    return p;
  }
  void finalize_refs() {
    st_.coverage_ref=median(warm_cov_); st_.lift_ref=median(warm_lift_);
    st_.coverage_threshold=clampv(p_.alpha_coverage*st_.coverage_ref,p_.coverage_min,p_.coverage_max);
    st_.lift_threshold=clampv(p_.alpha_lift*st_.lift_ref,p_.lift_min,p_.lift_max);
    st_.references_ready=true;
  }
  Decision fill(Decision d) const {
    d.observations=st_.observations; d.confirmations=confirmations_; d.coverage=st_.last_coverage; d.coverage_gain=st_.last_coverage_gain; d.lift_rel=st_.last_lift_rel;
    d.coverage_ref=st_.coverage_ref; d.lift_ref=st_.lift_ref; d.coverage_threshold=st_.coverage_threshold; d.lift_threshold=st_.lift_threshold;
    d.recent_coverage_median=st_.recent_coverage_median; d.recent_lift_median=st_.recent_lift_median; d.stop=d.stop||st_.stopped; return d;
  }
};

} // namespace lazyipdca
