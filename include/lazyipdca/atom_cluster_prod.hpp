#pragma once
/*
 * LazyIPDCA — production online clustering for sparse atomic directions.
 * Marker: ATOM_CLUSTER_PROD_V1
 *
 * C++17, header-only, deterministic, bounded-memory.
 *
 * classify() is structurally pure: thresholds are adapted only when a
 * candidate is committed. Thus rejected candidates and pool order do not
 * alter thresholds inside the same oracle decision.
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace lazyipdca {

enum class AtomClusterDecision : std::uint8_t {
  Invalid = 0, Existing = 1, Novel = 2, Ambiguous = 3
};

template <class Index>
class ProductionAtomClusterer {
 public:
  struct Params {
    double w_direction = 0.85;
    double w_support = 0.10;
    double w_curvature = 0.05;
    double tau_in = 0.965;
    double tau_out = 0.900;
    bool adaptive_thresholds = true;
    std::size_t adapt_min_samples = 16;
    std::size_t adapt_window = 128;
    double adapt_strength = 0.20;
    double tau_in_min = 0.940, tau_in_max = 0.990;
    double tau_out_min = 0.860, tau_out_max = 0.940;
    double min_hysteresis_gap = 0.020;
    std::size_t max_clusters = 64;
    double replace_rel_quality = 1e-3;
    double replace_min_similarity = 0.90;
    std::size_t ttl_epochs = 0;
    std::size_t prune_min_hits = 2;
    bool validate_sorted_indices = false;
  };

  struct Similarity {
    double direction = 0.0, support = 0.0, curvature = 0.0, combined = 0.0;
  };

  struct Result {
    bool valid = false;
    AtomClusterDecision decision = AtomClusterDecision::Invalid;
    std::size_t cluster = npos();
    Similarity sim;
    double tau_in_effective = 0.0, tau_out_effective = 0.0;
    bool representative_replaced = false, registry_saturated = false;
  };

  struct Representative {
    std::vector<Index> idx;
    std::vector<double> val;
    double norm2 = 0.0, curvature_signature = 0.0;
    double best_quality = -std::numeric_limits<double>::infinity();
    std::uint64_t hits = 0, created_epoch = 0, last_seen_epoch = 0;
  };

  struct Stats {
    std::uint64_t classify_calls = 0, invalid_candidates = 0;
    std::uint64_t novel_assignments = 0, existing_assignments = 0;
    std::uint64_t ambiguous_assignments = 0, representative_replacements = 0;
    std::uint64_t saturated_novel_candidates = 0, pruned_clusters = 0;
    double last_max_similarity = 0.0;
    double tau_in_effective = 0.0, tau_out_effective = 0.0;
  };

  ProductionAtomClusterer() : p_(sanitize(Params{})) { reset_thresholds(); }
  explicit ProductionAtomClusterer(const Params& p) : p_(sanitize(p)) {
    reset_thresholds();
  }

  static constexpr std::size_t npos() {
    return std::numeric_limits<std::size_t>::max();
  }

  void clear() {
    reps_.clear(); similarity_window_.clear(); window_pos_ = 0; epoch_ = 0;
    stats_ = Stats{}; reset_thresholds();
  }
  void begin_epoch() { ++epoch_; if (p_.ttl_epochs) prune_stale(); }
  std::size_t size() const noexcept { return reps_.size(); }
  const Stats& stats() const noexcept { return stats_; }

  Result classify(const std::vector<Index>& idx,
                  const std::vector<double>& val,
                  double curvature_signature) {
    ++stats_.classify_calls;
    Result out;
    out.tau_in_effective = tau_in_eff_;
    out.tau_out_effective = tau_out_eff_;
    const double n2 = norm2(val);
    if (idx.size() != val.size() || !(n2 > 1e-28) ||
        !std::isfinite(n2) || !std::isfinite(curvature_signature) ||
        (p_.validate_sorted_indices && !strictly_sorted_unique(idx))) {
      ++stats_.invalid_candidates; return out;
    }
    out.valid = true;
    if (reps_.empty()) {
      out.decision = AtomClusterDecision::Novel;
      stats_.last_max_similarity = 0.0;
      return out;
    }

    double best = -1.0;
    std::size_t best_k = npos();
    Similarity best_s{};
    for (std::size_t k=0;k<reps_.size();++k) {
      Similarity s = similarity_to(idx,val,n2,curvature_signature,reps_[k]);
      if (s.combined > best + 1e-15 ||
          (std::fabs(s.combined-best)<=1e-15 && k<best_k)) {
        best=s.combined; best_k=k; best_s=s;
      }
    }
    out.cluster=best_k; out.sim=best_s;
    stats_.last_max_similarity=std::max(0.0,best_s.combined);
    if (best_s.combined >= tau_in_eff_) out.decision=AtomClusterDecision::Existing;
    else if (best_s.combined <= tau_out_eff_) out.decision=AtomClusterDecision::Novel;
    else out.decision=AtomClusterDecision::Ambiguous;
    out.registry_saturated =
        out.decision==AtomClusterDecision::Novel && reps_.size()>=p_.max_clusters;
    return out;
  }

  Result commit(Result r,
                const std::vector<Index>& idx,
                const std::vector<double>& val,
                double curvature_signature,
                double oracle_quality) {
    if (!r.valid) return r;
    const double n2=norm2(val);
    if (!(n2>1e-28) || idx.size()!=val.size()) {
      r.valid=false; r.decision=AtomClusterDecision::Invalid;
      ++stats_.invalid_candidates; return r;
    }

    // Adapt only from committed nearest-neighbour evidence.
    if (r.cluster != npos()) {
      push_similarity(r.sim.combined);
      update_effective_thresholds();
    }

    if (r.decision==AtomClusterDecision::Novel && reps_.size()<p_.max_clusters) {
      Representative rep;
      rep.idx=idx; rep.val=val; rep.norm2=n2;
      rep.curvature_signature=curvature_signature;
      rep.best_quality=oracle_quality; rep.hits=1;
      rep.created_epoch=rep.last_seen_epoch=epoch_;
      reps_.push_back(std::move(rep));
      r.cluster=reps_.size()-1;
      ++stats_.novel_assignments;
    } else {
      if (r.decision==AtomClusterDecision::Novel && reps_.size()>=p_.max_clusters) {
        ++stats_.saturated_novel_candidates; r.registry_saturated=true;
      }
      if (r.cluster!=npos()) {
        auto& rep=reps_[r.cluster];
        ++rep.hits; rep.last_seen_epoch=epoch_;
        if (r.decision==AtomClusterDecision::Ambiguous) ++stats_.ambiguous_assignments;
        else ++stats_.existing_assignments;
        const double scale=std::max(1.0,std::fabs(rep.best_quality));
        const bool gain=std::isfinite(oracle_quality) &&
          (!std::isfinite(rep.best_quality) ||
           oracle_quality>rep.best_quality+p_.replace_rel_quality*scale);
        const bool coherent=r.sim.direction>=p_.replace_min_similarity;
        if (gain && coherent) {
          rep.idx=idx; rep.val=val; rep.norm2=n2;
          rep.curvature_signature=curvature_signature;
          rep.best_quality=oracle_quality;
          r.representative_replaced=true;
          ++stats_.representative_replacements;
        }
      }
    }
    r.tau_in_effective=tau_in_eff_;
    r.tau_out_effective=tau_out_eff_;
    return r;
  }

  Result observe(const std::vector<Index>& idx,
                 const std::vector<double>& val,
                 double curvature_signature,
                 double oracle_quality) {
    return commit(classify(idx,val,curvature_signature),
                  idx,val,curvature_signature,oracle_quality);
  }

  static double score_adjustment(const Result& r,
                                 double duplicate_penalty,
                                 double novelty_bonus) {
    if (!r.valid) return 0.0;
    duplicate_penalty=std::max(0.0,duplicate_penalty);
    novelty_bonus=std::max(0.0,novelty_bonus);
    if (r.decision==AtomClusterDecision::Existing)
      return -duplicate_penalty*r.sim.combined;
    if (r.decision==AtomClusterDecision::Novel)
      return novelty_bonus*(1.0-r.sim.combined);
    if (r.decision==AtomClusterDecision::Ambiguous) {
      const double mid=.5*(r.tau_in_effective+r.tau_out_effective);
      return r.sim.combined>=mid
        ? -.25*duplicate_penalty*r.sim.combined
        :  .25*novelty_bonus*(1.0-r.sim.combined);
    }
    return 0.0;
  }

 private:
  Params p_;
  std::vector<Representative> reps_;
  std::vector<double> similarity_window_;
  std::size_t window_pos_=0;
  std::uint64_t epoch_=0;
  mutable Stats stats_{};
  double tau_in_eff_=0.965, tau_out_eff_=0.900;

  static double clamp01(double x) { return std::min(1.0,std::max(0.0,x)); }

  static Params sanitize(Params p) {
    auto fin=[](double x,double d){return std::isfinite(x)?x:d;};
    p.w_direction=std::max(0.0,fin(p.w_direction,.85));
    p.w_support=std::max(0.0,fin(p.w_support,.10));
    p.w_curvature=std::max(0.0,fin(p.w_curvature,.05));
    double sw=p.w_direction+p.w_support+p.w_curvature;
    if (!(sw>0)) { p.w_direction=1; p.w_support=p.w_curvature=0; sw=1; }
    p.w_direction/=sw; p.w_support/=sw; p.w_curvature/=sw;
    p.tau_in=clamp01(fin(p.tau_in,.965)); p.tau_out=clamp01(fin(p.tau_out,.900));
    if (p.tau_out>p.tau_in) std::swap(p.tau_out,p.tau_in);
    p.tau_in_min=clamp01(fin(p.tau_in_min,.940));
    p.tau_in_max=clamp01(fin(p.tau_in_max,.990));
    p.tau_out_min=clamp01(fin(p.tau_out_min,.860));
    p.tau_out_max=clamp01(fin(p.tau_out_max,.940));
    if (p.tau_in_min>p.tau_in_max) std::swap(p.tau_in_min,p.tau_in_max);
    if (p.tau_out_min>p.tau_out_max) std::swap(p.tau_out_min,p.tau_out_max);
    p.min_hysteresis_gap=std::max(1e-6,std::min(.25,fin(p.min_hysteresis_gap,.020)));
    p.adapt_strength=std::max(0.0,std::min(1.0,fin(p.adapt_strength,.20)));
    if (p.adapt_window<8) p.adapt_window=8;
    if (p.adapt_min_samples<4) p.adapt_min_samples=4;
    if (p.adapt_min_samples>p.adapt_window) p.adapt_min_samples=p.adapt_window;
    if (!p.max_clusters) p.max_clusters=1;
    p.replace_rel_quality=std::max(0.0,fin(p.replace_rel_quality,1e-3));
    p.replace_min_similarity=clamp01(fin(p.replace_min_similarity,.90));
    return p;
  }

  static double norm2(const std::vector<double>& v) {
    double s=0;
    for(double x:v) { if(!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN(); s+=x*x; }
    return s;
  }
  static bool strictly_sorted_unique(const std::vector<Index>& x) {
    for(std::size_t i=1;i<x.size();++i) if(!(x[i-1]<x[i])) return false;
    return true;
  }
  static Similarity sparse_similarity(const std::vector<Index>& ia,
                                      const std::vector<double>& va,double na2,
                                      const std::vector<Index>& ib,
                                      const std::vector<double>& vb,double nb2) {
    std::size_t a=0,b=0,inter=0; double dot=0;
    while(a<ia.size() && b<ib.size()) {
      if(ia[a]==ib[b]) {dot+=va[a]*vb[b];++inter;++a;++b;}
      else if(ia[a]<ib[b]) ++a; else ++b;
    }
    const std::size_t uni=ia.size()+ib.size()-inter;
    Similarity s; const double den=std::sqrt(na2*nb2);
    s.direction=den>1e-28?clamp01(std::fabs(dot)/den):0;
    s.support=uni?double(inter)/double(uni):0;
    return s;
  }
  Similarity similarity_to(const std::vector<Index>& idx,
                           const std::vector<double>& val,double n2,
                           double curv,const Representative& rep) const {
    Similarity s=sparse_similarity(idx,val,n2,rep.idx,rep.val,rep.norm2);
    const double scale=std::max({1.0,std::fabs(curv),std::fabs(rep.curvature_signature)});
    s.curvature=clamp01(1.0-std::fabs(curv-rep.curvature_signature)/scale);
    s.combined=clamp01(p_.w_direction*s.direction+p_.w_support*s.support+
                       p_.w_curvature*s.curvature);
    return s;
  }
  void push_similarity(double x) {
    x=clamp01(x);
    if(similarity_window_.size()<p_.adapt_window) {similarity_window_.push_back(x);return;}
    similarity_window_[window_pos_]=x;
    window_pos_=(window_pos_+1)%p_.adapt_window;
  }
  static double quantile_copy(const std::vector<double>& x,double q) {
    if(x.empty()) return 0;
    std::vector<double> t=x; q=clamp01(q);
    double pos=q*double(t.size()-1);
    std::size_t lo=std::size_t(std::floor(pos)), hi=std::size_t(std::ceil(pos));
    std::nth_element(t.begin(),t.begin()+lo,t.end()); double vlo=t[lo];
    if(hi==lo) return vlo;
    std::nth_element(t.begin(),t.begin()+hi,t.end()); double vhi=t[hi];
    return vlo+(pos-double(lo))*(vhi-vlo);
  }
  void reset_thresholds() {
    tau_in_eff_=p_.tau_in; tau_out_eff_=p_.tau_out; enforce_hysteresis();
    stats_.tau_in_effective=tau_in_eff_; stats_.tau_out_effective=tau_out_eff_;
  }
  void update_effective_thresholds() {
    tau_in_eff_=p_.tau_in; tau_out_eff_=p_.tau_out;
    if(p_.adaptive_thresholds && similarity_window_.size()>=p_.adapt_min_samples) {
      const double q50=quantile_copy(similarity_window_,.50);
      const double q80=quantile_copy(similarity_window_,.80);
      tau_in_eff_=std::min(p_.tau_in_max,std::max(p_.tau_in_min,
        p_.tau_in+p_.adapt_strength*(q80-p_.tau_in)));
      tau_out_eff_=std::min(p_.tau_out_max,std::max(p_.tau_out_min,
        p_.tau_out+p_.adapt_strength*(q50-p_.tau_out)));
    }
    enforce_hysteresis();
    stats_.tau_in_effective=tau_in_eff_; stats_.tau_out_effective=tau_out_eff_;
  }
  void enforce_hysteresis() {
    if(tau_out_eff_>tau_in_eff_-p_.min_hysteresis_gap)
      tau_out_eff_=std::max(p_.tau_out_min,tau_in_eff_-p_.min_hysteresis_gap);
    tau_in_eff_=clamp01(tau_in_eff_); tau_out_eff_=clamp01(tau_out_eff_);
  }
  void prune_stale() {
    auto before=reps_.size();
    reps_.erase(std::remove_if(reps_.begin(),reps_.end(),[&](const Representative&r){
      bool stale=epoch_>r.last_seen_epoch && epoch_-r.last_seen_epoch>p_.ttl_epochs;
      return stale && r.hits<p_.prune_min_hits;
    }),reps_.end());
    stats_.pruned_clusters+=before-reps_.size();
  }
};

class ProductionAtomSeedStopper {
 public:
  struct Params {
    std::size_t window=12, min_observations=8, min_atoms=4;
    double max_novelty_rate=.10, max_coverage_gain=2.5e-3;
    double max_marginal_lift_rel=1e-3;
    std::size_t confirmation_rounds=2;
  };
  struct Decision {
    bool stop=false;
    double novelty_rate=1.0;
    double coverage_gain=std::numeric_limits<double>::infinity();
    double marginal_lift_rel=std::numeric_limits<double>::infinity();
    std::size_t confirmations=0;
  };

  ProductionAtomSeedStopper():p_(sanitize(Params{})){}
  explicit ProductionAtomSeedStopper(const Params&p):p_(sanitize(p)){}
  void clear(){novelty_.clear();novelty_pos_=0;have_coverage_=false;last_coverage_=0;confirmations_=0;}
  void observe_commit(AtomClusterDecision d) {
    double x=d==AtomClusterDecision::Novel?1.0:
             d==AtomClusterDecision::Ambiguous?.5:0.0;
    if(novelty_.size()<p_.window) novelty_.push_back(x);
    else {novelty_[novelty_pos_]=x;novelty_pos_=(novelty_pos_+1)%p_.window;}
  }
  Decision after_round(double coverage,double marginal_lift_rel,std::size_t atom_count) {
    Decision d; d.novelty_rate=novelty_rate();
    if(!std::isfinite(coverage)||!std::isfinite(marginal_lift_rel)) return d;
    coverage=std::min(1.0,std::max(0.0,coverage));
    d.marginal_lift_rel=std::max(0.0,marginal_lift_rel);
    if(!have_coverage_) {last_coverage_=coverage;have_coverage_=true;return d;}
    d.coverage_gain=std::max(0.0,coverage-last_coverage_);
    last_coverage_=std::max(last_coverage_,coverage);
    bool gate=novelty_.size()>=p_.min_observations && atom_count>=p_.min_atoms &&
      d.novelty_rate<=p_.max_novelty_rate &&
      d.coverage_gain<=p_.max_coverage_gain &&
      d.marginal_lift_rel<=p_.max_marginal_lift_rel;
    confirmations_=gate?confirmations_+1:0;
    d.confirmations=confirmations_; d.stop=confirmations_>=p_.confirmation_rounds;
    return d;
  }
 private:
  Params p_; std::vector<double> novelty_; std::size_t novelty_pos_=0;
  bool have_coverage_=false; double last_coverage_=0; std::size_t confirmations_=0;
  static Params sanitize(Params p) {
    if(p.window<4)p.window=4;if(p.min_observations<2)p.min_observations=2;
    if(p.min_observations>p.window)p.min_observations=p.window;if(!p.min_atoms)p.min_atoms=1;
    p.max_novelty_rate=std::min(1.0,std::max(0.0,p.max_novelty_rate));
    p.max_coverage_gain=std::max(0.0,p.max_coverage_gain);
    p.max_marginal_lift_rel=std::max(0.0,p.max_marginal_lift_rel);
    if(!p.confirmation_rounds)p.confirmation_rounds=1;return p;
  }
  double novelty_rate()const {
    return novelty_.empty()?1.0:
      std::accumulate(novelty_.begin(),novelty_.end(),0.0)/double(novelty_.size());
  }
};

} // namespace lazyipdca
