// runner/cutest_runner.cpp — manifest-driven CUTEst QP benchmark runner.
//
// For each problem NAME in the manifest CSV (column "name" or first column;
// optional column "sif_params" like "N=100000,M=25000"):
//   1. decode NAME.SIF with sifdecoder into <decode-dir>/NAME/ (skipped if
//      OUTSDIF.d already present — compatible with pre-decoded trees);
//   2. extract the QP data at x=0 (q = grad f(0), Q = Hessian of the
//      Lagrangian at (0,0); linear constraints cl-c0 <= Jx <= cu-c0);
//   3. solve with lazyipdca;
//   4. write <outdir>/NAME/NAME_raw.json (one directory per problem: no CSV
//      race conditions under parallel invocation).
//
// Usage:
//   lazyipdca_cutest --manifest manifest.csv --outdir results \
//       [--decode-dir /tmp/lazyipdca_decode] [--problem NAME] \
//       [--max-iter 500] [--time-limit 900] [--eps 1e-6] [--verbose 1] \
//       [--regularization-mode legacy|const|dyn|lazy] \
//       [--rho-init auto|user|lanczos|gershgorin|hint] [--rho0 VALUE]
//
// Environment: MASTSIF, SIFDECODE/MYARCH as usual for sifdecoder.
// For fair IPOPT comparisons run with OMP_NUM_THREADS=1 and all BLAS at 1.
#include "lazyipdca/solver.hpp"
#include "lazyipdca/inertial_complete_v24.hpp"  // INERTIA_COMPLETE_AMRS_V24
// CUTEST_RUNNER_V4HD_COMPAT_CLI
// CUTEST_RUNNER_V4HD_STATUS9_V2
// CUTEST_RUNNER_V4HD_RETRO_DEFAULTS
#include <dlfcn.h>
typedef int integer;
typedef bool logical;
struct CutestApi {
  void* handle = nullptr;
  void (*f_open)(const integer*, const char*, integer*);
  void (*f_close)(const integer*, integer*);
  void (*cdimen)(integer*, const integer*, integer*, integer*);
  void (*csetup)(integer*, const integer*, const integer*, const integer*,
                 integer*, integer*, double*, double*, double*, double*,
                 double*, double*, logical*, logical*, const integer*,
                 const integer*, const integer*);
  void (*cofg)(integer*, const integer*, const double*, double*, double*, const logical*);
  void (*cdimsh)(integer*, integer*);
  void (*csh)(integer*, const integer*, const integer*, const double*, const double*,
              integer*, const integer*, double*, integer*, integer*);
  void (*cdimsj)(integer*, integer*);
  void (*ccfsg)(integer*, const integer*, const integer*, const double*, double*,
                integer*, const integer*, double*, integer*, integer*, const logical*);
  void (*cterminate)(integer*);
  bool load(const std::string& sopath, std::string& err) {
    handle = dlopen(sopath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) { err = dlerror(); return false; }
    auto sym = [&](const char* nm) { return dlsym(handle, nm); };
    f_open = (decltype(f_open))sym("fortran_open_");
    f_close = (decltype(f_close))sym("fortran_close_");
    cdimen = (decltype(cdimen))sym("cutest_cdimen_");
    csetup = (decltype(csetup))sym("cutest_cint_csetup_");
    cofg = (decltype(cofg))sym("cutest_cint_cofg_");
    cdimsh = (decltype(cdimsh))sym("cutest_cdimsh_");
    csh = (decltype(csh))sym("cutest_csh_");
    cdimsj = (decltype(cdimsj))sym("cutest_cdimsj_");
    ccfsg = (decltype(ccfsg))sym("cutest_cint_ccfsg_");
    cterminate = (decltype(cterminate))sym("cutest_cterminate_");
    if (!f_open || !f_close || !cdimen || !csetup || !cofg || !cdimsh ||
        !csh || !cdimsj || !ccfsg || !cterminate) { err = "symbole CUTEst manquant"; return false; }
    return true;
  }
  ~CutestApi() { if (handle) dlclose(handle); }
};
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <memory>

using namespace lazyipdca;

namespace {

constexpr double kInfBound = 1e19;
inline double debound(double v) {
  if (v >= kInfBound) return INF;
  if (v <= -kInfBound) return -INF;
  return v;
}

struct ManifestEntry {
  std::string name;                 // SIF problem name
  std::string tag;                  // unique tag: NAME or NAME_n{target_n}
  std::vector<std::string> params;  // sifdecoder -param tokens, e.g. "N=2005"
  std::string status;               // manifest status column (may be empty)
  double lambda_min_est = std::numeric_limits<double>::quiet_NaN();
};

// RFC-4180-ish CSV line splitter (handles quoted fields with commas and "" escapes)
std::vector<std::string> csv_split(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool inq = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (inq) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') { cur += '"'; ++i; }
        else inq = false;
      } else cur += c;
    } else {
      if (c == '"') inq = true;
      else if (c == ',') { out.push_back(cur); cur.clear(); }
      else cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

// extract "KEY": number pairs from a small JSON object like {"N": 2005}
std::vector<std::string> params_from_json(const std::string& js) {
  std::vector<std::string> out;
  size_t i = 0;
  while ((i = js.find('"', i)) != std::string::npos) {
    size_t j = js.find('"', i + 1);
    if (j == std::string::npos) break;
    std::string key = js.substr(i + 1, j - i - 1);
    size_t k = js.find(':', j);
    if (k == std::string::npos) break;
    ++k;
    while (k < js.size() && std::isspace((unsigned char)js[k])) ++k;
    size_t e = k;
    while (e < js.size() && (std::isdigit((unsigned char)js[e]) || js[e] == '-' ||
                             js[e] == '+' || js[e] == '.' || js[e] == 'e' || js[e] == 'E')) ++e;
    if (e > k) out.push_back(key + "=" + js.substr(k, e - k));
    i = e;
  }
  return out;
}

std::vector<ManifestEntry> read_manifest(const std::string& path) {
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "cannot open manifest %s\n", path.c_str()); std::exit(2); }
  std::vector<ManifestEntry> out;
  std::string line;
  int c_name = -1, c_params = -1, c_pjson = -1, c_target = -1, c_status = -1, c_lmin = -1;
  bool first = true;
  auto trim = [](std::string& s) {
    while (!s.empty() && std::isspace((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && std::isspace((unsigned char)s.back())) s.pop_back();
  };
  while (std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> cells = csv_split(line);
    if (first) {
      first = false;
      for (size_t i = 0; i < cells.size(); ++i) {
        std::string c = cells[i];
        for (auto& ch : c) ch = (char)std::tolower(ch);
        trim(c);
        if (c == "name" || c == "problem" || c == "prob") c_name = (int)i;
        else if (c == "sif_params" || c == "params" || c == "param") c_params = (int)i;
        else if (c == "params_json") c_pjson = (int)i;
        else if (c == "target_n") c_target = (int)i;
        else if (c == "status") c_status = (int)i;
        else if (c == "lambda_min_est") c_lmin = (int)i;
      }
      if (c_name >= 0) continue;   // header line consumed
      c_name = 0;                  // headerless manifest: first column
    }
    if ((int)cells.size() <= c_name) continue;
    ManifestEntry e;
    e.name = cells[c_name]; trim(e.name);
    if (e.name.empty()) continue;
    if (c_pjson >= 0 && (int)cells.size() > c_pjson && !cells[c_pjson].empty())
      e.params = params_from_json(cells[c_pjson]);
    else if (c_params >= 0 && (int)cells.size() > c_params) {
      std::stringstream ss(cells[c_params]); std::string tok;
      while (std::getline(ss, tok, ';')) {
        std::stringstream s2(tok); std::string t2;
        while (std::getline(s2, t2, ' ')) { trim(t2); if (!t2.empty()) e.params.push_back(t2); }
      }
    }
    if (c_status >= 0 && (int)cells.size() > c_status) { e.status = cells[c_status]; trim(e.status); }
    if (c_lmin >= 0 && (int)cells.size() > c_lmin && !cells[c_lmin].empty())
      e.lambda_min_est = std::atof(cells[c_lmin].c_str());
    e.tag = e.name;
    if (c_target >= 0 && (int)cells.size() > c_target) {
      std::string t = cells[c_target]; trim(t);
      if (!t.empty()) e.tag = e.name + "_n" + t;
    }
    out.push_back(e);
  }
  return out;
}

bool file_exists(const std::string& p) { struct stat st{}; return ::stat(p.c_str(), &st) == 0; }

void write_json(const std::string& path, const std::string& name, idx_t n, idx_t p,
                idx_t m, const Info& info, const Settings& st, const std::string& note) {
  static const char* names[] = {"solved", "solved_soc", "max_iter", "time_limit",
                                "primal_infeasible_like", "numerical_error", "unbounded",
                                "dca_converged", "dual_degenerate", "factorization_budget"};
  const int status_i = static_cast<int>(info.status);
  const int n_status_names =
      static_cast<int>(sizeof(names) / sizeof(names[0]));
  const char* status_s =
      (status_i >= 0 && status_i < n_status_names)
          ? names[status_i] : "unknown";
  std::ofstream o(path);
  o.precision(16);
  o << "{\n"
    << "  \"solver\": \"lazyipdca\",\n"
    << "  \"name\": \"" << name << "\",\n"
    << "  \"n\": " << n << ",\n  \"m_eq\": " << p << ",\n  \"m_ineq\": " << m << ",\n"
    << "  \"status\": " << status_i << ",\n"
    << "  \"status_str\": \"" << status_s << "\",\n"
    << "  \"iters\": " << info.iters << ",\n"
    << "  \"obj\": " << info.obj << ",\n"
    << "  \"kkt_stat_abs\": " << info.res_stat_abs << ",\n"
    << "  \"kkt_stat_rel\": " << info.res_stat_rel << ",\n"
    << "  \"cv\": " << std::max(info.res_eq_abs, info.res_ineq_abs) << ",\n"
    << "  \"kkt_feas_rel\": " << info.res_feas_rel << ",\n"
    << "  \"kkt_comp_rel\": " << info.res_comp_rel << ",\n"
    << "  \"regularization_mode\": \"" << regularization_mode_name(st.regularization_mode) << "\",\n"
    << "  \"low_rank_effective\": " << (((st.regularization_mode == DcRegularizationMode::Lazy ||
         st.regularization_mode == DcRegularizationMode::Legacy) && st.atoms_enable) ? "true" : "false") << ",\n"
    << "  \"lazy_no_atoms\": " << ((st.regularization_mode == DcRegularizationMode::Lazy && !st.atoms_enable) ? "true" : "false") << ",\n"
    << "  \"rho_model_fixed\": " << ((st.regularization_mode == DcRegularizationMode::Const ||
         (st.regularization_mode == DcRegularizationMode::Lazy && st.rho_model_lock)) ? "true" : "false") << ",\n"
    << "  \"rho_model_lock\": " << (st.rho_model_lock ? "true" : "false") << ",\n"
    << "  \"rho_init_method\": \"" << rho_init_method_name(st.rho_init_method) << "\",\n"
    << "  \"rho_initial\": " << info.rho_initial << ",\n"
    << "  \"rho_final\": " << info.rho_final << ",\n"
    << "  \"rho_stability_final\": " << info.rho_stability_final << ",\n"
    << "  \"n_rho_model_updates\": " << info.n_rho_model_updates << ",\n"
    << "  \"n_rho_model_increases\": " << info.n_rho_model_increases << ",\n"
    << "  \"n_rho_model_decreases\": " << info.n_rho_model_decreases << ",\n"
    << "  \"n_rho_curvature_checks\": " << info.n_rho_curvature_checks << ",\n"
    << "  \"n_rho_curvature_rejects\": " << info.n_rho_curvature_rejects << ",\n"
    << "  \"rho_last_curvature_guard\": " << (std::isfinite(info.rho_last_curvature_guard) ? info.rho_last_curvature_guard : 0.0) << ",\n"
    << "  \"rho0_setting\": " << st.rho0 << ",\n"
    << "  \"rho_init_rel_margin\": " << st.rho_init_rel_margin << ",\n"
    << "  \"rho_init_abs_margin\": " << st.rho_init_abs_margin << ",\n"
    << "  \"rho_curvature_tol\": " << st.rho_curvature_tol << ",\n"
    << "  \"rho_lanczos_residual_safety\": " << st.rho_lanczos_residual_safety << ",\n"
    << "  \"rho_dyn_floor_rel_margin\": " << st.rho_dyn_floor_rel_margin << ",\n"
    << "  \"rho_dyn_reject_growth\": " << st.rho_dyn_reject_growth << ",\n"
    << "  \"rho_model_min\": " << st.rho_model_min << ",\n"
    << "  \"rho_model_lock_setting\": " << (st.rho_model_lock ? "true" : "false") << ",\n"

    << "  \"rho_floor\": " << st.rho_floor << ",\n"
    << "  \"rho_growth\": " << st.rho_growth << ",\n"
    << "  \"rho_rel_margin\": " << st.rho_rel_margin << ",\n"
    << "  \"rho_decay\": " << st.rho_decay << ",\n"
    << "  \"rho_trial_every\": " << st.rho_trial_every << ",\n"
    << "  \"rho_clean_streak\": " << st.rho_clean_streak << ",\n"
    << "  \"rho_abs_max\": " << st.rho_abs_max << ",\n"
    << "  \"rho_dynamic_terminal_schedule\": " << (st.rho_dynamic ? "true" : "false") << ",\n"
    << "  \"lambda_min_reduced\": " << (std::isfinite(info.lambda_min_reduced) ? info.lambda_min_reduced : 0.0) << ",\n"
    << "  \"soc_certified\": " << (info.soc_certified ? "true" : "false") << ",\n"
    << "  \"n_factorizations\": " << info.n_factorizations << ",\n"
    << "  \"max_kkt_factorizations\": " << st.max_kkt_factorizations << ",\n"
    << "  \"n_inertia_corrections\": " << info.n_inertia_corrections << ",\n"
    << "  \"n_lanczos_calls\": " << info.n_lanczos_calls << ",\n"
    << "  \"n_atoms\": " << info.n_atoms << ",\n"
    << "  \"max_seed_atoms\": " << st.atoms_seed_max_atoms << ",\n"
    << "  \"n_seed_atoms\": " << info.n_seed_atoms << ",\n"
    << "  \"seed_coverage_est\": " << info.seed_coverage_est << ",\n"
    << "  \"seed_atom_cap_hit\": " << (info.seed_atom_cap_hit ? "true" : "false") << ",\n"
    << "  \"auto_enabled\": " << (st.auto_atoms_enable ? "true" : "false") << ",\n"
    << "  \"auto_coverage_trigger\": " << st.auto_coverage_trigger << ",\n"
    << "  \"auto_triggered\": " << (info.auto_triggered ? "true" : "false") << ",\n"
    << "  \"auto_scalar_committed\": " << (info.auto_scalar_committed ? "true" : "false") << ",\n"
    << "  \"auto_seed_coverage\": " << info.auto_seed_coverage << ",\n"
    << "  \"auto_seed_lambda0\": " << info.auto_seed_lambda0 << ",\n"
    << "  \"auto_seed_lambda_best\": " << info.auto_seed_lambda_best << ",\n"
    << "  \"auto_seed_atoms\": " << info.auto_seed_atoms << ",\n"
    << "  \"atoms_freeze_after_seed\": " << (st.atoms_freeze_after_seed ? "true" : "false") << ",\n"
    << "  \"atoms_nnz\": " << info.atoms_nnz << ",\n"
    // ATOM_CLUSTER_PROD_V1 settings + telemetry
    << "  \"atom_cluster_enable\": " << (st.atom_cluster_enable ? "true" : "false") << ",\n"
    << "  \"atom_cluster_w_direction\": " << st.atom_cluster_w_direction << ",\n"
    << "  \"atom_cluster_w_support\": " << st.atom_cluster_w_support << ",\n"
    << "  \"atom_cluster_w_curvature\": " << st.atom_cluster_w_curvature << ",\n"
    << "  \"atom_cluster_tau_in\": " << st.atom_cluster_tau_in << ",\n"
    << "  \"atom_cluster_tau_out\": " << st.atom_cluster_tau_out << ",\n"
    << "  \"atom_cluster_adaptive_thresholds\": " << (st.atom_cluster_adaptive_thresholds ? "true" : "false") << ",\n"
    << "  \"atom_cluster_max_reps\": " << st.atom_cluster_max_reps << ",\n"
    << "  \"atom_cluster_duplicate_penalty\": " << st.atom_cluster_duplicate_penalty << ",\n"
    << "  \"atom_cluster_novelty_bonus\": " << st.atom_cluster_novelty_bonus << ",\n"
    << "  \"atom_cluster_stop_window\": " << st.atom_cluster_stop_window << ",\n"
    << "  \"atom_cluster_stop_max_novelty_rate\": " << st.atom_cluster_stop_max_novelty_rate << ",\n"
    << "  \"atom_cluster_stop_max_coverage_gain\": " << st.atom_cluster_stop_max_coverage_gain << ",\n"
    << "  \"atom_cluster_stop_max_lift_rel\": " << st.atom_cluster_stop_max_lift_rel << ",\n"
    << "  \"atom_cluster_count_final\": " << info.atom_cluster_count_final << ",\n"
    << "  \"n_atom_cluster_classify\": " << info.n_atom_cluster_classify << ",\n"
    << "  \"n_atom_cluster_novel\": " << info.n_atom_cluster_novel << ",\n"
    << "  \"n_atom_cluster_existing\": " << info.n_atom_cluster_existing << ",\n"
    << "  \"n_atom_cluster_ambiguous\": " << info.n_atom_cluster_ambiguous << ",\n"
    << "  \"n_atom_cluster_rep_replacements\": " << info.n_atom_cluster_rep_replacements << ",\n"
    << "  \"n_atom_cluster_saturated\": " << info.n_atom_cluster_saturated << ",\n"
    << "  \"n_atom_cluster_seed_stops\": " << info.n_atom_cluster_seed_stops << ",\n"
    << "  \"atom_cluster_tau_in_final\": " << info.atom_cluster_tau_in_final << ",\n"
    << "  \"atom_cluster_tau_out_final\": " << info.atom_cluster_tau_out_final << ",\n"
    << "  \"atom_cluster_last_similarity\": " << info.atom_cluster_last_similarity << ",\n"
    << "  \"atom_cluster_seed_novelty_rate\": " << info.atom_cluster_seed_novelty_rate << ",\n"
    << "  \"atom_cluster_seed_coverage_gain\": " << info.atom_cluster_seed_coverage_gain << ",\n"
    << "  \"atom_cluster_seed_lift_rel\": " << info.atom_cluster_seed_lift_rel << ",\n"
    // ADAPTIVE_MARGINAL_SEED_V2 settings + telemetry
    << "  \"seed_return_enable\": " << (st.adaptive_seed_return_enable ? "true" : "false") << ",\n"
    << "  \"seed_return_warmup\": " << st.adaptive_seed_return_warmup << ",\n"
    << "  \"seed_return_window\": " << st.adaptive_seed_return_window << ",\n"
    << "  \"seed_return_confirmations_required\": " << st.adaptive_seed_return_confirmations << ",\n"
    << "  \"seed_return_alpha_coverage\": " << st.adaptive_seed_return_alpha_coverage << ",\n"
    << "  \"seed_return_alpha_lift\": " << st.adaptive_seed_return_alpha_lift << ",\n"
    << "  \"seed_return_coverage_min\": " << st.adaptive_seed_return_coverage_min << ",\n"
    << "  \"seed_return_coverage_max\": " << st.adaptive_seed_return_coverage_max << ",\n"
    << "  \"seed_return_lift_min\": " << st.adaptive_seed_return_lift_min << ",\n"
    << "  \"seed_return_lift_max\": " << st.adaptive_seed_return_lift_max << ",\n"
    << "  \"n_seed_return_observations\": " << info.n_seed_return_observations << ",\n"
    << "  \"n_seed_return_low_rounds\": " << info.n_seed_return_low_rounds << ",\n"
    << "  \"n_seed_return_stops\": " << info.n_seed_return_stops << ",\n"
    << "  \"n_seed_return_multi_atom_jumps\": " << info.n_seed_return_multi_atom_jumps << ",\n"
    << "  \"seed_return_stopped\": " << (info.seed_return_stopped ? "true" : "false") << ",\n"
    << "  \"seed_return_stop_atom\": " << info.seed_return_stop_atom << ",\n"
    << "  \"seed_return_seed_atoms_final\": " << info.seed_return_seed_atoms_final << ",\n"
    << "  \"seed_return_cov_ref\": " << info.seed_return_cov_ref << ",\n"
    << "  \"seed_return_lift_ref\": " << info.seed_return_lift_ref << ",\n"
    << "  \"seed_return_cov_threshold\": " << info.seed_return_cov_threshold << ",\n"
    << "  \"seed_return_lift_threshold\": " << info.seed_return_lift_threshold << ",\n"
    << "  \"seed_return_last_cov_gain\": " << info.seed_return_last_cov_gain << ",\n"
    << "  \"seed_return_last_lift_rel\": " << info.seed_return_last_lift_rel << ",\n"
    << "  \"seed_return_recent_cov_median\": " << info.seed_return_recent_cov_median << ",\n"
    << "  \"seed_return_recent_lift_median\": " << info.seed_return_recent_lift_median << ",\n"
    << "  \"seed_return_stop_coverage\": " << info.seed_return_stop_coverage << ",\n"
    << "  \"seed_return_stop_cov_gain\": " << info.seed_return_stop_cov_gain << ",\n"
    << "  \"seed_return_stop_lift_rel\": " << info.seed_return_stop_lift_rel << ",\n"
    // ADAPTIVE_MARGINAL_RETURN_V21 settings + telemetry
    << "  \"adaptive_return_enable\": " << (st.adaptive_marginal_return_enable ? "true" : "false") << ",\n"
    << "  \"adaptive_return_warmup\": " << st.adaptive_return_warmup << ",\n"
    << "  \"adaptive_return_window\": " << st.adaptive_return_window << ",\n"
    << "  \"adaptive_return_confirmations_required\": " << st.adaptive_return_confirmations << ",\n"
    << "  \"adaptive_return_alpha_lift\": " << st.adaptive_return_alpha_lift << ",\n"
    << "  \"adaptive_return_alpha_effective\": " << st.adaptive_return_alpha_effective << ",\n"
    << "  \"adaptive_return_lift_min\": " << st.adaptive_return_lift_min << ",\n"
    << "  \"adaptive_return_lift_max\": " << st.adaptive_return_lift_max << ",\n"
    << "  \"adaptive_return_effective_min\": " << st.adaptive_return_effective_min << ",\n"
    << "  \"adaptive_return_effective_max\": " << st.adaptive_return_effective_max << ",\n"
    << "  \"adaptive_return_cooldown_factor\": " << st.adaptive_return_cooldown_factor << ",\n"
    << "  \"adaptive_return_cooldown_max_factor\": " << st.adaptive_return_cooldown_max_factor << ",\n"
    << "  \"adaptive_return_strong_factor\": " << st.adaptive_return_strong_factor << ",\n"
    << "  \"n_adaptive_return_commits\": " << info.n_adaptive_return_commits << ",\n"
    << "  \"n_adaptive_return_scored\": " << info.n_adaptive_return_scored << ",\n"
    << "  \"n_adaptive_return_seed_commits\": " << info.n_adaptive_return_seed_commits << ",\n"
    << "  \"n_adaptive_return_event_oracle_commits\": " << info.n_adaptive_return_event_oracle_commits << ",\n"
    << "  \"n_adaptive_return_event_direct_commits\": " << info.n_adaptive_return_event_direct_commits << ",\n"
    << "  \"n_adaptive_return_update_commits\": " << info.n_adaptive_return_update_commits << ",\n"
    << "  \"n_adaptive_return_low_rounds\": " << info.n_adaptive_return_low_rounds << ",\n"
    << "  \"n_adaptive_return_seed_stop_requests\": " << info.n_adaptive_return_seed_stop_requests << ",\n"
    << "  \"n_adaptive_return_seed_stops\": " << info.n_adaptive_return_seed_stops << ",\n"
    << "  \"n_adaptive_return_cooldowns\": " << info.n_adaptive_return_cooldowns << ",\n"
    << "  \"n_adaptive_return_reactivations\": " << info.n_adaptive_return_reactivations << ",\n"
    << "  \"n_adaptive_return_model_resets\": " << info.n_adaptive_return_model_resets << ",\n"
    << "  \"adaptive_return_atoms_after_seed\": " << info.adaptive_return_atoms_after_seed << ",\n"
    << "  \"adaptive_return_seed_stopped\": " << (info.adaptive_return_seed_stopped ? "true" : "false") << ",\n"
    << "  \"adaptive_return_seed_stop_atom\": " << info.adaptive_return_seed_stop_atom << ",\n"
    << "  \"adaptive_return_event_spacing_current\": " << info.adaptive_return_event_spacing_current << ",\n"
    << "  \"adaptive_return_last_source\": " << info.adaptive_return_last_source << ",\n"
    << "  \"adaptive_return_lift_ref\": " << info.adaptive_return_lift_ref << ",\n"
    << "  \"adaptive_return_effective_ref\": " << info.adaptive_return_effective_ref << ",\n"
    << "  \"adaptive_return_lift_threshold\": " << info.adaptive_return_lift_threshold << ",\n"
    << "  \"adaptive_return_effective_threshold\": " << info.adaptive_return_effective_threshold << ",\n"
    << "  \"adaptive_return_last_lift_rel\": " << info.adaptive_return_last_lift_rel << ",\n"
    << "  \"adaptive_return_last_effective_rel\": " << info.adaptive_return_last_effective_rel << ",\n"
    << "  \"adaptive_return_recent_lift_median\": " << info.adaptive_return_recent_lift_median << ",\n"
    << "  \"adaptive_return_recent_effective_median\": " << info.adaptive_return_recent_effective_median << ",\n"
    << "  \"adaptive_return_cooldown_multiplier\": " << info.adaptive_return_cooldown_multiplier << ",\n"
    // ADAPTIVE_MARGINAL_RETURN_V22 seed settings + telemetry
    << "  \"adaptive_return_seed_alpha_coverage\": " << st.adaptive_return_seed_alpha_coverage << ",\n"
    << "  \"adaptive_return_seed_alpha_lift\": " << st.adaptive_return_seed_alpha_lift << ",\n"
    << "  \"adaptive_return_seed_coverage_min\": " << st.adaptive_return_seed_coverage_min << ",\n"
    << "  \"adaptive_return_seed_coverage_max\": " << st.adaptive_return_seed_coverage_max << ",\n"
    << "  \"adaptive_return_seed_lift_min\": " << st.adaptive_return_seed_lift_min << ",\n"
    << "  \"adaptive_return_seed_lift_max\": " << st.adaptive_return_seed_lift_max << ",\n"
    << "  \"adaptive_return_seed_absolute_enable\": " << (st.adaptive_return_seed_absolute_enable ? "true" : "false") << ",\n"
    << "  \"adaptive_return_seed_absolute_window\": " << st.adaptive_return_seed_absolute_window << ",\n"
    << "  \"adaptive_return_seed_absolute_confirmations_required\": " << st.adaptive_return_seed_absolute_confirmations << ",\n"
    << "  \"adaptive_return_seed_absolute_coverage\": " << st.adaptive_return_seed_absolute_coverage << ",\n"
    << "  \"adaptive_return_seed_absolute_lift\": " << st.adaptive_return_seed_absolute_lift << ",\n"
    << "  \"n_adaptive_return_seed_scored_rounds\": " << info.n_adaptive_return_seed_scored_rounds << ",\n"
    << "  \"n_adaptive_return_seed_unreliable_rounds\": " << info.n_adaptive_return_seed_unreliable_rounds << ",\n"
    << "  \"n_adaptive_return_seed_low_relative_rounds\": " << info.n_adaptive_return_seed_low_relative_rounds << ",\n"
    << "  \"n_adaptive_return_seed_low_absolute_rounds\": " << info.n_adaptive_return_seed_low_absolute_rounds << ",\n"
    << "  \"adaptive_return_seed_relative_confirmations\": " << info.adaptive_return_seed_relative_confirmations << ",\n"
    << "  \"adaptive_return_seed_absolute_confirmations\": " << info.adaptive_return_seed_absolute_confirmations << ",\n"
    << "  \"adaptive_return_seed_stop_reason\": " << info.adaptive_return_seed_stop_reason << ",\n"
    << "  \"adaptive_return_seed_references_ready\": " << (info.adaptive_return_seed_references_ready ? "true" : "false") << ",\n"
    << "  \"adaptive_return_seed_last_coverage\": " << info.adaptive_return_seed_last_coverage << ",\n"
    << "  \"adaptive_return_seed_last_coverage_gain\": " << info.adaptive_return_seed_last_coverage_gain << ",\n"
    << "  \"adaptive_return_seed_last_lift_rel\": " << info.adaptive_return_seed_last_lift_rel << ",\n"
    << "  \"adaptive_return_seed_coverage_ref\": " << info.adaptive_return_seed_coverage_ref << ",\n"
    << "  \"adaptive_return_seed_lift_ref\": " << info.adaptive_return_seed_lift_ref << ",\n"
    << "  \"adaptive_return_seed_coverage_threshold\": " << info.adaptive_return_seed_coverage_threshold << ",\n"
    << "  \"adaptive_return_seed_lift_threshold\": " << info.adaptive_return_seed_lift_threshold << ",\n"
    << "  \"adaptive_return_seed_recent_coverage_median\": " << info.adaptive_return_seed_recent_coverage_median << ",\n"
    << "  \"adaptive_return_seed_recent_lift_median\": " << info.adaptive_return_seed_recent_lift_median << ",\n"
    << "  \"adaptive_return_seed_absolute_recent_coverage_median\": " << info.adaptive_return_seed_absolute_recent_coverage_median << ",\n"
    << "  \"adaptive_return_seed_absolute_recent_lift_median\": " << info.adaptive_return_seed_absolute_recent_lift_median << ",\n"
    << "  \"adaptive_return_seed_stop_coverage\": " << info.adaptive_return_seed_stop_coverage << ",\n"
    << "  \"adaptive_return_seed_stop_coverage_gain\": " << info.adaptive_return_seed_stop_coverage_gain << ",\n"
    << "  \"adaptive_return_seed_stop_lift_rel\": " << info.adaptive_return_seed_stop_lift_rel << ",\n"
    // ADAPTIVE_MARGINAL_RETURN_V23 plateau settings + telemetry
    << "  \"adaptive_return_seed_plateau_enable\": " << (st.adaptive_return_seed_plateau_enable ? "true" : "false") << ",\n"
    << "  \"adaptive_return_seed_plateau_horizon\": " << st.adaptive_return_seed_plateau_horizon << ",\n"
    << "  \"adaptive_return_seed_plateau_min_commits\": " << st.adaptive_return_seed_plateau_min_commits << ",\n"
    << "  \"adaptive_return_seed_plateau_confirmations_required\": " << st.adaptive_return_seed_plateau_confirmations << ",\n"
    << "  \"adaptive_return_seed_plateau_coverage_per_commit\": " << st.adaptive_return_seed_plateau_coverage_per_commit << ",\n"
    << "  \"adaptive_return_seed_plateau_lift\": " << st.adaptive_return_seed_plateau_lift << ",\n"
    << "  \"n_adaptive_return_seed_progress_rounds\": " << info.n_adaptive_return_seed_progress_rounds << ",\n"
    << "  \"n_adaptive_return_seed_plateau_evaluations\": " << info.n_adaptive_return_seed_plateau_evaluations << ",\n"
    << "  \"n_adaptive_return_seed_low_plateau_rounds\": " << info.n_adaptive_return_seed_low_plateau_rounds << ",\n"
    << "  \"adaptive_return_seed_plateau_confirmations\": " << info.adaptive_return_seed_plateau_confirmations << ",\n"
    << "  \"adaptive_return_seed_plateau_ready\": " << (info.adaptive_return_seed_plateau_ready ? "true" : "false") << ",\n"
    << "  \"adaptive_return_seed_last_reliable_coverage\": " << info.adaptive_return_seed_last_reliable_coverage << ",\n"
    << "  \"adaptive_return_seed_plateau_horizon_gain\": " << info.adaptive_return_seed_plateau_horizon_gain << ",\n"
    << "  \"adaptive_return_seed_plateau_gain_per_commit\": " << info.adaptive_return_seed_plateau_gain_per_commit << ",\n"
    << "  \"adaptive_return_seed_plateau_lift_median\": " << info.adaptive_return_seed_plateau_lift_median << ",\n"
    << "  \"adaptive_return_seed_stop_plateau_gain_per_commit\": " << info.adaptive_return_seed_stop_plateau_gain_per_commit << ",\n"
    << "  \"adaptive_return_seed_stop_plateau_lift_median\": " << info.adaptive_return_seed_stop_plateau_lift_median << ",\n"

    << "  \"n_kkt_reuses\": " << info.n_kkt_reuses << ",\n"
    << "  \"n_bp_events\": " << info.n_bp_events << ",\n"
    << "  \"n_bp_pushed\": " << info.n_bp_pushed << ",\n"
    << "  \"n_accepts\": " << info.n_accepts << ",\n"
    << "  \"n_rejects\": " << info.n_rejects << ",\n"
    << "  \"dca_momentum\": \"" << momentum_mode_name(st.dca_momentum) << "\",\n"
    << "  \"dca_acceptance\": \"" << acceptance_mode_name(st.dca_acceptance) << "\",\n"
    << "  \"momentum_theta\": " << st.dca_momentum_theta << ",\n"
    << "  \"momentum_theta_min\": " << st.dca_momentum_theta_min << ",\n"
    << "  \"momentum_theta_max\": " << st.dca_momentum_theta_max << ",\n"
    << "  \"momentum_gamma_mode\": \"" << euclidean_gamma_mode_name(st.dca_momentum_gamma_mode) << "\",\n"
    << "  \"momentum_gamma\": " << st.dca_momentum_gamma << ",\n"
    << "  \"momentum_gamma_min\": " << st.dca_momentum_gamma_min << ",\n"
    << "  \"momentum_gamma_max\": " << st.dca_momentum_gamma_max << ",\n"
    << "  \"momentum_rho_factor\": " << st.dca_momentum_rho_factor << ",\n"
    << "  \"momentum_atom_align_floor\": " << st.dca_momentum_atom_align_floor << ",\n"
    << "  \"momentum_atom_align_power\": " << st.dca_momentum_atom_align_power << ",\n"
    << "  \"momentum_start\": " << st.dca_momentum_start << ",\n"
    << "  \"momentum_backtrack\": " << st.dca_momentum_backtrack << ",\n"
    << "  \"momentum_growth\": " << st.dca_momentum_growth << ",\n"
    << "  \"momentum_reset_on_metric_change\": " << (st.dca_momentum_reset_on_metric_change ? "true" : "false") << ",\n"
    << "  \"accept_window\": " << st.dca_accept_window << ",\n"
    << "  \"lyapunov_theta_bar\": " << st.dca_lyapunov_theta_bar << ",\n"
    << "  \"lyapunov_eta\": " << st.dca_lyapunov_eta << ",\n"
    << "  \"n_momentum_steps\": " << info.n_momentum_steps << ",\n"
    << "  \"n_momentum_resets\": " << info.n_momentum_resets << ",\n"
    << "  \"n_momentum_backtracks\": " << info.n_momentum_backtracks << ",\n"
    << "  \"momentum_theta_final\": " << info.momentum_theta_final << ",\n"
    << "  \"momentum_gamma_final\": " << info.momentum_gamma_final << ",\n"
    << "  \"momentum_force_inf_final\": " << info.momentum_force_inf_final << ",\n"
    << "  \"lyapunov_energy_final\": " << (std::isfinite(info.lyapunov_energy_final) ? info.lyapunov_energy_final : 0.0) << ",\n"
    << "  \"momentum_complete\": " << (st.dca_momentum_complete ? "true" : "false") << ",\n"
    << "  \"momentum_atom_weighting\": \"" << momentum_atom_weighting_name(st.dca_momentum_atom_weighting) << "\",\n"
    << "  \"momentum_rho_policy\": \"" << momentum_rho_policy_name(st.dca_momentum_rho_policy) << "\",\n"
    << "  \"n_momentum_complete_builds\": " << info.n_momentum_complete_builds << ",\n"
    << "  \"n_momentum_complete_accepts\": " << info.n_momentum_complete_accepts << ",\n"
    << "  \"n_momentum_restarts_rho\": " << info.n_momentum_restarts_rho << ",\n"
    << "  \"n_momentum_restarts_atom\": " << info.n_momentum_restarts_atom << ",\n"
    << "  \"n_momentum_restarts_beta\": " << info.n_momentum_restarts_beta << ",\n"
    << "  \"n_momentum_rho_keeps\": " << info.n_momentum_rho_keeps << ",\n"
    << "  \"n_momentum_rho_transports\": " << info.n_momentum_rho_transports << ",\n"
    << "  \"n_momentum_force_capped\": " << info.n_momentum_force_capped << ",\n"
    << "  \"n_momentum_stat_guard_shrinks\": " << info.n_momentum_stat_guard_shrinks << ",\n"
    << "  \"n_euclidean_gamma_capped\": " << info.n_euclidean_gamma_capped << ",\n"
    << "  \"n_theta_capped_by_lyapunov\": " << info.n_theta_capped_by_lyapunov << ",\n"
    << "  \"n_theta_align_decays\": " << info.n_theta_align_decays << ",\n"
    << "  \"momentum_theta_rho_final\": " << info.momentum_theta_rho_final << ",\n"
    << "  \"momentum_theta_atom_min\": " << info.momentum_theta_atom_min << ",\n"
    << "  \"momentum_theta_atom_mean\": " << info.momentum_theta_atom_mean << ",\n"
    << "  \"momentum_theta_atom_max\": " << info.momentum_theta_atom_max << ",\n"
    << "  \"momentum_align_mean\": " << info.momentum_align_mean << ",\n"
    << "  \"momentum_align_max\": " << info.momentum_align_max << ",\n"
    << "  \"momentum_energy_rho_share\": " << info.momentum_energy_rho_share << ",\n"
    << "  \"momentum_energy_atom_max_share\": " << info.momentum_energy_atom_max_share << ",\n"
    << "  \"momentum_energy_atom_effective_count\": " << info.momentum_energy_atom_effective_count << ",\n"
    << "  \"momentum_force_rel\": " << info.momentum_force_rel << ",\n"
    << "  \"stat_model_base_rel\": " << info.stat_model_base_rel << ",\n"
    << "  \"stat_model_inertial_rel\": " << info.stat_model_inertial_rel << ",\n"

    << "  \"obj_scale\": " << info.obj_scale << ",\n"
    << "  \"best_feasible_obj\": " << (std::isfinite(info.best_feasible_obj) ? info.best_feasible_obj : 0.0) << ",\n"
    << "  \"time\": " << info.solve_time << ",\n"
    << "  \"message\": \"" << info.message << "\",\n"
    << "  \"note\": \"" << note << "\"\n"
    << "}\n";
}

int run_one(const ManifestEntry& ent, const std::string& decode_dir,
            const std::string& outdir, Settings st, bool use_manifest_lambda) {
  if (use_manifest_lambda && std::isfinite(ent.lambda_min_est))
    st.lambda_min_hint = ent.lambda_min_est;
  const std::string pdir = decode_dir + "/" + ent.tag;
  ::mkdir(decode_dir.c_str(), 0755);
  ::mkdir(pdir.c_str(), 0755);
  const std::string odir = outdir + "/" + ent.tag;
  ::mkdir(outdir.c_str(), 0755);
  ::mkdir(odir.c_str(), 0755);

  if (!file_exists(pdir + "/OUTSDIF.d")) {
    std::string cmd = "cd '" + pdir + "' && sifdecoder";
    for (const auto& t2 : ent.params) cmd += " -param " + t2;
    cmd += " " + ent.name + " > sifdecoder.log 2>&1";
    if (std::system(cmd.c_str()) != 0 || !file_exists(pdir + "/OUTSDIF.d")) {
      std::fprintf(stderr, "[%s] sifdecoder failed (see %s/sifdecoder.log)\n",
                   ent.name.c_str(), pdir.c_str());
      return 1;
    }
  }
  if (!file_exists(pdir + "/libcutestprob.so")) {
    const char* ce = std::getenv("CUTEST"); const char* ma = std::getenv("MYARCH");
    if (!ce || !ma) { std::fprintf(stderr, "[%s] CUTEST/MYARCH non exportes\n", ent.tag.c_str()); return 1; }
    std::string cmd = "cd '" + pdir + "' && gfortran -fPIC -O2 -shared *.f "
        "-o libcutestprob.so -Wl,--whole-archive " + std::string(ce) +
        "/objects/" + std::string(ma) + "/double/libcutest.a -Wl,--no-whole-archive"
        " > gfortran.log 2>&1";
    if (std::system(cmd.c_str()) != 0 || !file_exists(pdir + "/libcutestprob.so")) {
      std::fprintf(stderr, "[%s] compilation du probleme echouee (voir %s/gfortran.log)\n",
                   ent.tag.c_str(), pdir.c_str());
      return 1;
    }
  }
  if (::chdir(pdir.c_str()) != 0) return 1;
  CutestApi api;
  { std::string err;
    if (!api.load(pdir + "/libcutestprob.so", err)) {
      std::fprintf(stderr, "[%s] dlopen: %s\n", ent.tag.c_str(), err.c_str()); return 1; } }

  integer funit = 42, ierr = 0, status = 0, iout = 6, io_buffer = 11;
  api.f_open(&funit, "OUTSDIF.d", &ierr);
  if (ierr != 0) { std::fprintf(stderr, "[%s] cannot open OUTSDIF.d\n", ent.name.c_str()); return 1; }

  integer nvar = 0, ncon = 0;
  api.cdimen(&status, &funit, &nvar, &ncon);
  const idx_t n = (idx_t)nvar, mc = (idx_t)ncon;

  std::vector<double> x(n), bl(n), bu(n), v(std::max<idx_t>(1, mc)),
      clo(std::max<idx_t>(1, mc)), cup(std::max<idx_t>(1, mc));
  std::unique_ptr<logical[]> equatn(new logical[std::max<idx_t>(1, mc)]());
  std::unique_ptr<logical[]> linear(new logical[std::max<idx_t>(1, mc)]());
  integer e_order = 0, l_order = 0, v_order = 0;
  api.csetup(&status, &funit, &iout, &io_buffer, &nvar, &ncon, x.data(),
                bl.data(), bu.data(), v.data(), clo.data(), cup.data(),
                equatn.get(), linear.get(), &e_order, &l_order, &v_order);

  // data at x = 0 (objective is quadratic, constraints are linear)
  std::vector<double> x0(n, 0.0), g(n, 0.0), y0(std::max<idx_t>(1, mc), 0.0);
  double f0 = 0.0;
  logical grad_flag = 1;
  api.cofg(&status, &nvar, x0.data(), &f0, g.data(), &grad_flag);

  integer nnzh = 0;
  api.cdimsh(&status, &nnzh);
  std::vector<double> Hval(std::max<integer>(1, nnzh));
  std::vector<integer> Hrow(std::max<integer>(1, nnzh)), Hcol(std::max<integer>(1, nnzh));
  integer lh = nnzh, nnzh_out = 0;
  api.csh(&status, &nvar, &ncon, x0.data(), y0.data(), &nnzh_out, &lh,
             Hval.data(), Hrow.data(), Hcol.data());

  integer nnzj = 0;
  api.cdimsj(&status, &nnzj);
  std::vector<double> c0(std::max<idx_t>(1, mc), 0.0), Jval(std::max<integer>(1, nnzj));
  std::vector<integer> Jvar(std::max<integer>(1, nnzj)), Jfun(std::max<integer>(1, nnzj));
  integer lj = nnzj, nnzj_out = 0;
  logical grad_j = 1;
  api.ccfsg(&status, &nvar, &ncon, x0.data(), c0.data(), &nnzj_out, &lj,
               Jval.data(), Jvar.data(), Jfun.data(), &grad_j);

  bool all_linear = true;
  for (idx_t i = 0; i < mc; ++i) if (!linear[i]) all_linear = false;

  // split rows: equalities -> A, inequalities -> C  (bounds shifted by c(0))
  std::vector<idx_t> row_kind(mc, 0), row_new(mc, 0);  // 0=eq, 1=ineq
  idx_t p = 0, m = 0;
  for (idx_t i = 0; i < mc; ++i) {
    if (equatn[i]) { row_kind[i] = 0; row_new[i] = p++; }
    else { row_kind[i] = 1; row_new[i] = m++; }
  }
  Problem P;
  P.n = n; P.p = p; P.m = m; P.name = ent.name; P.c0 = f0;
  P.q.assign(g.begin(), g.end());
  {
    std::vector<idx_t> I, J; std::vector<double> V;
    I.reserve(nnzh_out); J.reserve(nnzh_out); V.reserve(nnzh_out);
    for (integer k = 0; k < nnzh_out; ++k) {
      idx_t i = (idx_t)Hrow[k] - 1, j = (idx_t)Hcol[k] - 1;
      I.push_back(std::min(i, j)); J.push_back(std::max(i, j)); V.push_back(Hval[k]);
    }
    P.Q.U = csc_from_coo(n, n, I, J, V);
  }
  {
    std::vector<idx_t> AI, AJ, CI, CJ; std::vector<double> AV, CV;
    for (integer k = 0; k < nnzj_out; ++k) {
      idx_t r = (idx_t)Jfun[k] - 1, col = (idx_t)Jvar[k] - 1;
      if (r < 0) continue;  // Jfun==0 rows are objective gradient entries
      if (row_kind[r] == 0) { AI.push_back(row_new[r]); AJ.push_back(col); AV.push_back(Jval[k]); }
      else { CI.push_back(row_new[r]); CJ.push_back(col); CV.push_back(Jval[k]); }
    }
    P.A = csc_from_coo(p, n, AI, AJ, AV);
    P.C = csc_from_coo(m, n, CI, CJ, CV);
  }
  P.b.assign(p, 0.0); P.cl.assign(m, -INF); P.cu.assign(m, INF);
  for (idx_t i = 0; i < mc; ++i) {
    if (row_kind[i] == 0) P.b[row_new[i]] = debound(clo[i]) - c0[i];
    else {
      double lo = debound(clo[i]), hi = debound(cup[i]);
      P.cl[row_new[i]] = std::isfinite(lo) ? lo - c0[i] : -INF;
      P.cu[row_new[i]] = std::isfinite(hi) ? hi - c0[i] : INF;
    }
  }
  P.l.resize(n); P.u.resize(n);
  for (idx_t i = 0; i < n; ++i) { P.l[i] = debound(bl[i]); P.u[i] = debound(bu[i]); }

  api.cterminate(&status);
  api.f_close(&funit, &ierr);

  std::printf("[runner] %s: n=%lld p=%lld m=%lld nnz(Q)=%lld nnz(J)=%lld%s\n",
              ent.tag.c_str(), (long long)n, (long long)p, (long long)m,
              (long long)P.Q.U.nnz(), (long long)(P.A.nnz() + P.C.nnz()),
              all_linear ? "" : "  [WARNING: nonlinear constraints linearized at 0]");

  Result r = solve(P, st);
  write_json(odir + "/" + ent.tag + "_raw.json", ent.tag, n, p, m, r.info, st,
             all_linear ? "" : "nonlinear constraints linearized at x=0");
  return (r.info.status == Status::Solved || r.info.status == Status::SolvedSOC) ? 0
       : (r.info.status == Status::DualDegenerate) ? 2
       : (r.info.status == Status::FactorizationBudget) ? 3
       : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string manifest, outdir = "results", decode_dir = "/tmp/lazyipdca_decode", only;
  bool only_accepted = false, use_manifest_lambda = false;
  Settings st; st.verbose = 1; st.max_iter = 3000;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (a == "--manifest") manifest = next();
    else if (a == "--outdir") outdir = next();
    else if (a == "--decode-dir") decode_dir = next();
    else if (a == "--problem") only = next();
    else if (a == "--max-iter") st.max_iter = std::atoll(next().c_str());
    else if (a == "--time-limit") st.time_limit = std::atof(next().c_str());
    else if (a == "--eps") { double e = std::atof(next().c_str()); st.eps_stat = st.eps_feas = st.eps_comp = e; }
    else if (a == "--verbose") st.verbose = std::atoi(next().c_str());
    else if (a == "--only-accepted") only_accepted = true;
    else if (a == "--use-manifest-lambda") use_manifest_lambda = true;
    else if (a == "--convdesc") st.accept_convex_descent = true;
    else if (a == "--init") st.init_strategy = std::atoi(next().c_str());
    else if (a == "--regularization-mode" || a == "--reg-mode") {
      const std::string v = next();
      if (v == "legacy") st.regularization_mode = DcRegularizationMode::Legacy;
      else if (v == "const" || v == "fixed") st.regularization_mode = DcRegularizationMode::Const;
      else if (v == "dyn" || v == "dynamic") st.regularization_mode = DcRegularizationMode::Dyn;
      else if (v == "lazy" || v == "lowrank" || v == "low-rank") st.regularization_mode = DcRegularizationMode::Lazy;
      else {
        std::fprintf(stderr, "[runner] ERREUR : --regularization-mode %s inconnu. Valeurs: legacy, const, dyn, lazy.\n", v.c_str());
        return 2;
      }
    }
    else if (a == "--const") st.regularization_mode = DcRegularizationMode::Const;
    else if (a == "--dyn") st.regularization_mode = DcRegularizationMode::Dyn;
    else if (a == "--lazy-mode") st.regularization_mode = DcRegularizationMode::Lazy;
    else if (a == "--rho-init") {
      const std::string v = next();
      if (v == "auto") st.rho_init_method = RhoInitMethod::Auto;
      else if (v == "user") st.rho_init_method = RhoInitMethod::User;
      else if (v == "lanczos") st.rho_init_method = RhoInitMethod::Lanczos;
      else if (v == "gershgorin") st.rho_init_method = RhoInitMethod::Gershgorin;
      else if (v == "hint") st.rho_init_method = RhoInitMethod::Hint;
      else {
        std::fprintf(stderr, "[runner] ERREUR : --rho-init %s inconnu. Valeurs: auto, user, lanczos, gershgorin, hint.\n", v.c_str());
        return 2;
      }
    }
    else if (a == "--rho0") st.rho0 = std::atof(next().c_str());
    else if (a == "--rho-init-rel-margin") st.rho_init_rel_margin = std::atof(next().c_str());
    else if (a == "--rho-init-abs-margin") st.rho_init_abs_margin = std::atof(next().c_str());
    else if (a == "--rho-curvature-tol") st.rho_curvature_tol = std::atof(next().c_str());
    else if (a == "--rho-lanczos-residual-safety") st.rho_lanczos_residual_safety = std::atof(next().c_str());
    else if (a == "--rho-dyn-floor-rel-margin") st.rho_dyn_floor_rel_margin = std::atof(next().c_str());
    else if (a == "--rho-dyn-reject-growth") st.rho_dyn_reject_growth = std::atof(next().c_str());
    else if (a == "--rho-model-min") st.rho_model_min = std::atof(next().c_str());
    else if (a == "--rho-model-lock") st.rho_model_lock = true;
    else if (a == "--rho-floor") st.rho_floor = std::atof(next().c_str());
    else if (a == "--rho-growth") st.rho_growth = std::atof(next().c_str());
    else if (a == "--rho-rel-margin") st.rho_rel_margin = std::atof(next().c_str());
    else if (a == "--rho-decay") st.rho_decay = std::atof(next().c_str());
    else if (a == "--rho-clean-streak") st.rho_clean_streak = std::atoll(next().c_str());
    else if (a == "--atoms") st.atoms_enable = true;
    else if (a == "--no-atoms") st.atoms_enable = false;
    else if (a == "--max-kkt-factorizations") st.max_kkt_factorizations = std::atoll(next().c_str());
    else if (a == "--max-atoms") st.max_atoms = std::atoll(next().c_str());
    else if (a == "--initial-atoms") st.initial_atoms = std::atoll(next().c_str());
    else if (a == "--atoms-per-update") st.atoms_per_update = std::atoll(next().c_str());
    else if (a == "--atom-update-every") st.atom_update_every_accepts = std::atoll(next().c_str());
    else if (a == "--atom-every-iters") st.atom_update_every_iters = std::atoll(next().c_str());
    else if (a == "--atom-selection") st.atom_selection_enable = true;
    else if (a == "--atom-criterion") {
      st.atom_selection_criterion = std::atoll(next().c_str());
      if (st.atom_selection_criterion < 0 || st.atom_selection_criterion > 2) {
        std::fprintf(stderr, "[runner] ERREUR : --atom-criterion %lld inconnu (0 = Rayleigh, 1 = delta-lambda, 2 = lift/beta).\n",
                     (long long)st.atom_selection_criterion);
        return 1;
      }
    }
    else if (a == "--seed-coverage") st.atoms_seed_coverage = std::atof(next().c_str());
    else if (a == "--max-seed-atoms") st.atoms_seed_max_atoms = std::atoll(next().c_str());
    else if (a == "--freeze-atoms-after-seed") st.atoms_freeze_after_seed = true;
    else if (a == "--no-freeze-atoms-after-seed") st.atoms_freeze_after_seed = false;
    else if (a == "--seed-stall-tol") st.atoms_seed_stall_tol = std::atof(next().c_str());
    else if (a == "--seed-patience") st.atoms_seed_stall_patience = std::atoll(next().c_str());
    else if (a == "--bound-push") st.bound_push = std::atof(next().c_str());
    else if (a == "--bound-frac") st.bound_frac = std::atof(next().c_str());
    else if (a == "--bound-mult-init") st.bound_mult_init_val = std::atof(next().c_str());
    else if (a == "--bp-adaptive") st.bp_adaptive = true;
    else if (a == "--no-bp-adaptive") st.bp_adaptive = false;
    else if (a == "--bp-slack-tol") st.bp_slack_tol = std::atof(next().c_str());
    else if (a == "--bp-margin-tol") st.bp_margin_tol = std::atof(next().c_str());
    else if (a == "--bp-push-kappa") st.bp_push_kappa = std::atof(next().c_str());
    else if (a == "--bp-kappa-init") st.bp_kappa_init = std::atof(next().c_str());
    else if (a == "--bp-kappa-mult") st.bp_kappa_mult = std::atof(next().c_str());
    else if (a == "--bp-kappa-max") st.bp_kappa_max = std::atof(next().c_str());
    else if (a == "--bp-max-events") st.bp_max_events = std::atoll(next().c_str());
    else if (a == "--dca-stop") st.dca_stop_enable = true;
    else if (a == "--no-dca-stop") st.dca_stop_enable = false;
    else if (a == "--dca-stall-tol-abs") st.dca_stall_tol_abs = std::atof(next().c_str());
    else if (a == "--dca-stall-tol-rel") st.dca_stall_tol_rel = std::atof(next().c_str());
    else if (a == "--dca-stall-iters") st.dca_stall_iters = std::atoll(next().c_str());
    else if (a == "--dca-monotone-tol-abs") st.dca_monotone_tol_abs = std::atof(next().c_str());
    else if (a == "--dca-monotone-tol-rel") st.dca_monotone_tol_rel = std::atof(next().c_str());
    else if (a == "--dca-smooth-window") st.dca_smooth_window = std::atoll(next().c_str());
    else if (a == "--dca-momentum") {
      const std::string v = next();
      if (v == "original" || v == "none" || v == "off") st.dca_momentum = DcaMomentumMode::Original;
      else if (v == "inertiel_euclidien" || v == "euclidien" || v == "euclidean") st.dca_momentum = DcaMomentumMode::InertialEuclidean;
      else if (v == "inertiel_global" || v == "global") st.dca_momentum = DcaMomentumMode::InertialGlobal;
      else if (v == "inertiel_atomique" || v == "atomique" || v == "atomic") st.dca_momentum = DcaMomentumMode::InertialAtomic;
      else { std::fprintf(stderr, "[runner] ERREUR : --dca-momentum %s inconnu. Valeurs: original, inertiel_euclidien, inertiel_global, inertiel_atomique.\n", v.c_str()); return 2; }
    }
    else if (a == "--dca-acceptance") {
      const std::string v = next();
      if (v == "monotone") st.dca_acceptance = DcaAcceptanceMode::Monotone;
      else if (v == "window") st.dca_acceptance = DcaAcceptanceMode::Window;
      else if (v == "lyapunov") st.dca_acceptance = DcaAcceptanceMode::Lyapunov;
      else { std::fprintf(stderr, "[runner] ERREUR : --dca-acceptance %s inconnu. Valeurs: monotone, window, lyapunov.\n", v.c_str()); return 2; }
    }
    else if (a == "--momentum-theta") st.dca_momentum_theta = std::atof(next().c_str());
    else if (a == "--momentum-theta-min") st.dca_momentum_theta_min = std::atof(next().c_str());
    else if (a == "--momentum-theta-max") st.dca_momentum_theta_max = std::atof(next().c_str());
    else if (a == "--momentum-backtrack") st.dca_momentum_backtrack = std::atof(next().c_str());
    else if (a == "--momentum-growth") st.dca_momentum_growth = std::atof(next().c_str());
    else if (a == "--momentum-start") st.dca_momentum_start = std::atoll(next().c_str());
    else if (a == "--momentum-gamma-mode") {
      const std::string v = next();
      if (v == "fixed") st.dca_momentum_gamma_mode = DcaEuclideanGammaMode::Fixed;
      else if (v == "rayleigh" || v == "metric-rayleigh") st.dca_momentum_gamma_mode = DcaEuclideanGammaMode::MetricRayleigh;
      else { std::fprintf(stderr, "[runner] ERREUR : --momentum-gamma-mode %s inconnu. Valeurs: fixed, rayleigh.\n", v.c_str()); return 2; }
    }
    else if (a == "--momentum-gamma") st.dca_momentum_gamma = std::atof(next().c_str());
    else if (a == "--momentum-gamma-min") st.dca_momentum_gamma_min = std::atof(next().c_str());
    else if (a == "--momentum-gamma-max") st.dca_momentum_gamma_max = std::atof(next().c_str());
    else if (a == "--momentum-rho-factor") st.dca_momentum_rho_factor = std::atof(next().c_str());
    else if (a == "--momentum-atom-align-floor") st.dca_momentum_atom_align_floor = std::atof(next().c_str());
    else if (a == "--momentum-atom-align-power") st.dca_momentum_atom_align_power = std::atof(next().c_str());
    else if (a == "--momentum-reset-on-metric-change") st.dca_momentum_reset_on_metric_change = true;
    else if (a == "--no-momentum-reset-on-metric-change") st.dca_momentum_reset_on_metric_change = false;
    else if (a == "--momentum-complete") st.dca_momentum_complete = true;
    else if (a == "--no-momentum-complete") st.dca_momentum_complete = false;
    else if (a == "--momentum-atom-weighting") { const std::string v=next(); if(v=="uniform")st.dca_momentum_atom_weighting=DcaMomentumAtomWeighting::Uniform; else if(v=="cosine")st.dca_momentum_atom_weighting=DcaMomentumAtomWeighting::Cosine; else if(v=="energy")st.dca_momentum_atom_weighting=DcaMomentumAtomWeighting::Energy; else {std::fprintf(stderr,"[runner] invalid atom weighting %s\n",v.c_str());return 2;} }
    else if (a == "--momentum-rho-policy") { const std::string v=next(); if(v=="restart")st.dca_momentum_rho_policy=DcaMomentumRhoPolicy::Restart; else if(v=="keep")st.dca_momentum_rho_policy=DcaMomentumRhoPolicy::Keep; else if(v=="transport")st.dca_momentum_rho_policy=DcaMomentumRhoPolicy::Transport; else {std::fprintf(stderr,"[runner] invalid rho momentum policy %s\n",v.c_str());return 2;} }
    else if (a == "--momentum-selective-restart") st.dca_momentum_selective_restart=true;
    else if (a == "--momentum-full-restart") st.dca_momentum_selective_restart=false;
    else if (a == "--momentum-energy-weight-rho") st.dca_momentum_energy_weight_rho=true;
    else if (a == "--momentum-energy-no-rho-weight") st.dca_momentum_energy_weight_rho=false;
    else if (a == "--momentum-theta-inc") st.dca_momentum_theta_inc=std::atof(next().c_str());
    else if (a == "--momentum-align-threshold") st.dca_momentum_align_threshold=std::atof(next().c_str());
    else if (a == "--momentum-align-decay") st.dca_momentum_align_decay=std::atof(next().c_str());
    else if (a == "--momentum-force-rel-max") st.dca_momentum_force_rel_max=std::atof(next().c_str());
    else if (a == "--momentum-stat-guard-factor") st.dca_momentum_stat_guard_factor=std::atof(next().c_str());
    else if (a == "--lyapunov-cap-momentum-theta") st.dca_lyapunov_cap_momentum_theta=true;
    else if (a == "--no-lyapunov-cap-momentum-theta") st.dca_lyapunov_cap_momentum_theta=false;
    else if (a == "--lyapunov-enforce-euclidean-bound") st.dca_lyapunov_enforce_euclidean_bound=true;
    else if (a == "--lyapunov-unsafe-euclidean") st.dca_lyapunov_unsafe_euclidean=true;

    else if (a == "--accept-window") st.dca_accept_window = std::atoll(next().c_str());
    else if (a == "--lyapunov-theta-bar") st.dca_lyapunov_theta_bar = std::atof(next().c_str());
    else if (a == "--lyapunov-eta") st.dca_lyapunov_eta = std::atof(next().c_str());
    else if (a == "--sigma-max") st.sigma_max = std::atof(next().c_str());
    else if (a == "--heartbeat-every") st.heartbeat_every = std::atoll(next().c_str());
    else if (a == "--reject-streak-rho-bump") st.reject_streak_rho_bump = std::atoll(next().c_str());
    else if (a == "--rho-abs-max") st.rho_abs_max = std::atof(next().c_str());
    else if (a == "--model-solved-tol") st.model_solved_tol = std::atof(next().c_str());
    else if (a == "--ipopt-stop") st.ipopt_stop = true;
    else if (a == "--ipopt-tol") st.ipopt_tol = std::atof(next().c_str());
    else if (a == "--ipopt-s-max") st.ipopt_s_max = std::atof(next().c_str());
    else if (a == "--ipopt-dual-inf-tol") st.ipopt_dual_inf_tol = std::atof(next().c_str());
    else if (a == "--ipopt-constr-viol-tol") st.ipopt_constr_viol_tol = std::atof(next().c_str());
    else if (a == "--ipopt-compl-inf-tol") st.ipopt_compl_inf_tol = std::atof(next().c_str());
    else if (a == "--ipopt-acceptable-tol") st.ipopt_acceptable_tol = std::atof(next().c_str());
    else if (a == "--ipopt-acceptable-iter") st.ipopt_acceptable_iter = std::atoll(next().c_str());
    else if (a == "--ipopt-smooth-window") st.ipopt_smooth_window = std::atoll(next().c_str());
    else if (a == "--seed-regress-patience") st.atoms_seed_regress_patience = std::atoll(next().c_str());
    else if (a == "--atom-events") st.atom_event_enable = true;
    else if (a == "--no-atom-events") st.atom_event_enable = false;
    else if (a == "--atom-event-every") st.atom_event_every = std::atoll(next().c_str());
    // (verification post-parse plus bas : reglages seed-* sans --seed-coverage)
    else if (a == "--rho-trial-every") st.rho_trial_every = std::atoll(next().c_str());
    else if (a == "--lanczos-maxit") st.lanczos_maxit = std::atoll(next().c_str());
    else if (a == "--lanczos-tol") st.lanczos_tol = std::atof(next().c_str());
    else if (a == "--mehrotra") st.mehrotra_enable = true;
    else if (a == "--no-mehrotra") st.mehrotra_enable = false;
    else if (a == "--no-anneal") st.mehrotra_anneal = false;
    else if (a == "--escapes") st.soc_escape_max = std::atoll(next().c_str());
    else if (a == "--rho-dynamic") st.rho_dynamic = true;
    else if (a == "--no-rho-dynamic") st.rho_dynamic = false;
    else if (a == "--atom-max-nnz") st.atom_max_nnz = std::atoll(next().c_str());
    else if (a == "--atom-sparse-recertify-beta") st.atom_sparse_recertify_beta = true;
    else if (a == "--no-atom-sparse-recertify-beta") st.atom_sparse_recertify_beta = false;
    else if (a == "--atom-sparse-recertify-tol-rel") st.atom_sparse_recertify_tol_rel = std::atof(next().c_str());
    else if (a == "--atom-sparse-recertify-maxit") st.atom_sparse_recertify_maxit = std::atoll(next().c_str());
    else if (a == "--atom-sparse-beta-max") st.atom_sparse_beta_max = std::atof(next().c_str());
    else if (a == "--auto-atoms") {
      // AUTO always starts from the exact legacy v0.18.1 atom policy.
      st.auto_atoms_enable = true;
      st.atoms_enable = true;
      st.atom_oracle_enable = false;
      st.atom_energy_sparsify = false;
      st.atom_oracle_retune_enable = false;
      st.atom_oracle_rhox_rollback = false;
    }
    else if (a == "--no-auto-atoms") st.auto_atoms_enable = false;
    else if (a == "--auto-coverage-trigger")
      st.auto_coverage_trigger = std::atof(next().c_str());
    else if (a == "--adaptive-atoms") { st.atoms_enable = true; st.atom_oracle_enable = true; st.atom_energy_sparsify = true; st.atom_oracle_retune_enable = true; }
    // ADAPTIVE_MARGINAL_SEED_V2
    else if (a == "--adaptive-seed-return") st.adaptive_seed_return_enable = true;
    else if (a == "--no-adaptive-seed-return") st.adaptive_seed_return_enable = false;
    else if (a == "--seed-return-warmup") st.adaptive_seed_return_warmup = std::atoll(next().c_str());
    else if (a == "--seed-return-window") st.adaptive_seed_return_window = std::atoll(next().c_str());
    else if (a == "--seed-return-confirmations") st.adaptive_seed_return_confirmations = std::atoll(next().c_str());
    else if (a == "--seed-return-alpha-coverage") st.adaptive_seed_return_alpha_coverage = std::atof(next().c_str());
    else if (a == "--seed-return-alpha-lift") st.adaptive_seed_return_alpha_lift = std::atof(next().c_str());
    else if (a == "--seed-return-coverage-min") st.adaptive_seed_return_coverage_min = std::atof(next().c_str());
    else if (a == "--seed-return-coverage-max") st.adaptive_seed_return_coverage_max = std::atof(next().c_str());
    else if (a == "--seed-return-lift-min") st.adaptive_seed_return_lift_min = std::atof(next().c_str());
    else if (a == "--seed-return-lift-max") st.adaptive_seed_return_lift_max = std::atof(next().c_str());
    // ADAPTIVE_MARGINAL_RETURN_V21
    else if (a == "--adaptive-marginal-return") st.adaptive_marginal_return_enable = true;
    else if (a == "--no-adaptive-marginal-return") st.adaptive_marginal_return_enable = false;
    else if (a == "--adaptive-return-warmup") st.adaptive_return_warmup = std::atoll(next().c_str());
    else if (a == "--adaptive-return-window") st.adaptive_return_window = std::atoll(next().c_str());
    else if (a == "--adaptive-return-confirmations") st.adaptive_return_confirmations = std::atoll(next().c_str());
    else if (a == "--adaptive-return-alpha-lift") st.adaptive_return_alpha_lift = std::atof(next().c_str());
    else if (a == "--adaptive-return-alpha-effective") st.adaptive_return_alpha_effective = std::atof(next().c_str());
    else if (a == "--adaptive-return-lift-min") st.adaptive_return_lift_min = std::atof(next().c_str());
    else if (a == "--adaptive-return-lift-max") st.adaptive_return_lift_max = std::atof(next().c_str());
    else if (a == "--adaptive-return-effective-min") st.adaptive_return_effective_min = std::atof(next().c_str());
    else if (a == "--adaptive-return-effective-max") st.adaptive_return_effective_max = std::atof(next().c_str());
    else if (a == "--adaptive-return-cooldown-factor") st.adaptive_return_cooldown_factor = std::atof(next().c_str());
    else if (a == "--adaptive-return-cooldown-max-factor") st.adaptive_return_cooldown_max_factor = std::atof(next().c_str());
    else if (a == "--adaptive-return-strong-factor") st.adaptive_return_strong_factor = std::atof(next().c_str());
    // ADAPTIVE_MARGINAL_RETURN_V22 seed policy
    else if (a == "--adaptive-return-seed-alpha-coverage") st.adaptive_return_seed_alpha_coverage = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-alpha-lift") st.adaptive_return_seed_alpha_lift = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-coverage-min") st.adaptive_return_seed_coverage_min = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-coverage-max") st.adaptive_return_seed_coverage_max = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-lift-min") st.adaptive_return_seed_lift_min = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-lift-max") st.adaptive_return_seed_lift_max = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-absolute") st.adaptive_return_seed_absolute_enable = true;
    else if (a == "--no-adaptive-return-seed-absolute") st.adaptive_return_seed_absolute_enable = false;
    else if (a == "--adaptive-return-seed-absolute-window") st.adaptive_return_seed_absolute_window = std::atoll(next().c_str());
    else if (a == "--adaptive-return-seed-absolute-confirmations") st.adaptive_return_seed_absolute_confirmations = std::atoll(next().c_str());
    else if (a == "--adaptive-return-seed-absolute-coverage") st.adaptive_return_seed_absolute_coverage = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-absolute-lift") st.adaptive_return_seed_absolute_lift = std::atof(next().c_str());
    // ADAPTIVE_MARGINAL_RETURN_V23 robust seed plateau
    else if (a == "--adaptive-return-seed-plateau") st.adaptive_return_seed_plateau_enable = true;
    else if (a == "--no-adaptive-return-seed-plateau") st.adaptive_return_seed_plateau_enable = false;
    else if (a == "--adaptive-return-seed-plateau-horizon") st.adaptive_return_seed_plateau_horizon = std::atoll(next().c_str());
    else if (a == "--adaptive-return-seed-plateau-min-commits") st.adaptive_return_seed_plateau_min_commits = std::atoll(next().c_str());
    else if (a == "--adaptive-return-seed-plateau-confirmations") st.adaptive_return_seed_plateau_confirmations = std::atoll(next().c_str());
    else if (a == "--adaptive-return-seed-plateau-coverage") st.adaptive_return_seed_plateau_coverage_per_commit = std::atof(next().c_str());
    else if (a == "--adaptive-return-seed-plateau-lift") st.adaptive_return_seed_plateau_lift = std::atof(next().c_str());




    // ATOM_CLUSTER_PROD_V1
    else if (a == "--atom-cluster") st.atom_cluster_enable = true;
    else if (a == "--no-atom-cluster") st.atom_cluster_enable = false;
    else if (a == "--atom-cluster-w-direction") st.atom_cluster_w_direction = std::atof(next().c_str());
    else if (a == "--atom-cluster-w-support") st.atom_cluster_w_support = std::atof(next().c_str());
    else if (a == "--atom-cluster-w-curvature") st.atom_cluster_w_curvature = std::atof(next().c_str());
    else if (a == "--atom-cluster-tau-in") st.atom_cluster_tau_in = std::atof(next().c_str());
    else if (a == "--atom-cluster-tau-out") st.atom_cluster_tau_out = std::atof(next().c_str());
    else if (a == "--atom-cluster-adaptive-thresholds") st.atom_cluster_adaptive_thresholds = true;
    else if (a == "--no-atom-cluster-adaptive-thresholds") st.atom_cluster_adaptive_thresholds = false;
    else if (a == "--atom-cluster-adapt-min-samples") st.atom_cluster_adapt_min_samples = std::atoll(next().c_str());
    else if (a == "--atom-cluster-adapt-window") st.atom_cluster_adapt_window = std::atoll(next().c_str());
    else if (a == "--atom-cluster-adapt-strength") st.atom_cluster_adapt_strength = std::atof(next().c_str());
    else if (a == "--atom-cluster-max-reps") st.atom_cluster_max_reps = std::atoll(next().c_str());
    else if (a == "--atom-cluster-replace-rel-quality") st.atom_cluster_replace_rel_quality = std::atof(next().c_str());
    else if (a == "--atom-cluster-replace-min-similarity") st.atom_cluster_replace_min_similarity = std::atof(next().c_str());
    else if (a == "--atom-cluster-duplicate-penalty") st.atom_cluster_duplicate_penalty = std::atof(next().c_str());
    else if (a == "--atom-cluster-novelty-bonus") st.atom_cluster_novelty_bonus = std::atof(next().c_str());
    else if (a == "--atom-cluster-stop-window") st.atom_cluster_stop_window = std::atoll(next().c_str());
    else if (a == "--atom-cluster-stop-min-observations") st.atom_cluster_stop_min_observations = std::atoll(next().c_str());
    else if (a == "--atom-cluster-stop-min-atoms") st.atom_cluster_stop_min_atoms = std::atoll(next().c_str());
    else if (a == "--atom-cluster-stop-max-novelty-rate") st.atom_cluster_stop_max_novelty_rate = std::atof(next().c_str());
    else if (a == "--atom-cluster-stop-max-coverage-gain") st.atom_cluster_stop_max_coverage_gain = std::atof(next().c_str());
    else if (a == "--atom-cluster-stop-max-lift-rel") st.atom_cluster_stop_max_lift_rel = std::atof(next().c_str());
    else if (a == "--atom-cluster-stop-confirmations") st.atom_cluster_stop_confirmation_rounds = std::atoll(next().c_str());

    else if (a == "--legacy-atoms") {
      // TRUE historical v0.18.1 atomic campaign policy.
      // Keep campaign axes such as max_atoms and seed_coverage independent.
      st.atoms_enable = true;

      // No modern adaptive oracle machinery.
      st.atom_oracle_enable = false;
      st.atom_energy_sparsify = false;
      st.atom_oracle_retune_enable = false;
      st.atom_oracle_rhox_rollback = false;
      st.atom_selection_enable = false;
      st.atom_sparse_recertify_beta = false;

      // Historical campaign atom-event doctrine.
      st.atom_event_enable = true;
      st.atom_event_every = 50;
      st.atoms_per_update = 1;
      st.atom_update_every_accepts = 20;
      st.atom_update_every_iters = 10;
      st.atoms_freeze_after_seed = false;

      // Legacy is mutually exclusive with AUTO.
      st.auto_atoms_enable = false;
      st.auto_portfolio_enable = false;
    }
    else if (a == "--scalar-atoms") { st.atoms_enable = false; st.atom_oracle_enable = false; st.atom_energy_sparsify = false; st.atom_oracle_retune_enable = false; st.atom_oracle_rhox_rollback = false; }
    else if (a == "--atom-oracle") st.atom_oracle_enable = true;
    else if (a == "--no-atom-oracle") st.atom_oracle_enable = false;
    else if (a == "--atom-oracle-pool") st.atom_oracle_pool_size = std::atoll(next().c_str());
    else if (a == "--atom-oracle-activity-mode") st.atom_oracle_activity_mode = std::atoll(next().c_str());
    else if (a == "--atom-oracle-activity-weight") st.atom_oracle_activity_weight = std::atof(next().c_str());
    else if (a == "--atom-oracle-density-weight") st.atom_oracle_density_weight = std::atof(next().c_str());
    else if (a == "--atom-oracle-coupling-weight") st.atom_oracle_coupling_weight = std::atof(next().c_str());
    else if (a == "--atom-oracle-beta-weight") st.atom_oracle_beta_weight = std::atof(next().c_str());
    else if (a == "--atom-oracle-directional-weight") st.atom_oracle_directional_weight = std::atof(next().c_str());
    else if (a == "--atom-oracle-min-lift-rel") st.atom_oracle_min_lift_rel = std::atof(next().c_str());
    else if (a == "--atom-oracle-min-activity") st.atom_oracle_min_activity = std::atof(next().c_str());
    else if (a == "--atom-oracle-min-score") st.atom_oracle_min_score = std::atof(next().c_str());
    else if (a == "--atom-oracle-min-directional-ratio") st.atom_oracle_min_directional_ratio = std::atof(next().c_str());
    else if (a == "--no-atom-oracle-scalar-fallback") st.atom_oracle_scalar_fallback = false;
    else if (a == "--atom-energy-sparsify") st.atom_energy_sparsify = true;
    else if (a == "--no-atom-energy-sparsify") st.atom_energy_sparsify = false;
    else if (a == "--atom-energy-target") st.atom_energy_target = std::atof(next().c_str());
    else if (a == "--atom-min-nnz") st.atom_min_nnz = std::atoll(next().c_str());
    else if (a == "--atom-oracle-retune") st.atom_oracle_retune_enable = true;
    else if (a == "--no-atom-oracle-retune") st.atom_oracle_retune_enable = false;
    else if (a == "--atom-oracle-brake-fraction") st.atom_oracle_brake_fraction = std::atof(next().c_str());
    else if (a == "--atom-oracle-brake-patience") st.atom_oracle_brake_patience = std::atoll(next().c_str());
    else if (a == "--atom-oracle-rhox-rollback") st.atom_oracle_rhox_rollback = true;
    else if (a == "--atom-oracle-rhox-ratio") st.atom_oracle_rhox_ratio_trigger = std::atof(next().c_str());
    else if (a == "--sd-abs") st.sd_abs = std::atof(next().c_str());
    else if (a == "--sd-rel") st.sd_rel = std::atof(next().c_str());
    else if (a == "--sd-sigma") st.sd_sigma = std::atof(next().c_str());
    else if (a == "--cv-tol") st.sd_cv_abs = std::atof(next().c_str());
    else if (a == "--backend") { std::string b = next(); st.linear_backend = (b == "mumps") ? 1 : (b == "ldl") ? 2 : (b == "qdldl" || b == "piqp") ? 3 : 0; }
    else if (a == "--no-scaling") st.scaling_enable = false;
    else if (a == "--scaling-max-grad") st.scaling_max_grad = std::atof(next().c_str());

    else if (a == "--no-reuse") st.kkt_reuse = false;
    else if (a == "--reuse-max") st.kkt_reuse_max = std::atoll(next().c_str());
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }

  // ADAPTIVE_MARGINAL_RETURN_V23 parameter validation.
  if (st.adaptive_return_seed_plateau_horizon < 1 ||
      st.adaptive_return_seed_plateau_min_commits < st.adaptive_return_seed_plateau_horizon + 1 ||
      st.adaptive_return_seed_plateau_confirmations < 1 ||
      st.adaptive_return_seed_plateau_coverage_per_commit < 0.0 ||
      st.adaptive_return_seed_plateau_lift < 0.0) {
    std::fprintf(stderr, "[lazyipdca] invalid ADAPTIVE_MARGINAL_RETURN_V23 parameters\n");
    return 2;
  }
  // ADAPTIVE_MARGINAL_RETURN_V22 parameter validation.
  if (st.adaptive_return_seed_alpha_coverage < 0.0 || st.adaptive_return_seed_alpha_coverage > 1.0 ||
      st.adaptive_return_seed_alpha_lift < 0.0 || st.adaptive_return_seed_alpha_lift > 1.0 ||
      st.adaptive_return_seed_coverage_min < 0.0 ||
      st.adaptive_return_seed_coverage_max < st.adaptive_return_seed_coverage_min ||
      st.adaptive_return_seed_lift_min < 0.0 ||
      st.adaptive_return_seed_lift_max < st.adaptive_return_seed_lift_min ||
      st.adaptive_return_seed_absolute_window < 1 ||
      st.adaptive_return_seed_absolute_confirmations < 1 ||
      st.adaptive_return_seed_absolute_coverage < 0.0 ||
      st.adaptive_return_seed_absolute_lift < 0.0) {
    std::fprintf(stderr, "[lazyipdca] invalid ADAPTIVE_MARGINAL_RETURN_V22 parameters\n");
    return 2;
  }
  // ADAPTIVE_MARGINAL_RETURN_V21 parameter validation.
  if (st.adaptive_return_warmup < 1 || st.adaptive_return_window < 1 ||
      st.adaptive_return_confirmations < 1 ||
      st.adaptive_return_alpha_lift < 0.0 || st.adaptive_return_alpha_lift > 1.0 ||
      st.adaptive_return_alpha_effective < 0.0 || st.adaptive_return_alpha_effective > 1.0 ||
      st.adaptive_return_lift_min < 0.0 ||
      st.adaptive_return_lift_max < st.adaptive_return_lift_min ||
      st.adaptive_return_effective_min < 0.0 ||
      st.adaptive_return_effective_max < st.adaptive_return_effective_min ||
      st.adaptive_return_cooldown_factor < 1.0 ||
      st.adaptive_return_cooldown_max_factor < st.adaptive_return_cooldown_factor ||
      st.adaptive_return_strong_factor < 1.0) {
    std::fprintf(stderr, "[lazyipdca] invalid ADAPTIVE_MARGINAL_RETURN_V21 parameters\n");
    return 2;
  }
  // ADAPTIVE_MARGINAL_SEED_V2 parameter validation.
  if (st.adaptive_seed_return_warmup < 1 || st.adaptive_seed_return_window < 1 ||
      st.adaptive_seed_return_confirmations < 1 ||
      st.adaptive_seed_return_alpha_coverage < 0.0 || st.adaptive_seed_return_alpha_coverage > 1.0 ||
      st.adaptive_seed_return_alpha_lift < 0.0 || st.adaptive_seed_return_alpha_lift > 1.0 ||
      st.adaptive_seed_return_coverage_min < 0.0 ||
      st.adaptive_seed_return_coverage_max < st.adaptive_seed_return_coverage_min ||
      st.adaptive_seed_return_lift_min < 0.0 ||
      st.adaptive_seed_return_lift_max < st.adaptive_seed_return_lift_min) {
    std::fprintf(stderr, "[lazyipdca] invalid ADAPTIVE_MARGINAL_SEED_V2 parameters\n");
    return 2;
  }
  // ATOM_CLUSTER_PROD_V1 parameter validation.
  if (st.atom_cluster_w_direction < 0.0 || st.atom_cluster_w_support < 0.0 ||
      st.atom_cluster_w_curvature < 0.0 ||
      st.atom_cluster_w_direction + st.atom_cluster_w_support + st.atom_cluster_w_curvature <= 0.0 ||
      st.atom_cluster_tau_in < 0.0 || st.atom_cluster_tau_in > 1.0 ||
      st.atom_cluster_tau_out < 0.0 || st.atom_cluster_tau_out > 1.0 ||
      st.atom_cluster_tau_out > st.atom_cluster_tau_in ||
      st.atom_cluster_adapt_min_samples < 1 ||
      st.atom_cluster_adapt_window < st.atom_cluster_adapt_min_samples ||
      st.atom_cluster_adapt_strength < 0.0 || st.atom_cluster_adapt_strength > 1.0 ||
      st.atom_cluster_max_reps < 1 || st.atom_cluster_replace_rel_quality < 0.0 ||
      st.atom_cluster_replace_min_similarity < 0.0 || st.atom_cluster_replace_min_similarity > 1.0 ||
      st.atom_cluster_duplicate_penalty < 0.0 || st.atom_cluster_novelty_bonus < 0.0 ||
      st.atom_cluster_stop_window < 1 || st.atom_cluster_stop_min_observations < 1 ||
      st.atom_cluster_stop_min_observations > st.atom_cluster_stop_window ||
      st.atom_cluster_stop_min_atoms < 1 ||
      st.atom_cluster_stop_max_novelty_rate < 0.0 || st.atom_cluster_stop_max_novelty_rate > 1.0 ||
      st.atom_cluster_stop_max_coverage_gain < 0.0 ||
      st.atom_cluster_stop_max_lift_rel < 0.0 ||
      st.atom_cluster_stop_confirmation_rounds < 1) {
    std::fprintf(stderr, "[lazyipdca] invalid ATOM_CLUSTER_PROD_V1 parameters\n");
    return 2;
  }
  if (st.max_kkt_factorizations < 0 ||
      st.auto_coverage_trigger <= 0.0 ||
      st.auto_coverage_trigger > 1.0 ||
      st.rho0 < 0.0 || st.rho_init_rel_margin < 0.0 || st.rho_init_abs_margin < 0.0 ||
      st.rho_model_min < 0.0 || st.rho_model_min > st.rho_abs_max ||
      st.atom_oracle_pool_size < 1 || st.atom_oracle_activity_mode < 0 || st.atom_oracle_activity_mode > 3 ||
      st.atom_energy_target < 0.0 || st.atom_energy_target > 1.0 || st.atom_min_nnz < 1 ||
      st.atom_oracle_min_directional_ratio < 0.0 || st.atom_oracle_rhox_ratio_trigger < 0.0 ||
      st.atom_sparse_recertify_tol_rel <= 0.0 || st.atom_sparse_recertify_maxit < 1 ||
      st.atom_sparse_beta_max <= 0.0 ||
      st.rho_floor < 0.0 || st.rho_growth < 1.0 || st.rho_rel_margin < 0.0 ||
      st.rho_curvature_tol < 0.0 || st.rho_lanczos_residual_safety < 0.0 ||
      st.rho_dyn_floor_rel_margin < 0.0 || st.rho_dyn_reject_growth <= 1.0 ||
      st.rho_decay <= 0.0 || st.rho_decay > 1.0 || st.rho_trial_every < 1 ||
      st.rho_clean_streak < 1 || st.rho_abs_max <= 0.0 || st.lanczos_maxit < 1 ||
      st.lanczos_tol <= 0.0) {
    std::fprintf(stderr, "[runner] ERREUR : parametres rho hors domaine (rho>=0, 0<=rho_model_min<=rho_abs_max, growth>=1, decay in (0,1], cadences>=1, rho_abs_max>0).\n");
    return 2;
  }
  if ((st.regularization_mode == DcRegularizationMode::Const ||
       st.regularization_mode == DcRegularizationMode::Dyn) &&
      st.rho_init_method == RhoInitMethod::Hint && !use_manifest_lambda) {
    std::fprintf(stderr, "[runner] ATTENTION : --rho-init hint sans --use-manifest-lambda ; Auto/Lanczos sera utilise si aucun hint n'est fourni au solveur.\n");
  }
  if ((st.regularization_mode == DcRegularizationMode::Const ||
       st.regularization_mode == DcRegularizationMode::Dyn) && st.atoms_enable && st.verbose >= 1) {
    std::fprintf(stderr, "[runner] INFO : mode %s impose H_k=0 ; --atoms est ignore pour ce run controle.\n",
                 regularization_mode_name(st.regularization_mode));
  }
  if (st.dca_accept_window < 1) {
    std::fprintf(stderr, "[runner] ERREUR : --accept-window doit etre >= 1.\n"); return 2;
  }
  if (st.dca_momentum_theta_min < 0.0 || st.dca_momentum_theta_max < st.dca_momentum_theta_min ||
      st.dca_momentum_theta < st.dca_momentum_theta_min ||
      st.dca_momentum_theta > st.dca_momentum_theta_max ||
      st.dca_momentum_backtrack <= 0.0 || st.dca_momentum_backtrack > 1.0 ||
      st.dca_momentum_growth < 1.0 || st.dca_momentum_start < 0 ||
      st.dca_momentum_gamma < 0.0 || st.dca_momentum_gamma_min < 0.0 ||
      st.dca_momentum_gamma_max < st.dca_momentum_gamma_min || st.dca_momentum_rho_factor < 0.0 ||
      st.dca_momentum_rho_factor > 1.0 || st.dca_momentum_atom_align_floor < 0.0 ||
      st.dca_momentum_atom_align_floor > 1.0 || st.dca_momentum_atom_align_power < 0.0 ||
      st.dca_lyapunov_eta < 0.0) {
    std::fprintf(stderr, "[runner] ERREUR : parametres inertiels/acceptation hors domaine.\n"); return 2;
  }
  if (st.dca_acceptance == DcaAcceptanceMode::Lyapunov) {
    const double tb = st.dca_lyapunov_theta_bar > 0.0 ? st.dca_lyapunov_theta_bar : st.dca_momentum_theta;
    if (tb >= 0.5)
      std::fprintf(stderr, "[runner] ATTENTION : lyapunov_theta_bar=%.3g >= 1/2 ; la marge theorique (1-2*theta_bar)/2 n'est plus positive.\n", tb);
  }
  if (st.atoms_seed_coverage <= 0.0 &&
      (st.atoms_seed_stall_tol != 0.01 || st.atoms_seed_stall_patience != 3))
    std::fprintf(stderr, "[runner] ATTENTION : --seed-stall-tol/--seed-patience fournis sans "
                         "--seed-coverage : le seeding par couverture est INACTIF "
                         "(c'est --seed-coverage X qui l'active).\n");
  if (manifest.empty()) { std::fprintf(stderr, "usage: lazyipdca_cutest --manifest FILE [--outdir DIR]\n"); return 2; }
  char cwd[4096]; if (!getcwd(cwd, sizeof(cwd))) return 2;
  std::string mabs = (manifest[0] == '/') ? manifest : std::string(cwd) + "/" + manifest;
  std::string oabs = (outdir[0] == '/') ? outdir : std::string(cwd) + "/" + outdir;
  if (decode_dir[0] != '/') decode_dir = std::string(cwd) + "/" + decode_dir;
  auto entries = read_manifest(mabs);
  int fails = 0, degenerate = 0, budgeted = 0, run = 0;
  for (const auto& e : entries) {
    if (!only.empty() && e.name != only && e.tag != only) continue;
    if (only_accepted && e.status != "accepted") continue;
    if (::chdir(cwd) != 0) return 2;
    run++;
    int rc = run_one(e, decode_dir, oabs, st, use_manifest_lambda);
    if (rc == 1) fails++;
    else if (rc == 2) degenerate++;
    else if (rc == 3) budgeted++;
  }
  // v0.16.2 : DualDegenerate compte a part -- primal converge, meilleur
  // point faisable rendu, ce n'est PAS un echec numerique (cf. exit_msg).
  std::printf("[runner] finished: %d problems, %d not solved, %d dual-degenerate, "
              "%d factorization-budget stops\n",
              run, fails, degenerate, budgeted);
  return fails > 0 ? 1 : 0;
}
