// lazyipdca — problem.hpp
// General sparse QP:
//   min 1/2 x'Qx + q'x + c0
//   s.t. A x = b                      (p equalities)
//        cl <= C x <= cu              (m two-sided inequalities, +-inf allowed)
//        l  <=  x  <= u               (box, +-inf allowed)
#pragma once
#include "sparse.hpp"
#include "inertia.hpp"
#include "regularization.hpp"
#include <string>

namespace lazyipdca {
// INERTIA_COMPLETE_AMRS_V24
// INERTIA_COMPLETE_AMRS_V24 — second-generation inertial A/B controls.
enum class DcaMomentumAtomWeighting : int { Uniform=0, Cosine=1, Energy=2 };
enum class DcaMomentumRhoPolicy : int { Restart=0, Keep=1, Transport=2 };


struct Problem {
  idx_t n = 0, p = 0, m = 0;
  SparseSymUpper Q;      // upper triangle of Q (n x n), may be indefinite
  std::vector<double> q; // n
  double c0 = 0.0;
  SparseCSC A;           // p x n (equalities)
  std::vector<double> b; // p
  SparseCSC C;           // m x n (inequalities)
  std::vector<double> cl, cu; // m
  std::vector<double> l, u;   // n
  std::string name = "qp";
};

enum class Status : int {
  Solved = 0,             // first-order KKT certified to tolerance
  SolvedSOC = 1,          // KKT + measured reduced curvature >= -tol (2nd order)
  MaxIter = 2,
  TimeLimit = 3,
  PrimalInfeasibleLike = 4,
  NumericalError = 5,
  Unbounded = 6,
  DCAConverged = 7,       // sequence des objectifs ACCEPTES stationnaire (theorie DCA)
  DualDegenerate = 8,     // primal converge (faisable, objectif stable) mais
                          // multiplicateurs non bornes (LICQ probablement en
                          // defaut) -- resultat LEGITIME, pas un echec numerique
  FactorizationBudget = 9 // voluntary resource stop: best feasible incumbent is returned
};

struct Settings {
  // convergence (all residuals measured in ORIGINAL coordinates)
  double eps_stat = 1e-6;     // stationarity (relative)
  double eps_feas = 1e-6;     // primal feasibility (relative)
  double eps_comp = 1e-6;     // complementarity (relative)
  idx_t  max_iter = 3000;
  // KKT_FACTORIZATION_BUDGET_V4G
  // 0 = unlimited; otherwise no factorization beyond this count.
  idx_t  max_kkt_factorizations = 0;
  double time_limit = 1e10;   // seconds

  // barrier / stepsizes (paper defaults)
  double mu0 = -1.0;          // <0 => set from initial gap
  double barrier_r = 0.1;     // mu_{k+1} = r * gap / n_comp
  double barrier_theta = 0.1; // reduce mu when ||F_mu||_REL <= theta*mu (scale-free)
  idx_t  mu_inner_max = 0;    // optional safety: force mu decrease after N inner
                              // iters (0 = off; the relative gate suffices)
                              // (LOQO/paper spirit: never over-solve the barrier)
  double ftb_delta = 0.995;   // fraction-to-boundary (paper: delta)
  double armijo_rho = 1e-4;   // Armijo constant in (0, 1/2)
  double armijo_beta = 0.5;   // backtracking factor
  idx_t  armijo_max = 40;

  // exact l1 penalty (paper §3.2.1)
  double sigma0 = 10.0;
  double sigma_bar = 1.0;     // safety margin over ||z~||_inf
  double sigma_tau = 2.0;     // growth factor (>1)
  double sigma_max = 1e10;    // plafond -- au-dela, ||z~||_inf est degenere
                              // (LICQ deficient) ; suivre plus loin ne fait
                              // que precipiter le blow-up total (>1e14)

  // DC regularization policy.  Legacy preserves v0.17.1 semantics.
  // CONST/DYN/lazy are the controlled variants used in the OMS article:
  //   CONST: H_k=0 and rho is frozen after a spectral initialization;
  //   DYN:   H_k=0 and rho may adapt;
  //   lazy:  adaptive rho plus low-rank H_k.
  DcRegularizationMode regularization_mode = DcRegularizationMode::Legacy;

  // Scalar-shift initialization.  In CONST/DYN, Auto implements the article's
  // rho_0=max(0,-lambda_hat)*(1+rho_init_rel_margin)+rho_init_abs_margin,
  // with lambda_hat from an external hint when available, otherwise full-space
  // Lanczos and a Gershgorin fallback.  User takes rho0 verbatim.
  RhoInitMethod rho_init_method = RhoInitMethod::Auto;
  double rho0 = 0.0;          // explicit value when rho_init_method=User; legacy seed otherwise
  double rho_init_rel_margin = 1e-6;
  double rho_init_abs_margin = 1e-8;
  // Numerical curvature gate used when DYN commits a smaller model shift.
  // The Lanczos Ritz estimate is guarded by
  //   lambda_guard = lambda_hat - rho_lanczos_residual_safety * residual
  // and the candidate is accepted only if lambda_guard + rho >= -tol.
  // This is deliberately separate from KKT inertia: barrier stabilization
  // may justify rho_x but does not by itself justify changing the DC model.
  double rho_curvature_tol = 1e-8;
  double rho_lanczos_residual_safety = 1.0;
  // DYN keeps a certified full-space floor and adapts only the excess
  // proximal regularization above that floor.  This makes rho genuinely
  // dynamic without ever violating Q + rho I >= 0.
  double rho_dyn_floor_rel_margin = 1e-6;
  double rho_dyn_reject_growth = 2.0;
  // Experimental A/B control for the LAZY model shift only.  When > 0,
  // the isotropic DC-model regularization is never allowed below this value:
  //     rho_k >= rho_model_min  (LAZY only).
  // This is deliberately distinct from rho_floor, which is the first
  // nonzero escalation level for numerical recovery/rho_x.  Default 0 keeps
  // the v0.18.1 numerical semantics unchanged.
  double rho_model_min = 0.0;
  // Experimental causal A/B: in controlled LAZY mode, lock the DC-model
  // scalar shift exactly at rho_model_min for the whole solve.  Numerical
  // factorization stabilization remains separate in rho_x.  Default false.
  bool rho_model_lock = false;
  // v19av-convdesc (ported from piqp-lazydca-19av-convdesc): when the DCA
  // model Q + rho*I is certified strictly convex on ker(A)
  // (rho + safety*lambda_red > 0, lambda_red measured by projected Lanczos and
  // cached while rho is unchanged), the Armijo sufficient-descent margin is
  // dropped to 0: ANY genuine merit descent is accepted -- the certified
  // convex subproblem's minimiser is the canonical DCA step and descends by
  // construction.  Prevents small-but-real steps from being rejected and
  // piling into forced rho escalation (NCVXQP1 regime).  OFF by default.
  bool   accept_convex_descent = false;
  double convex_descent_safety = 1.0;  // >=1; larger = stricter certificate
  double rho_floor = 1e-8;    // first nonzero escalation value
  double rho_growth = 10.0;   // geometric escalation
  double rho_rel_margin = 0.10; // cap = |lambda_min_hat| * (1 + margin)
  double rho_abs_max = 1e10;  // hard safety cap
  double rho_decay = 0.5;     // lazy decay after clean streak
  // ---- atomic low-rank convexification (port of the reference LazyDCA) ----
  // On a curvature deficit, before growing rho further, add a sparsified
  // negative-curvature Ritz atom u (H_k += beta u u', beta = |Rayleigh|*(1+m)).
  // Atoms convexify only the offending modes so rho_eff stays small.
  // Global atom switch. In controlled LAZY mode this is also honored for
  // A/B tests: atoms_enable=false gives H_k=0 while retaining the LAZY rho
  // adaptation. The solver then performs a one-time spectral scalar shift so
  // the first DCA subproblem remains convex. Default true preserves v0.18.1.
  bool   atoms_enable = true;
  // Exact port of the zip's progressive lazy convexification (v0.9.08.19e):
  //  - pre-loop seeding adds `initial_atoms` atoms;
  //  - after EACH accepted step, add `atoms_per_update` atoms, repeated every
  //    iteration, until the curvature probe finds no negative direction on
  //    Q+H_k (last_lambda_est >= -neg_tol) or `max_atoms` is reached; NaN
  //    lambda (no probe yet) counts as "still to convexify";
  //  - additionally every `atom_update_every_accepts` accepts (stagnation);
  //  - rho_eff is set FROM the measured curvature, |min(0,lambda)|*(1+margin),
  //    and is MONOTONE after atom insertions (19ak: never re-raised by probes;
  //    the inertia certification chain remains the safety net).
  idx_t  initial_atoms = 2;
  idx_t  atoms_per_update = 1;
  // Fidelite zip legacy (atom_selection_enable=false, defaut) : projection de
  // face (coords de x_ref sur borne active mises a zero), top-nnz, rejet
  // support vide / norme minimale, beta = |lambda_pre| + margin ADDITIF.
  // La selection (Rayleigh post-sparsification + filtre d'orthogonalite) est
  // OPTIONNELLE comme dans le zip.
  bool   atom_selection_enable = false;
  // 0 = Rayleigh post-sparsification (zip crit. 0) ; 1 = delta-lambda (zip
  // crit. 1) : insertion temporaire + re-mesure de lambda_min(Q+H+cand),
  // garde le candidat qui remonte le plus la courbure = MINIMISE rho_eff.
  idx_t  atom_selection_criterion = 0;
  // Seeding par COUVERTURE : > 0 => semer jusqu'a couvrir cette fraction de
  // la courbure negative (1 - lam_k/lam_0), au lieu du k fixe initial_atoms.
  double atoms_seed_coverage = -1.0;
  // Separate budget for INITIAL coverage seeding only.
  // <= 0 preserves the historical behaviour: seed may use max_atoms.
  // A positive value limits only the pre-loop seed; later atom growth
  // remains governed by max_atoms.
  idx_t  atoms_seed_max_atoms = 0;
  // A/B control: in LAZY coverage-seeding mode, once the requested initial
  // coverage has actually been reached, freeze the seeded low-rank metric H_k.
  // No later atom insertion or event-driven beta retuning is allowed.  If the
  // seeding exits below target (stall/max-atoms/Lanczos failure), growth remains
  // enabled.  Default false preserves v0.18.1 semantics exactly.
  bool   atoms_freeze_after_seed = false;
  double atoms_seed_stall_tol = 0.01;      // gain minimal par round (fraction de |lam0|)
  idx_t  atoms_seed_stall_patience = 3;    // rounds stagnants avant arret
  // Convexification EVENEMENTIELLE : l'evolution de l'optimisation guide
  // l'ajout d'atomes — au plus un atome evenementiel par fenetre.
  //  A : strie de corrections d'inertie (courbure non couverte sur la face)
  //  B : essai rho/2 refuse (rho bloque par de la courbure non couverte)
  //  C : strie Armijo + d'(Q+H)d < 0 (le modele ment le long de la direction
  //      courante — la direction coupable EST le candidat, zero Lanczos)
  // ---- Test d'arret IPOPT (Wachter-Biegler, eq. 5) ----
  // E = max( stat/s_d , viol , comp/s_c ) <= ipopt_tol, avec
  //   s_d = max(s_max, (|y|_1+|z|_1)/(m+n))/s_max,  s_c = max(s_max, |z|_1/n)/s_max
  // (tolere les residus duals proportionnels a la taille des multiplicateurs),
  // + tolerances absolues compagnes + niveau "acceptable" apres N iterations.
  bool   ipopt_stop = false;
  double ipopt_tol = 1e-8;
  double ipopt_s_max = 100.0;
  double ipopt_dual_inf_tol = 1.0;
  double ipopt_constr_viol_tol = 1e-4;
  double ipopt_compl_inf_tol = 1e-4;
  double ipopt_acceptable_tol = 1e-6;
  idx_t  ipopt_acceptable_iter = 15;
  idx_t  ipopt_smooth_window = 5;
  idx_t  atoms_seed_regress_patience = 3;  // reculs avant d'arreter d'escalader (n'arrete plus le seeding)   // fenetre du max glissant (robustesse aux pics)

  // AUTO_FULLCOVERAGE_V4HE0
  // OFF by default: exact backward compatibility.
  bool   auto_atoms_enable = false;
  double auto_coverage_trigger = 0.99;
  // AUTO_PORTFOLIO_V4HE1
  // false reproduces validated E0.1 exactly.
  bool  auto_portfolio_enable = false;

  // Per-child budgets. 0 is not used here: E1 scouts are intentionally bounded.
  idx_t auto_atomic_kkt_budget = 40;
  idx_t auto_scalar_kkt_budget = 40;
  idx_t auto_atomic_max_atoms = 1;

  // Generic quality-first comparison thresholds.
  double auto_quality_stat_tol = 1e-4;
  double auto_quality_comp_tol = 1e-4;

  // AUTO v4h-E1.1: intrinsic complementarity-QP recognition.
  bool   auto_lcp_structure_enable = true;
  double auto_lcp_structure_tol = 1e-10;
  double auto_lcp_admissible_tol = 1e-6;
  double auto_lcp_gap_target = 1e-6;

  // Native child settings used only after exact algebraic LCP recognition.
  double auto_lcp_eps_comp = 1e-9;
  idx_t  auto_lcp_init_strategy = 0;
  // AUTO_LCP_SEMISMOOTH_V4HE14
  bool   auto_lcp_polish_enable = true;
  idx_t  auto_lcp_polish_max_iters = 5;
  idx_t  auto_lcp_polish_max_backtracks = 12;
  double auto_lcp_polish_natural_tol = 1e-13;
  double auto_lcp_polish_armijo = 1e-4;
  double auto_lcp_polish_kkt_tol = 1e-8;
bool   atom_event_enable = false;  // opt-in (--atom-events) : record replique mais regression smoke-2 a instruire
  idx_t  atom_event_every = 20;
  double atom_bound_active_tol = 1e-6;   // defaut zip
  double atom_min_norm = 1e-12;
  double atom_margin = 1e-6;          // zip lazydca_margin (beta additif)
  // Espacement des essais certifies rho/2 (en iterations) : chaque essai
  // coute UNE factorisation ; a 5, ~18% du wall etait gaspille (mesure
  // NCVXQP3_n10000 : 1657 factos pour 1409 iterations).
  idx_t  rho_trial_every = 5;
  // Mehrotra predicteur-correcteur : le pas affine (meme factorisation)
  // calibre sigma=(gap_aff/gap)^3, mu = sigma*gap/n remplace le gate pour la
  // BAISSE (recentrages/planchers conserves), le correcteur du 2nd ordre
  // reutilise la factorisation (~1 backsolve = ~1% du cout d'une facto).
  bool   mehrotra_enable = false;  // plein : superbe en convexe (29->8 its),
                                   // nuit a la traversee DCA (A/B replique) ;
                                   // en hybride il sert de DEBLOQUEUR de mu
  idx_t  mu_stall_iters = 50;      // gel de mu > K iterations => pas sigma force
  double mehrotra_sigma_min = 1e-4;
  double mehrotra_sigma_max = 0.9;
  bool   mehrotra_anneal = true;   // sigma >= 0.7 tant que courbure non certifiee
  // Evasion du second ordre : au point du premier ordre a courbure reduite
  // negative, pas le long de la direction de Ritz projetee (f decroit
  // quadratiquement), re-ancrage, et la course repart. 0 = desactive.
  idx_t  soc_escape_max = 3;
  bool   rho_dynamic = false;      // schedule terminal agressif du rho MODELE
                                   // (historical flag; distinct from regularization_mode=Dyn)
  idx_t  atom_update_every_accepts = 20;
  // Cadence d'ajout d'atomes en ITERATIONS : au plus un ajout toutes les
  // atom_update_every_iters iterations (l'ajout reste declenche a une
  // ACCEPTATION — le modele ne change qu'aux accepts).
  idx_t  atom_update_every_iters = 10;
  double atom_neg_tol = 1e-8;        // zip: lazydca_neg_tol
  bool   rho_monotone_after_atoms = true;  // zip: 19ak
  // ---- zip acceptance: sufficient ORIGINAL-objective descent + feasibility --
  // (production defaults of the reference runner).  The DCA reference x_ref
  // moves ONLY on acceptance events (integrated mode); between accepts the
  // convex model is anchored at x_ref (nonzero proximal gradient in the RHS).
  //   required = max(sd_abs, sd_rel*max(1,|ref|)) + sd_sigma*||x-x_ref||^2
  //   feasible = cv_abs <= sd_cv_abs  OR  (sd_cv_rel>0 AND cv_rel <= sd_cv_rel
  //              AND cv_abs <= sd_rel_feas_abs_cap)          [19g cap]
  //   accept   = feasible AND descent >= required   (19g: both together;
  //              first anchor exempt from descent, never from feasibility)
  double sd_abs = 1e-6;
  double sd_rel = 1e-10;
  double sd_sigma = 1e-8;
  double sd_cv_abs = 1e-6;
  double sd_cv_rel = 0.0;
  double sd_rel_feas_abs_cap = 1e-5;
  bool   sd_accept_against_best_feasible = true;
  // Granularite d'acceptation (zip : l'acceptation s'evalue en sortie de
  // resolution INTERNE du modele) : n'accepter que quand la stationnarite
  // relative du modele courant est descendue sous ce seuil — le pas DCA
  // devient un pas proximal PLEIN au lieu d'un micro-pas re-ancre.
  double sd_model_gate = 1.0;   // neutre (le gate contracte le remplace)
  idx_t  dca_inner_min = 1;           // 1 = neutre (2 chez 0.9.4)
  idx_t  dca_inner_max = 50;          // recovery seulement, jamais "resolu"
  double dca_model_reduction = 1e300; // neutre par defaut (0.10 chez 0.9.4)
  double dca_model_abs = 1e-5;        // plancher absolu du gate
  bool   terminal_polish = true;
  double terminal_anchor_stat_trigger = 1e-4;
  // ---- IPOPT-style gradient-based scaling (nlp_scaling_method=gradient-based)
  // s_f = min(1, max_grad/||grad f(x0)||_inf) ; s_i = min(1, max_grad/||row_i||_inf).
  // The solver WORKS in the scaled space; displays, stopping tests, SOC
  // certificate and the DCA acceptance are all in ORIGINAL (unscaled) values.
  bool   scaling_enable = true;
  double scaling_max_grad = 100.0;   // IPOPT: nlp_scaling_max_gradient
  idx_t  max_atoms = 50;            // zip: lazydca_max_atoms
  idx_t  atom_max_nnz = 1000;       // zip: lazydca_atom_max_nnz
  // A/B sparsity study: after Top-k sparsification of a Ritz direction, retune
  // beta so that the sparse rank-one atom reaches (when attainable) the same
  // lambda_min(Q+H) as the UNSPARSIFIED legacy atom built from the very same
  // Ritz direction and legacy beta. This isolates support sparsity from the
  // amount of spectral convexification. Opt-in: default false preserves the
  // legacy atom path.
  bool   atom_sparse_recertify_beta = false;
  double atom_sparse_recertify_tol_rel = 1e-6;
  idx_t  atom_sparse_recertify_maxit = 24;
  double atom_sparse_beta_max = 1e8;

  // ---- Adaptive atom oracle (post-v0.18.1 experimental production branch) --
  // The legacy rule is almost purely spectral: a negative Ritz direction is
  // inserted because it lifts lambda_min.  The oracle below corrects four
  // weaknesses exposed by the LCP ablations:
  //   (i) remove energetically useless dense Ritz tails;
  //  (ii) rank a POOL of candidates by spectral lift AND DCA activity;
  // (iii) price support/coupling and beta, not only lambda_min;
  //  (iv) keep scalar-LAZY (no atom) as an explicit admissible alternative.
  // Backward-compatible default: preserve the v0.18.1 atom path.
  // The adaptive oracle and energy sparsification are explicit opt-in features.
  bool   atom_oracle_enable = false;
  idx_t  atom_oracle_pool_size = 8;
  // activity mode: 0=none, 1=original-objective gradient at the DCA anchor,
  // 2=last accepted DCA displacement, 3=max(gradient,last displacement).
  idx_t  atom_oracle_activity_mode = 3;
  double atom_oracle_activity_weight = 0.50;
  double atom_oracle_density_weight = 0.10;
  double atom_oracle_coupling_weight = 0.10;
  double atom_oracle_beta_weight = 0.05;
  // ADAPTIVE_ORACLE_BETA_SCALE_V4E
  // Strength multiplier for adaptive-oracle atoms only.
  // 1.0 reproduces v4d; 0.0 suppresses the low-rank contribution.
  double atom_oracle_beta_scale = 1.0;
  // ADAPTIVE_ORACLE_DELAYED_START_V4F
  // Earliest main DCA iteration at which adaptive atoms may be committed.
  // 0 reproduces the historical immediate-seeding behaviour.
  idx_t atom_oracle_start_iter = 0;
  double atom_oracle_directional_weight = 0.50;
  double atom_oracle_min_lift_rel = 1e-4;
  double atom_oracle_min_activity = 1e-6;
  double atom_oracle_min_score = 1e-4;
  // ADAPTIVE_ORACLE_HYBRID_V4C
  // Normalize DCA activity to remove the O(n_free^-1/2) size effect.
  bool   atom_oracle_activity_sqrt_nfree = true;
  double atom_oracle_min_scaled_activity = 1e-2;

  // false: historical hard reject if dirRatio < theta.
  // true : theta becomes the minimum residual isotropic fraction.
  bool   atom_oracle_directional_residual = true;
  // An atom must retain at least this fraction of the scalar alternative's
  // curvature along the current DCA-relevant direction.  Otherwise the
  // scalar no-atom model is preferred even if the atom perfectly fixes
  // lambda_min.  0 disables this gate.
  double atom_oracle_min_directional_ratio = 0.10;
  bool   atom_oracle_scalar_fallback = true;

  // Energy-aware support: choose the smallest Top-k retaining target energy,
  // capped by atom_max_nnz.  It replaces a fixed density percentage and makes
  // support scale with the actual Ritz localisation.
  bool   atom_energy_sparsify = false;
  double atom_energy_target = 0.999999;
  idx_t  atom_min_nnz = 1;

  // Post-insertion self-correction.  The existing braking diagnostic can be
  // used even when generic atom events are off: a repeatedly over-penalising
  // atom gets beta/2 only if the KKT factorisation certifies the change.
  bool   atom_oracle_retune_enable = false;
  double atom_oracle_brake_fraction = 0.50;
  idx_t  atom_oracle_brake_patience = 5;
  // Optional emergency rollback for A/B: if numerical KKT stabilization rho_x
  // dwarfs model rho, purge H and let scalar-LAZY take over.  Off by default
  // until the full CUTEst campaign validates a universal trigger.
  bool   atom_oracle_rhox_rollback = false;
  double atom_oracle_rhox_ratio_trigger = 8.0;
  // ATOM_CLUSTER_PROD_V1 — opt-in C5 structural clustering.
  bool   atom_cluster_enable = false;  // false => exact C5 control path
  double atom_cluster_w_direction = 0.85;
  double atom_cluster_w_support = 0.10;
  double atom_cluster_w_curvature = 0.05;
  double atom_cluster_tau_in = 0.965;
  double atom_cluster_tau_out = 0.900;
  bool   atom_cluster_adaptive_thresholds = true;
  idx_t  atom_cluster_adapt_min_samples = 16;
  idx_t  atom_cluster_adapt_window = 128;
  double atom_cluster_adapt_strength = 0.20;
  idx_t  atom_cluster_max_reps = 64;
  double atom_cluster_replace_rel_quality = 1e-3;
  double atom_cluster_replace_min_similarity = 0.90;
  double atom_cluster_duplicate_penalty = 0.20;
  double atom_cluster_novelty_bonus = 0.05;
  idx_t  atom_cluster_stop_window = 12;
  idx_t  atom_cluster_stop_min_observations = 8;
  idx_t  atom_cluster_stop_min_atoms = 4;
  double atom_cluster_stop_max_novelty_rate = 0.10;
  double atom_cluster_stop_max_coverage_gain = 2.5e-3;
  double atom_cluster_stop_max_lift_rel = 1.0e-3;
  idx_t  atom_cluster_stop_confirmation_rounds = 2;
  // ADAPTIVE_MARGINAL_SEED_V2 — opt-in commit-only diminishing-return controller.
  bool   adaptive_seed_return_enable = false;  // false => exact C5 control path
  idx_t  adaptive_seed_return_warmup = 4;
  idx_t  adaptive_seed_return_window = 3;
  idx_t  adaptive_seed_return_confirmations = 2;
  double adaptive_seed_return_alpha_coverage = 0.20;
  double adaptive_seed_return_alpha_lift = 0.20;
  double adaptive_seed_return_coverage_min = 1e-5;
  double adaptive_seed_return_coverage_max = 2.5e-3;
  double adaptive_seed_return_lift_min = 1e-5;
  double adaptive_seed_return_lift_max = 1e-3;
  // ADAPTIVE_MARGINAL_RETURN_V21 — global commit-time AMRS, opt-in.
  bool   adaptive_marginal_return_enable = false; // false => exact v2/C5 control path
  idx_t  adaptive_return_warmup = 4;
  idx_t  adaptive_return_window = 3;
  idx_t  adaptive_return_confirmations = 2;
  double adaptive_return_alpha_lift = 0.20;
  double adaptive_return_alpha_effective = 0.20;
  double adaptive_return_lift_min = 1e-5;
  double adaptive_return_lift_max = 1e-3;
  double adaptive_return_effective_min = 1e-5;
  double adaptive_return_effective_max = 2.5e-3;
  double adaptive_return_cooldown_factor = 2.0;
  double adaptive_return_cooldown_max_factor = 8.0;
  double adaptive_return_strong_factor = 4.0;
  // ADAPTIVE_MARGINAL_RETURN_V22 — seed decision from TRUE marginal coverage.
  // Overall AMRS remains opt-in through adaptive_marginal_return_enable.
  double adaptive_return_seed_alpha_coverage = 0.20;
  double adaptive_return_seed_alpha_lift = 0.20;
  double adaptive_return_seed_coverage_min = 1e-5;
  double adaptive_return_seed_coverage_max = 2.5e-3;
  double adaptive_return_seed_lift_min = 1e-5;
  double adaptive_return_seed_lift_max = 1e-3;
  bool   adaptive_return_seed_absolute_enable = false; // v2.3: legacy v2.2 branch retired
  idx_t  adaptive_return_seed_absolute_window = 5;
  idx_t  adaptive_return_seed_absolute_confirmations = 3;
  double adaptive_return_seed_absolute_coverage = 2.5e-3;
  double adaptive_return_seed_absolute_lift = 1e-3;
  // ADAPTIVE_MARGINAL_RETURN_V23 — robust committed-rank seed plateau.
  // v2.2 instantaneous absolute branch is retired; overall AMRS stays OFF.
  bool   adaptive_return_seed_plateau_enable = true;
  idx_t  adaptive_return_seed_plateau_horizon = 8;
  idx_t  adaptive_return_seed_plateau_min_commits = 12;
  idx_t  adaptive_return_seed_plateau_confirmations = 2;
  double adaptive_return_seed_plateau_coverage_per_commit = 2.5e-3;
  double adaptive_return_seed_plateau_lift = 1e-3;






  double atom_beta_margin = 0.5;    // beta = |rayleigh| * (1 + margin)
  double atom_ortho_tol = 0.9;      // zip: reject |cos| > tol vs existing atoms

  // ---- KKT factorization reuse (inexact Newton in smooth stretches) ----
  // Reuse the current factorization for up to kkt_reuse_max consecutive
  // steps when mu is unchanged, the previous step was clean (no inertia
  // correction) and nearly full (alpha_pr >= kkt_reuse_alpha_min) and the
  // barrier residual decreased.  Line search safeguards the inexactness.
  bool   kkt_reuse = true;
  idx_t  kkt_reuse_max = 4;
  double kkt_reuse_alpha_min = 0.9;
  idx_t  rho_clean_streak = 3;// # consecutive clean-inertia iters before decay
  idx_t  lanczos_maxit = 60;  // projected Lanczos iterations
  double lanczos_tol = 1e-3;  // relative residual tolerance
  // optional external estimate of lambda_min (e.g. manifest lambda_min_est);
  // if finite it seeds the rho cap without a Lanczos call (still enlargeable).
  double lambda_min_hint = std::numeric_limits<double>::quiet_NaN();
  double eq_density_skip = 0.5; // if p/n > this => feasibility-dominated:
                                // skip projected Lanczos, use Gershgorin
  // regularizations
  double delta_dual = 1e-9;   // static dual regularization (quasidefiniteness)
  // bound_push d'IPOPT : repousse x0 (et les slacks) a l'interieur des
  // bornes : p = min(bound_push*max(1,|borne|), bound_frac*(u-l)).
  double bound_push = 1e-2;
  double bound_frac = 1e-2;
  // bound_mult_init_val d'IPOPT : > 0 => multiplicateurs de bornes initialises
  // UNIFORMEMENT a cette valeur (IPOPT : 1.0) au lieu de l'init centree mu/sl.
  // <= 0 (defaut) : init centree historique.
  double bound_mult_init_val = -1.0;
  // ---- adaptive bound push (v0.16.0) --------------------------------------
  // Le bound_push GLOBAL ci-dessus modifie TOUJOURS x0 (kappa=1e-2 partout).
  // La variante adaptative ne pousse que les composantes dont le slack
  // RELATIF est pathologique (< bp_slack_tol), et seulement quand l'evidence
  // le justifie :
  //   Etage 0 (preventif, gratuit)  : scan post-initialisation ; les
  //     composantes collees aux bornes (ancre faisable atterrissant sur une
  //     face, bound_push desactive par l'utilisateur, ...) sont repoussees
  //     de kappa_init — le point de depart est respecte a epsilon pres.
  //   Etage 1 (reactif, gate par SIGNATURE) : sur NumericalError de
  //     factorisation (MUMPS Fail), plafond de regularisation duale, ou gel
  //     bit-identique, on verifie d'abord la signature "slack pathologique"
  //     (min_i slack_rel < bp_slack_tol). Confirmee => push selectif avec
  //     kappa escalade (x bp_kappa_mult par evenement, plafond = defaut
  //     IPOPT), duaux des composantes poussees RE-INITIES coherents
  //     (mu/slack, ou bound_mult_init_val si actif), delta remis au plancher,
  //     retry. Absente => le push ne servirait a rien : comportement 0.15.x
  //     inchange (abandon). Un push n'est JAMAIS declenche par un
  //     CurvatureDeficit — la courbure reste la juridiction de rho/atomes.
  //   Etage 2 (garde-fous) : budget bp_max_events puis abandon avec l'erreur
  //     originale ; hysteresis (kappa ne redescend jamais dans un run).
  // bp_adaptive=false (defaut) => comportement 0.15.19 STRICTEMENT identique.
  bool   bp_adaptive = false;
  double bp_slack_tol = 1e-10;   // seuil de detection (slack relatif)
  double bp_kappa_init = 1e-8;   // push initial (etage 0 et 1er evenement)
  double bp_kappa_mult = 100.0;  // escalade geometrique par evenement
  double bp_kappa_max = 1e-2;    // plafond (= bound_push IPOPT par defaut)
  idx_t  bp_max_events = 3;      // evenements reactifs max (etage 1)
  // ---- Examen cible par variable, une seule passe (v0.17.0) ----------------
  // Recalibrage suite a un retour direct : bp_slack_tol (1e-10) ne detectait
  // que les slacks quasi-nuls, huit ordres de grandeur sous l'echelle
  // observee empiriquement (1e-3 a 5e-2) ou un bound_push MANUEL resout des
  // problemes sans erreur. Remplace le retry (rejete : re-resoudre plusieurs
  // fois est une recherche couteuse) par UN SEUL passage deterministe,
  // parallelisable (chaque variable/slack s'examine independamment) :
  // pour chaque composante bornee, si son slack RELATIF est sous
  // bp_margin_tol, la pousser a bp_push_kappa (formule bound_push/bound_frac
  // standard, appliquee SELECTIVEMENT) ; sinon rien. Un seul passage, pas
  // d'escalade, pas de budget d'evenements -- la decision est prise une fois,
  // avant la boucle principale, sur le point apres ancre/restauration
  // d'interiorite.
  double bp_margin_tol = 1e-2;   // seuil de detection (slack relatif)
  double bp_push_kappa = 5e-2;   // marge cible quand le seuil est franchi
  // "modele resolu" : seuil de stationnarite RELATIVE du sous-probleme
  // convexe au-dela duquel tout point faisable non-ascendant est accepte
  // sans exiger la decroissance suffisante (garantie de majorisation DCA).
  double model_solved_tol = 1e-4;
  // MUMPS relative pivoting threshold CNTL(1).  The inertia-corrected KKT
  // matrix is quasidefinite hence STRONGLY FACTORIZABLE (Vanderbei): near-
  // static pivoting is both stable and fast; large values trigger delayed-
  // pivot fill explosions (workspace -9) on hard instances.
  double pivot_tol = 1e-6;
  // misc
  // Initial point strategy:
  //   0 = feasible anchor (existing default: interior clip + least-squares
  //       projection of x0 onto {Ax=b});
  //   1 = article §4.1 trust-region start: affine map of the box [l,u] onto
  //       [-1,1]^n, global solve of the ball-relaxed QP (radius sqrt(n)) by
  //       the projection-DCA (paper Algorithm 3), back-transform, interior
  //       clip.  Guarantees f(x0) <= alpha (lower bound of the relaxation).
  //       Falls back to 0 if the box is unbounded.
  //   2 = strategy 1 followed by the equality anchor projection of 0.
  int    init_strategy = 0;
  idx_t  tr_dca_maxit = 1000;
  double tr_interior_margin = 0.05; // interior pull-back of the TR start point   // Algorithm 3 iteration cap
  double tr_dca_tol = 1e-6;     // paper stopping: ||v_{k+1}-v_k||/(1+||v_k||)
  int    verbose = 1;
  int    ordering = 5;        // MUMPS ICNTL(7): 5 = METIS (validated default)
  // Backend lineaire : 0 = auto (LDL mince si N <= slim_backend_max_dim,
  // MUMPS au-dela), 1 = MUMPS force, 2 = LDL mince force (Eigen SimplicialLDLT,
  // inertie par les signes de D — certification inchangee).
  int    linear_backend = 0;
  idx_t  slim_backend_max_dim = 5000;   // mesure : a fort remplissage (NCVXQP), MUMPS bat le simplicial des ~10k
  bool   feasible_anchor = true; // equality least-squares projection of x0
  bool   soc_certificate = true; // measure reduced curvature at the solution
  // ---- Test d'arret au niveau DCA : sur la SEQUENCE DES OBJECTIFS DES PAS
  // ACCEPTES (pas les residus IPM). Theorie DCA standard : f(x_k) decroit,
  // borne inferieurement, converge — quand Delta_f devient stationnaire sur
  // plusieurs accepts consecutifs, x_k tend vers un point critique du
  // programme DC, INDEPENDAMMENT du test KKT/dual (utile quand le primal est
  // deja converge mais le dual est degenere — cf. NCVXQP7 : meilleur objectif
  // deja atteint, mais le test KKT echoue sur des multiplicateurs enormes).
  bool   dca_stop_enable = false;  // opt-in (--dca-stop) : l'objectif plateau plus vite que le gradient pres d'un optimum bien conditionne, un defaut ON preempterait des convergences KKT/SOC plus fines
  double dca_stall_tol_abs = 1e-8;
  double dca_stall_tol_rel = 1e-10;
  idx_t  dca_stall_iters = 5;   // accepts consecutifs sous tolerance avant arret
  // Verrou de monotonie DCA (TOUJOURS actif, pas un flag opt-in) : un point
  // ne peut jamais etre accepte s'il degrade le dernier objectif accepte
  // au-dela de cette tolerance (absolue + relative, correctement mise a
  // l'echelle — remplace le f_noise trop etroit pour les grandes echelles).
  double dca_monotone_tol_abs = 1e-8;
  double dca_monotone_tol_rel = 1e-9;
  idx_t  dca_smooth_window = 5;   // fenetre du max glissant pour le test d'arret DCA

  // ---- inertial DCA layer -------------------------------------------------
  // Two independent A/B axes:
  //   momentum   : original | inertiel_euclidien | inertiel_global | inertiel_atomique
  //   acceptance : monotone | window | lyapunov
  // Defaults reproduce the historical solver exactly (original + monotone).
  DcaMomentumMode dca_momentum = DcaMomentumMode::Original;
  DcaAcceptanceMode dca_acceptance = DcaAcceptanceMode::Monotone;

  // theta_k control.  The current value is reduced after a rejected DCA
  // candidate and may recover geometrically after an acceptance.
  double dca_momentum_theta = 0.25;
  double dca_momentum_theta_min = 0.0;
  double dca_momentum_theta_max = 0.45;
  double dca_momentum_backtrack = 0.5;
  double dca_momentum_growth = 1.05;
  idx_t  dca_momentum_start = 2;       // accepted anchors before inertia is enabled

  // Euclidean control p_k = theta_k * gamma_k * v_k.  gamma_k can be
  // fixed or chosen as the Rayleigh scale (v'M_kv)/(v'v), which preserves
  // the local units of the current DC metric while keeping an identity force.
  DcaEuclideanGammaMode dca_momentum_gamma_mode = DcaEuclideanGammaMode::MetricRayleigh;
  double dca_momentum_gamma = 1.0;
  double dca_momentum_gamma_min = 0.0;
  double dca_momentum_gamma_max = 1e12;

  // Atom-wise controller:
  // p_k = theta_k*rho_factor*rho*v + sum theta_j beta_j u_j(u_j'v),
  // theta_j/theta_k = align_floor + (1-align_floor)|cos(u_j,v)|^align_power.
  double dca_momentum_rho_factor = 1.0;
  double dca_momentum_atom_align_floor = 0.0;
  double dca_momentum_atom_align_power = 1.0;
  bool   dca_momentum_reset_on_metric_change = true;
  // INERTIA_COMPLETE_AMRS_V24. OFF preserves the first-generation inertial path.
  bool dca_momentum_complete = false;
  DcaMomentumAtomWeighting dca_momentum_atom_weighting = DcaMomentumAtomWeighting::Cosine;
  DcaMomentumRhoPolicy dca_momentum_rho_policy = DcaMomentumRhoPolicy::Restart;
  bool dca_momentum_selective_restart = true;
  bool dca_momentum_energy_weight_rho = true;
  double dca_momentum_theta_inc = 0.10;
  double dca_momentum_align_threshold = 0.0;
  double dca_momentum_align_decay = 0.50;
  double dca_momentum_force_rel_max = 0.50;
  double dca_momentum_stat_guard_factor = 10.0;
  bool dca_lyapunov_cap_momentum_theta = true;
  bool dca_lyapunov_enforce_euclidean_bound = false;
  bool dca_lyapunov_unsafe_euclidean = false;


  // Non-monotone acceptance.  Window uses the maximum accepted objective in
  // the last W anchors.  Lyapunov uses
  //   E_k = f(x_k) + theta_bar/2 ||x_k-x_{k-1}||_{M_k}^2
  // and requires E_{k+1} <= E_k - eta ||s_k||_{M_k}^2 (up to roundoff).
  idx_t  dca_accept_window = 5;
  double dca_lyapunov_theta_bar = 0.25;  // <=0 => use dca_momentum_theta
  double dca_lyapunov_eta = 0.0;
  idx_t  heartbeat_every = 20;     // signe de vie (verbose 1-2) si aucun accept depuis N its (0=off)
  idx_t  reject_streak_rho_bump = 50;  // remonte rho (x4) apres N rejets consecutifs
                                       // sans accept (soupape anti-blocage, 0=off)
};

struct Info {
  Status status = Status::NumericalError;
  idx_t iters = 0;
  double obj = 0.0;
  // KKT residuals in ORIGINAL coordinates (absolute and relative)
  double res_stat_abs = INF, res_stat_rel = INF;
  double res_eq_abs = INF, res_ineq_abs = INF, res_feas_rel = INF;
  double res_comp_abs = INF, res_comp_rel = INF;
  double mu_final = 0.0;
  // DC regularization report
  double rho_initial = 0.0;       // model shift entering the main loop
  double rho_final = 0.0;         // final model shift (excludes rho_x)
  double rho_stability_final = 0.0; // final numerical-stability shift rho_x
  idx_t n_rho_model_updates = 0, n_rho_model_increases = 0, n_rho_model_decreases = 0;
  idx_t n_rho_curvature_checks = 0, n_rho_curvature_rejects = 0;
  double rho_last_curvature_guard = INF;
  idx_t n_factorizations = 0, n_inertia_corrections = 0, n_lanczos_calls = 0;
  idx_t n_atoms = 0, atoms_nnz = 0, n_kkt_reuses = 0;
  // Initial-seeding diagnostics.  seed_coverage_est is Ritz-based and
  // deliberately not advertised as a certified spectral coverage.
  idx_t n_seed_atoms = 0;
  double seed_coverage_est = 0.0;
  bool seed_atom_cap_hit = false;
  // AUTO_FULLCOVERAGE_V4HE0 telemetry
  bool   auto_triggered = false;
  bool   auto_scalar_committed = false;
  double auto_seed_coverage = 0.0;
  double auto_seed_lambda0 = 0.0;
  double auto_seed_lambda_best = 0.0;
  idx_t  auto_seed_atoms = 0;

  // AUTO_PORTFOLIO_V4HE1 telemetry.
  bool  auto_portfolio_used = false;

  // 0=legacy/no trigger, 1=scalar, 2=atomic.
  idx_t auto_selected_mode = 0;

  idx_t auto_atomic_factorizations = 0;
  idx_t auto_scalar_factorizations = 0;
  idx_t auto_portfolio_factorizations = 0;

  double auto_atomic_objective = 0.0;
  double auto_scalar_objective = 0.0;

  double auto_atomic_stat_rel = 0.0;
  double auto_scalar_stat_rel = 0.0;
  double auto_atomic_feas_rel = 0.0;
  double auto_scalar_feas_rel = 0.0;
  double auto_atomic_comp_rel = 0.0;
  double auto_scalar_comp_rel = 0.0;

  bool   auto_lcp_structure = false;
  double auto_lcp_structure_residual = INF;

  double auto_atomic_lcp_feas = INF;
  double auto_scalar_lcp_feas = INF;
  double auto_atomic_lcp_comp = INF;
  double auto_scalar_lcp_comp = INF;
  double auto_atomic_lcp_gap = INF;
  double auto_scalar_lcp_gap = INF;
  // AUTO_LCP_SEMISMOOTH_V4HE14 telemetry
  bool  auto_lcp_polish_used = false;
  bool  auto_lcp_polish_accepted = false;
  bool  auto_lcp_polish_singular = false;
  bool  auto_lcp_polish_kkt_certified = false;
  idx_t auto_lcp_polish_iterations = 0;
  idx_t auto_lcp_polish_factorizations = 0;
  idx_t auto_lcp_polish_backtracks = 0;
  double auto_lcp_pre_gap = INF;
  double auto_lcp_post_gap = INF;
  double auto_lcp_pre_comp = INF;
  double auto_lcp_post_comp = INF;
  double auto_lcp_pre_natural = INF;
  double auto_lcp_post_natural = INF;
  double auto_lcp_pre_feas = INF;
  double auto_lcp_post_feas = INF;
  double auto_lcp_post_kkt_stat = INF;
idx_t n_atom_candidates = 0, n_atom_oracle_rejects = 0, n_atom_scalar_fallbacks = 0;
  idx_t n_atom_retunes = 0, n_atom_rollbacks = 0;
  double atom_last_score = 0.0, atom_last_activity = 0.0, atom_last_lift_rel = 0.0;
  double atom_last_energy = 0.0, atom_last_support_frac = 0.0;
  // ATOM_CLUSTER_PROD_V1 telemetry.
  idx_t atom_cluster_count_final = 0;
  idx_t n_atom_cluster_classify = 0;
  idx_t n_atom_cluster_novel = 0;
  idx_t n_atom_cluster_existing = 0;
  idx_t n_atom_cluster_ambiguous = 0;
  idx_t n_atom_cluster_rep_replacements = 0;
  idx_t n_atom_cluster_saturated = 0;
  idx_t n_atom_cluster_seed_stops = 0;
  double atom_cluster_tau_in_final = 0.0;
  double atom_cluster_tau_out_final = 0.0;
  double atom_cluster_last_similarity = 0.0;
  double atom_cluster_seed_novelty_rate = 0.0;
  double atom_cluster_seed_coverage_gain = 0.0;
  double atom_cluster_seed_lift_rel = 0.0;
  // ADAPTIVE_MARGINAL_SEED_V2 telemetry.
  idx_t n_seed_return_observations = 0;
  idx_t n_seed_return_low_rounds = 0;
  idx_t n_seed_return_stops = 0;
  idx_t n_seed_return_multi_atom_jumps = 0;
  idx_t seed_return_stop_atom = 0;
  idx_t seed_return_seed_atoms_final = 0;
  bool seed_return_stopped = false;
  double seed_return_cov_ref = 0.0;
  double seed_return_lift_ref = 0.0;
  double seed_return_cov_threshold = 0.0;
  double seed_return_lift_threshold = 0.0;
  double seed_return_last_cov_gain = 0.0;
  double seed_return_last_lift_rel = 0.0;
  double seed_return_recent_cov_median = 0.0;
  double seed_return_recent_lift_median = 0.0;
  double seed_return_stop_coverage = 0.0;
  double seed_return_stop_cov_gain = 0.0;
  double seed_return_stop_lift_rel = 0.0;
  // ADAPTIVE_MARGINAL_RETURN_V21 telemetry.
  idx_t n_adaptive_return_commits = 0;
  idx_t n_adaptive_return_scored = 0;
  idx_t n_adaptive_return_seed_commits = 0;
  idx_t n_adaptive_return_event_oracle_commits = 0;
  idx_t n_adaptive_return_event_direct_commits = 0;
  idx_t n_adaptive_return_update_commits = 0;
  idx_t n_adaptive_return_low_rounds = 0;
  idx_t n_adaptive_return_seed_stop_requests = 0;
  idx_t n_adaptive_return_seed_stops = 0;
  idx_t n_adaptive_return_cooldowns = 0;
  idx_t n_adaptive_return_reactivations = 0;
  idx_t n_adaptive_return_model_resets = 0;
  idx_t adaptive_return_atoms_after_seed = 0;
  idx_t adaptive_return_seed_stop_atom = 0;
  idx_t adaptive_return_event_spacing_current = 0;
  idx_t adaptive_return_last_source = 0;
  bool adaptive_return_seed_stopped = false;
  double adaptive_return_lift_ref = 0.0;
  double adaptive_return_effective_ref = 0.0;
  double adaptive_return_lift_threshold = 0.0;
  double adaptive_return_effective_threshold = 0.0;
  double adaptive_return_last_lift_rel = 0.0;
  double adaptive_return_last_effective_rel = 0.0;
  double adaptive_return_recent_lift_median = 0.0;
  double adaptive_return_recent_effective_median = 0.0;
  double adaptive_return_cooldown_multiplier = 1.0;
  // ADAPTIVE_MARGINAL_RETURN_V22 seed telemetry.
  idx_t n_adaptive_return_seed_scored_rounds = 0;
  idx_t n_adaptive_return_seed_unreliable_rounds = 0;
  idx_t n_adaptive_return_seed_low_relative_rounds = 0;
  idx_t n_adaptive_return_seed_low_absolute_rounds = 0;
  idx_t adaptive_return_seed_relative_confirmations = 0;
  idx_t adaptive_return_seed_absolute_confirmations = 0;
  idx_t adaptive_return_seed_stop_reason = 0; // 0 none, 1 relative, 2 absolute
  bool adaptive_return_seed_references_ready = false;
  double adaptive_return_seed_last_coverage = 0.0;
  double adaptive_return_seed_last_coverage_gain = 0.0;
  double adaptive_return_seed_last_lift_rel = 0.0;
  double adaptive_return_seed_coverage_ref = 0.0;
  double adaptive_return_seed_lift_ref = 0.0;
  double adaptive_return_seed_coverage_threshold = 0.0;
  double adaptive_return_seed_lift_threshold = 0.0;
  double adaptive_return_seed_recent_coverage_median = 0.0;
  double adaptive_return_seed_recent_lift_median = 0.0;
  double adaptive_return_seed_absolute_recent_coverage_median = 0.0;
  double adaptive_return_seed_absolute_recent_lift_median = 0.0;
  double adaptive_return_seed_stop_coverage = 0.0;
  double adaptive_return_seed_stop_coverage_gain = 0.0;
  double adaptive_return_seed_stop_lift_rel = 0.0;
  // ADAPTIVE_MARGINAL_RETURN_V23 seed plateau telemetry.
  idx_t n_adaptive_return_seed_progress_rounds = 0;
  idx_t n_adaptive_return_seed_plateau_evaluations = 0;
  idx_t n_adaptive_return_seed_low_plateau_rounds = 0;
  idx_t adaptive_return_seed_plateau_confirmations = 0;
  bool adaptive_return_seed_plateau_ready = false;
  double adaptive_return_seed_last_reliable_coverage = 0.0;
  double adaptive_return_seed_plateau_horizon_gain = 0.0;
  double adaptive_return_seed_plateau_gain_per_commit = 0.0;
  double adaptive_return_seed_plateau_lift_median = 0.0;
  double adaptive_return_seed_stop_plateau_gain_per_commit = 0.0;
  double adaptive_return_seed_stop_plateau_lift_median = 0.0;





  // adaptive bound push (v0.16.0) : evenements reactifs (etage 1) et
  // composantes poussees cumulees (etages 0+1)
  idx_t n_bp_events = 0, n_bp_pushed = 0;
  idx_t n_accepts = 0, n_rejects = 0;
  // inertial/acceptance diagnostics
  idx_t n_momentum_steps = 0, n_momentum_resets = 0, n_momentum_backtracks = 0;
  double momentum_theta_final = 0.0, momentum_gamma_final = 0.0, momentum_force_inf_final = 0.0;
  double lyapunov_energy_final = INF;
  // INERTIA_COMPLETE_AMRS_V24 diagnostics.
  idx_t n_momentum_complete_builds=0, n_momentum_complete_accepts=0;
  idx_t n_momentum_restarts_rho=0, n_momentum_restarts_atom=0, n_momentum_restarts_beta=0;
  idx_t n_momentum_rho_transports=0, n_momentum_rho_keeps=0, n_momentum_force_capped=0;
  idx_t n_momentum_stat_guard_shrinks=0, n_euclidean_gamma_capped=0;
  idx_t n_theta_capped_by_lyapunov=0, n_theta_align_decays=0;
  double momentum_theta_rho_final=0.0, momentum_theta_atom_min=0.0;
  double momentum_theta_atom_mean=0.0, momentum_theta_atom_max=0.0;
  double momentum_align_mean=0.0, momentum_align_max=0.0;
  double momentum_energy_rho_share=0.0, momentum_energy_atom_max_share=0.0;
  double momentum_energy_atom_effective_count=0.0, momentum_force_rel=0.0;
  double stat_model_base_rel=0.0, stat_model_inertial_rel=0.0;

  double ref_obj = INF, best_feasible_obj = INF;
  double obj_scale = 1.0;
  double lambda_min_reduced = INF; // measured reduced curvature (if requested)
  bool soc_certified = false;
  double solve_time = 0.0;
  std::string message;
};

struct Result {
  std::vector<double> x, y, z;      // primal, eq duals, ineq duals
  std::vector<double> zl, zu;       // bound duals (>=0)
  Info info;
};

}  // namespace lazyipdca
