// bp_test.cpp — validation du adaptive bound push (v0.16.0).
// Test 1 : DORMANCE — bp_adaptive=ON sur les problemes sains du smoke :
//          resultats identiques, zero evenement BP (le mecanisme ne perturbe
//          jamais un run en bonne sante).
// Test 2 : ETAGE 0 — ancre faisable atterrissant sur une face d'une boite a
//          grande echelle (|l|=1e8) avec bound_push=0 (point de depart
//          respecte) : slack relatif ~1e-12 => signature => push selectif
//          post-anchor, et le solveur converge.
// Test 3 : IMMUNITE STRUCTURELLE — variables quasi-fixes (l=u et boites
//          epsilon, regime QPNSTAIR) : slack relatif nul STRUCTUREL, mais
//          non poussable => aucune signature, aucun evenement, budget intact.
#include "lazyipdca/solver.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace lazyipdca;

static SparseSymUpper tridiag_indef(idx_t n, double dscale, unsigned seed = 7) {
  std::vector<idx_t> I, J; std::vector<double> V;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> ud(-1.0, 1.0);
  for (idx_t i = 0; i < n; ++i) {
    I.push_back(i); J.push_back(i); V.push_back(dscale * ud(gen));
    if (i + 1 < n) { I.push_back(i); J.push_back(i + 1); V.push_back(0.5 * ud(gen)); }
  }
  SparseSymUpper Q; Q.U = csc_from_coo(n, n, I, J, V);
  return Q;
}

static int fails = 0;
static void expect(bool ok, const char* what) {
  std::printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

int main() {
  // ---- Test 1 : dormance sur probleme sain (box-only indefini) ------------
  {
    idx_t n = 1000;
    Problem P; P.n = n; P.name = "bp_dormant";
    P.Q = tridiag_indef(n, 2.0);
    P.q.assign(n, 0.0);
    std::mt19937 gen(13); std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (auto& v : P.q) v = ud(gen);
    P.l.assign(n, -3.0); P.u.assign(n, 3.0);
    Settings s_off; s_off.verbose = 0;
    Settings s_on = s_off; s_on.bp_adaptive = true;
    Result r0 = solve(P, s_off), r1 = solve(P, s_on);
    std::printf("[T1] dormance : OFF obj=%+.8e it=%lld | ON obj=%+.8e it=%lld bp_ev=%lld bp_push=%lld\n",
                r0.info.obj, (long long)r0.info.iters, r1.info.obj,
                (long long)r1.info.iters, (long long)r1.info.n_bp_events,
                (long long)r1.info.n_bp_pushed);
    expect(r1.info.n_bp_events == 0 && r1.info.n_bp_pushed == 0,
           "T1 : zero evenement BP sur run sain");
    expect(r0.info.iters == r1.info.iters && r0.info.obj == r1.info.obj,
           "T1 : trajectoire bit-identique ON vs OFF (sain)");
  }

  // ---- Test 2 : etage 0 sur ancre atterrissant sur une face ----------------
  // x >= 1e8, equalite Sum x = n*1e8 : l'ancre min-norm du x0 interne
  // (l+1) tombe EXACTEMENT sur la face x=l ; la restauration d'ancrage laisse
  // un slack absolu ~1e-4 soit un slack RELATIF ~1e-12 (< bp_slack_tol).
  // bound_push=0 pour respecter le point de depart (scenario utilisateur).
  {
    idx_t n = 40;
    const double L = 1e8;
    Problem P; P.n = n; P.p = 1; P.name = "bp_face_anchor";
    P.Q = tridiag_indef(n, 1.0, 21);
    P.q.assign(n, 0.0);
    std::mt19937 gen(17); std::uniform_real_distribution<double> ud(0.5, 1.5);
    for (auto& v : P.q) v = ud(gen);
    std::vector<idx_t> I(n, 0), J(n); std::vector<double> V(n, 1.0);
    for (idx_t j = 0; j < n; ++j) J[j] = j;
    P.A = csc_from_coo(1, n, I, J, V);
    P.b.assign(1, (double)n * L);          // ancre => x_i = L exactement
    P.l.assign(n, L);
    P.u.assign(n, INF);
    Settings s_off; s_off.verbose = 0; s_off.bound_push = 0.0; s_off.bound_frac = 0.0;
    Settings s_on = s_off; s_on.bp_adaptive = true; s_on.verbose = 0;
    Result r0 = solve(P, s_off);
    std::printf("[T2] OFF : status=%d it=%lld obj=%+.6e stat=%.1e feas=%.1e\n",
                (int)r0.info.status, (long long)r0.info.iters, r0.info.obj,
                r0.info.res_stat_rel, r0.info.res_feas_rel);
    Result r1 = solve(P, s_on);
    std::printf("[T2] ON  : status=%d it=%lld obj=%+.6e stat=%.1e feas=%.1e bp_ev=%lld bp_push=%lld\n",
                (int)r1.info.status, (long long)r1.info.iters, r1.info.obj,
                r1.info.res_stat_rel, r1.info.res_feas_rel,
                (long long)r1.info.n_bp_events, (long long)r1.info.n_bp_pushed);
    // NOTE : ce probleme synthetique cumule slack pathologique ET mauvais
    // conditionnement d'echelle (gradient ~1e8 non compense par le scaling
    // evalue a l'origine) ; le sauvetage de bout en bout se valide sur les
    // instances CUTEst reelles en echec (--bp-adaptive). Ici on asserte
    // l'ACTIVATION du mecanisme et la NON-DEGRADATION vs OFF.
    expect(r1.info.n_bp_pushed > 0, "T2 : etage 0 a pousse (signature detectee)");
    auto sev = [](Status st2) {
      return (st2 == Status::Solved || st2 == Status::SolvedSOC ||
              st2 == Status::DCAConverged) ? 0
           : (st2 == Status::DualDegenerate || st2 == Status::MaxIter) ? 1 : 2;
    };
    expect(sev(r1.info.status) <= sev(r0.info.status),
           "T2 : bp_adaptive ne degrade pas le statut vs OFF");
  }

  // ---- Test 3 : immunite structurelle (variables fixes / boites epsilon) ---
  {
    idx_t n = 60;
    Problem P; P.n = n; P.name = "bp_fixed_immune";
    P.Q = tridiag_indef(n, 1.5, 31);
    P.q.assign(n, 0.25);
    P.l.assign(n, -2.0); P.u.assign(n, 2.0);
    for (idx_t i = 0; i < 10; ++i) { P.l[i] = 1.0; P.u[i] = 1.0; }        // fixes
    for (idx_t i = 10; i < 20; ++i) { P.l[i] = 0.0; P.u[i] = 1e-11; }     // eps-boites
    Settings s_on; s_on.verbose = 0; s_on.bp_adaptive = true;
    Result r = solve(P, s_on);
    std::printf("[T3] ON  : status=%d it=%lld bp_ev=%lld bp_push=%lld\n",
                (int)r.info.status, (long long)r.info.iters,
                (long long)r.info.n_bp_events, (long long)r.info.n_bp_pushed);
    // Les composantes fixes/epsilon ne sont jamais poussees : tout push
    // eventuel ne peut venir que des composantes normales (slack sain ici).
    expect(r.info.n_bp_pushed == 0, "T3 : fixes/eps-boites jamais poussees (non movable)");
  }

  // ---- Test 4 : SAUVETAGE de bout en bout — slacks d'inegalites colles -----
  // bound_push=0 et Cx0 atterrissant EXACTEMENT sur cl : slack 0 => duaux
  // wl degeneres a l'init, Theta_s infini, iteres s figes. Etage 0 pousse
  // les slacks et re-initialise leurs duaux => la course part proprement.
  {
    idx_t n = 200, m = 30;
    Problem P; P.n = n; P.m = m; P.name = "bp_ineq_glued";
    P.Q = tridiag_indef(n, 1.2, 41);
    P.q.assign(n, 0.0);
    std::mt19937 gen(43); std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (auto& v : P.q) v = ud(gen);
    P.l.assign(n, -2.0); P.u.assign(n, 2.0);
    // C : m lignes = selection de coordonnees ; x0 interne = 0 => Cx0 = 0 = cl
    std::vector<idx_t> I(m), J(m); std::vector<double> V(m, 1.0);
    for (idx_t i = 0; i < m; ++i) { I[i] = i; J[i] = 3 * i; }
    P.C = csc_from_coo(m, n, I, J, V);
    P.cl.assign(m, 0.0);          // Cx0 = cl exactement -> slack 0
    P.cu.assign(m, 4.0);
    Settings s_off; s_off.verbose = 0; s_off.bound_push = 0.0; s_off.bound_frac = 0.0;
    Settings s_on = s_off; s_on.bp_adaptive = true;
    Result r0 = solve(P, s_off), r1 = solve(P, s_on);
    auto okst = [](Status st2) {
      return st2 == Status::Solved || st2 == Status::SolvedSOC ||
             st2 == Status::DCAConverged;
    };
    std::printf("[T4] OFF : status=%d it=%lld stat=%.1e | ON : status=%d it=%lld "
                "stat=%.1e bp_ev=%lld bp_push=%lld\n",
                (int)r0.info.status, (long long)r0.info.iters, r0.info.res_stat_rel,
                (int)r1.info.status, (long long)r1.info.iters, r1.info.res_stat_rel,
                (long long)r1.info.n_bp_events, (long long)r1.info.n_bp_pushed);
    expect(r1.info.n_bp_pushed >= m, "T4 : etage 0 a pousse les slacks colles");
    expect(okst(r1.info.status), "T4 : convergence AVEC bp_adaptive");
    if (okst(r0.info.status))
      std::printf("  (note : OFF converge aussi — la restauration d'interiorite\n"
                  "   INCONDITIONNELLE v0.16.0 couvre les slacks exactement nuls)\n");
  }

  // ---- Test 5 : interaction bound_mult_init_val (init duale uniforme) ------
  {
    idx_t n = 200, m = 30;
    Problem P; P.n = n; P.m = m; P.name = "bp_ineq_glued_bmiv";
    P.Q = tridiag_indef(n, 1.2, 41);
    P.q.assign(n, 0.0);
    std::mt19937 gen(43); std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (auto& v : P.q) v = ud(gen);
    P.l.assign(n, -2.0); P.u.assign(n, 2.0);
    std::vector<idx_t> I(m), J(m); std::vector<double> V(m, 1.0);
    for (idx_t i = 0; i < m; ++i) { I[i] = i; J[i] = 3 * i; }
    P.C = csc_from_coo(m, n, I, J, V);
    P.cl.assign(m, 0.0); P.cu.assign(m, 4.0);
    Settings s_on; s_on.verbose = 0; s_on.bound_push = 0.0; s_on.bound_frac = 0.0;
    s_on.bp_adaptive = true; s_on.bound_mult_init_val = 1.0;
    Result r = solve(P, s_on);
    std::printf("[T5] ON+bmiv=1 : status=%d it=%lld bp_push=%lld\n",
                (int)r.info.status, (long long)r.info.iters,
                (long long)r.info.n_bp_pushed);
    expect(r.info.status == Status::Solved || r.info.status == Status::SolvedSOC ||
           r.info.status == Status::DCAConverged,
           "T5 : convergence avec init duale uniforme + BP");
  }

  // ---- Test 6 : plafond de la soupape anti-blocage (v0.16.0, hors BP) ------
  // La soupape (reject_streak_rho_bump) etait la SEULE voie d'escalade de
  // rho a ne pas respecter rho_abs_max -- observe en campagne : rho=7.6e18
  // apres 1300+ rejets, 1400+ iterations gaspillees avant NumericalError.
  // Le plafond doit desormais TOUJOURS etre respecte, et un plafond plus bas
  // doit arreter le run PLUS TOT (message clair) plutot que de spiraler.
  {
    idx_t n = 40; const double L = 1e8;
    Problem P; P.n = n; P.p = 1; P.name = "valve_cap";
    P.Q = tridiag_indef(n, 1.0, 21);
    P.q.assign(n, 0.0);
    std::mt19937 gen(17); std::uniform_real_distribution<double> ud(0.5, 1.5);
    for (auto& v : P.q) v = ud(gen);
    std::vector<idx_t> I(n, 0), J(n); std::vector<double> V(n, 1.0);
    for (idx_t j = 0; j < n; ++j) J[j] = j;
    P.A = csc_from_coo(1, n, I, J, V);
    P.b.assign(1, (double)n * L);
    P.l.assign(n, L); P.u.assign(n, INF);
    Settings s_hi; s_hi.verbose = 0; s_hi.bound_push = 0.0; s_hi.bound_frac = 0.0;
    s_hi.rho_abs_max = 1e10; s_hi.max_iter = 3000;
    Settings s_lo = s_hi; s_lo.rho_abs_max = 1e6;
    Result r_hi = solve(P, s_hi), r_lo = solve(P, s_lo);
    std::printf("[T6] cap=1e10 : it=%lld rho=%.3e | cap=1e6 : it=%lld rho=%.3e msg=\"%s\"\n",
                (long long)r_hi.info.iters, r_hi.info.rho_final,
                (long long)r_lo.info.iters, r_lo.info.rho_final, r_lo.info.message.c_str());
    expect(r_hi.info.rho_final <= s_hi.rho_abs_max * 1.0000001,
           "T6 : rho_final jamais > rho_abs_max (cap eleve)");
    expect(r_lo.info.rho_final <= s_lo.rho_abs_max * 1.0000001,
           "T6 : rho_final jamais > rho_abs_max (cap bas)");
    expect(r_lo.info.iters < r_hi.info.iters,
           "T6 : plafond plus bas => arret plus tot (pas de spirale)");
    expect(r_lo.info.message.find("safety valve") != std::string::npos ||
           r_lo.info.message.find("absolute cap") != std::string::npos,
           "T6 : message diagnostique explicite sur l'arret par plafond");
  }

  // ---- Test 7 : detection a l'echelle recalibree (v0.17.0) -----------------
  // Slack d'inegalite CONTROLE a 1e-3 relatif (cl=-1e-3, x0 par defaut=0
  // pour les variables non bornees en x) : au-dessus de l'ancien seuil de
  // pathologie (1e-10), sous le nouveau bp_margin_tol (1e-2) -- exactement
  // l'ecart signale par le retour utilisateur. DOIT etre detecte et pousse.
  {
    idx_t n = 100, m = 100;
    Problem P; P.n = n; P.m = m; P.name = "bp_margin_scale";
    P.Q = tridiag_indef(n, 1.0, 51);
    P.q.assign(n, 0.0);
    P.l.assign(n, -50.0); P.u.assign(n, 50.0);   // box large, non contraignante
    std::vector<idx_t> I(m), J(m); std::vector<double> V(m, 1.0);
    for (idx_t i = 0; i < m; ++i) { I[i] = i; J[i] = i; }
    P.C = csc_from_coo(m, n, I, J, V);
    P.cl.assign(m, -1e-3);   // x0=0 par defaut => slack initial = 1e-3
    P.cu.assign(m, 50.0);
    Settings s_old; s_old.verbose = 0; s_old.bp_adaptive = true;
    s_old.bp_margin_tol = 1e-10;   // reproduit l'ANCIEN seuil : ne doit RIEN detecter
    Settings s_new = s_old; s_new.bp_margin_tol = 1e-2;   // NOUVEAU defaut
    Result r_old = solve(P, s_old), r_new = solve(P, s_new);
    std::printf("[T7] echelle 1e-3 : ancien seuil (1e-10) n_pushed=%lld | "
                "nouveau seuil (1e-2) n_pushed=%lld\n",
                (long long)r_old.info.n_bp_pushed, (long long)r_new.info.n_bp_pushed);
    expect(r_old.info.n_bp_pushed == 0,
           "T7 : l'ancien seuil (1e-10) ne detectait PAS cette echelle (1e-3)");
    expect(r_new.info.n_bp_pushed >= m,
           "T7 : le nouveau seuil (1e-2) detecte et pousse cette echelle");
  }
  // ---- Test 7b : preservation de la faisabilite (regression, v0.17.0) ------
  // La passe ciblee pousse chaque x_i INDEPENDAMMENT -- decouvert lors du
  // dev : casse Ax=b des que p>0 (feas 2.4e-14 -> 4.0e-3 avant correction).
  // Verrouille que la correction d'egalite post-push la restaure.
  {
    idx_t n = 40; const double L = 1e8;
    Problem P; P.n = n; P.p = 1; P.name = "bp_margin_feas_regression";
    P.Q = tridiag_indef(n, 1.0, 21);
    P.q.assign(n, 0.0);
    std::mt19937 gen(17); std::uniform_real_distribution<double> ud(0.5, 1.5);
    for (auto& v : P.q) v = ud(gen);
    std::vector<idx_t> I(n, 0), J(n); std::vector<double> V(n, 1.0);
    for (idx_t j = 0; j < n; ++j) J[j] = j;
    P.A = csc_from_coo(1, n, I, J, V);
    P.b.assign(1, (double)n * L);
    P.l.assign(n, L); P.u.assign(n, INF);
    Settings s; s.verbose = 0; s.bound_push = 0.0; s.bound_frac = 0.0; s.bp_adaptive = true;
    Result r = solve(P, s);
    std::printf("[T7b] feasibilite post-push (egalites presentes) : feas_rel=%.2e "
                "n_pushed=%lld\n", r.info.res_feas_rel, (long long)r.info.n_bp_pushed);
    expect(r.info.n_bp_pushed > 0, "T7b : la passe ciblee a pousse des composantes");
    expect(r.info.res_feas_rel <= 1e-6,
           "T7b : Ax=b restauree apres le push (pas de regression de faisabilite)");
  }
  // ---- Test 7c : precision du seuil -- pas de push au-dessus de la marge --
  {
    idx_t n = 100;
    Problem P; P.n = n; P.name = "bp_margin_precision";
    P.Q = tridiag_indef(n, 1.0, 61);
    P.q.assign(n, 0.0);
    // marge relative fixe a 5% (0.05), au-dessus de bp_margin_tol=1e-2 par
    // defaut : NE DOIT PAS etre poussee.
    P.l.assign(n, -1.0); P.u.assign(n, 1.0);
    Settings s; s.verbose = 0; s.bp_adaptive = true; s.bp_margin_tol = 1e-2;
    Result r = solve(P, s);
    std::printf("[T7c] point de depart par defaut (loin des bornes) : n_pushed=%lld\n",
                (long long)r.info.n_bp_pushed);
    expect(r.info.n_bp_pushed == 0,
           "T7c : aucune composante poussee quand la marge par defaut est confortable");
  }

  std::printf(fails == 0 ? "BP_TEST: ALL PASS\n" : "BP_TEST: %d FAIL\n", fails);
  return fails;
}
