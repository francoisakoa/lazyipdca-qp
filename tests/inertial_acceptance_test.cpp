#include "lazyipdca/solver.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace lazyipdca;

static Problem make_problem() {
  Problem P; P.n=12; P.p=0; P.m=0; P.name="inertial_matrix_smoke";
  std::vector<idx_t> I,J; std::vector<double> V;
  for (idx_t i=0;i<P.n;++i) {
    I.push_back(i); J.push_back(i);
    V.push_back((i%3==0) ? -1.0 : 2.0 + 0.1*i);
  }
  P.Q.U = csc_from_coo(P.n,P.n,I,J,V);
  P.q.assign(P.n,0.0);
  for (idx_t i=0;i<P.n;++i) P.q[i]=0.05*(double)(i+1);
  P.l.assign(P.n,-2.0); P.u.assign(P.n,2.0);
  return P;
}

int main() {
  int fails=0;
  const DcaMomentumMode mm[] = {
    DcaMomentumMode::Original,
    DcaMomentumMode::InertialEuclidean,
    DcaMomentumMode::InertialGlobal,
    DcaMomentumMode::InertialAtomic
  };
  const DcaAcceptanceMode aa[] = {
    DcaAcceptanceMode::Monotone,
    DcaAcceptanceMode::Window,
    DcaAcceptanceMode::Lyapunov
  };
  for (auto m : mm) for (auto a : aa) {
    Settings st;
    st.verbose=0; st.max_iter=250; st.time_limit=30.0;
    st.linear_backend=3; // QDLDL: self-contained test backend
    st.dca_momentum=m; st.dca_acceptance=a;
    st.dca_momentum_start=1;
    st.dca_momentum_theta=0.2; st.dca_momentum_theta_max=0.45;
    st.dca_lyapunov_theta_bar=0.25;
    st.dca_accept_window=4;
    st.initial_atoms=1; st.max_atoms=6; st.atoms_per_update=1;
    st.terminal_polish=false;
    Result r=solve(make_problem(),st);
    bool finite=std::isfinite(r.info.obj) && std::isfinite(r.info.res_feas_rel);
    bool ok=finite && r.info.status != Status::NumericalError;
    if (m != DcaMomentumMode::Original)
      ok = ok && (r.info.n_momentum_steps > 0 || r.info.n_momentum_resets > 0);
    std::printf("%-20s %-8s status=%d it=%lld acc=%lld rej=%lld mom=%lld reset=%lld theta=%.3g %s\n",
      momentum_mode_name(m), acceptance_mode_name(a), (int)r.info.status,
      (long long)r.info.iters, (long long)r.info.n_accepts,
      (long long)r.info.n_rejects, (long long)r.info.n_momentum_steps,
      (long long)r.info.n_momentum_resets, r.info.momentum_theta_final,
      ok?"PASS":"FAIL");
    if (!ok) ++fails;
  }
  std::printf(fails ? "INERTIAL ACCEPTANCE MATRIX: %d FAILURES\n" : "INERTIAL ACCEPTANCE MATRIX: ALL PASS\n", fails);
  return fails?1:0;
}
