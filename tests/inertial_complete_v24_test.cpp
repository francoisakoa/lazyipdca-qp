// INERTIA_COMPLETE_AMRS_V24 deterministic header test.
#include "lazyipdca/problem.hpp"
#include "lazyipdca/inertial_complete_v24.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
using namespace lazyipdca;
static void ck(bool q,const char* s){std::printf("[%s] %s\n",q?"PASS":"FAIL",s);if(!q)std::exit(2);} 
int main(){
  Settings st; st.dca_momentum_complete=true; st.dca_acceptance=DcaAcceptanceMode::Lyapunov;
  st.dca_momentum_theta_max=0.45; st.dca_momentum_theta=0.25; st.dca_lyapunov_theta_bar=0.25;
  st.dca_lyapunov_cap_momentum_theta=true; st.dca_lyapunov_enforce_euclidean_bound=true;
  ck(std::fabs(complete_theta_cap(st)-0.25)<1e-15,"common Lyapunov theta cap");
  bool cp=false; double g=complete_euclidean_gamma_cap(2.0,0.25,2.0,st,&cp);
  ck(std::fabs(g-2.0)<1e-15 && !cp,"Euclidean gamma=rho is not spuriously capped");
  g=complete_euclidean_gamma_cap(20.0,0.25,2.0,st,&cp);
  ck(std::fabs(g-2.0)<1e-15 && cp,"Euclidean certified gamma cap uses theta_bar/theta");
  double t=complete_transport_theta(0.20,10.0,1.0,st,&cp);
  ck(std::fabs(t-0.25)<1e-15 && cp,"rho transport obeys Lyapunov cap");
  AtomSet a; a.idx={{0},{1}}; a.val={{1.0},{1.0}}; a.beta={4.0,1.0};
  std::vector<double> v={1.0,2.0};
  auto ew=complete_atomic_weights(DcaMomentumAtomWeighting::Energy,v,a,2.0,true);
  const double E=2.0*5.0+4.0*1.0+1.0*4.0; // 18
  ck(std::fabs(ew.energy_rho_share-10.0/E)<1e-14,"energy rho share");
  ck(std::fabs(ew.atom[0]-std::sqrt(4.0/E))<1e-14,"energy atom 0 weight");
  ck(std::fabs(ew.atom[1]-std::sqrt(4.0/E))<1e-14,"energy atom 1 weight");
  auto cw=complete_atomic_weights(DcaMomentumAtomWeighting::Cosine,v,a,2.0,true);
  ck(std::fabs(cw.atom[0]-1.0/std::sqrt(5.0))<1e-14,"cosine atom 0");
  ck(std::fabs(cw.atom[1]-2.0/std::sqrt(5.0))<1e-14,"cosine atom 1");
  std::vector<double> th={0.2,0.1}, p; complete_atomic_force(v,a,2.0,0.25,th,p);
  ck(p.size()==2 && std::isfinite(p[0]) && std::isfinite(p[1]),"componentwise atomic force finite");
  bool fc=false; double pi=complete_cap_force(p,v,a,2.0,0.1,&fc);
  ck(fc && pi>0.0,"relative force cap active");
  std::puts("INERTIAL_COMPLETE_V24_TEST: ALL PASS"); return 0;
}
