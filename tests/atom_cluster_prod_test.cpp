#include "atom_cluster_prod.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
using C=lazyipdca::ProductionAtomClusterer<long long>;
using D=lazyipdca::AtomClusterDecision;
static std::vector<double> u(double a,double b){double n=std::sqrt(a*a+b*b);return{a/n,b/n};}
int main(){
  C::Params p;p.validate_sorted_indices=true;p.max_clusters=8;C c(p);
  std::vector<long long>a{0,2},b{1,3},bad{2,1};auto v=u(1,1);std::vector<double>vm{-v[0],-v[1]};
  auto r1=c.classify(a,v,-2);assert(r1.valid&&r1.decision==D::Novel);c.commit(r1,a,v,-2,1);assert(c.size()==1);
  auto r2=c.classify(a,vm,-2);assert(r2.decision==D::Existing&&r2.sim.direction>.999999);
  double ti=r2.tau_in_effective,to=r2.tau_out_effective;
  for(int k=0;k<30;++k){auto q=c.classify(a,v,-2);assert(q.tau_in_effective==ti&&q.tau_out_effective==to);}
  auto r3=c.classify(b,v,-2);assert(r3.decision==D::Novel);c.commit(r3,b,v,-2,1.1);assert(c.size()==2);
  assert(!c.classify(bad,v,-2).valid);assert(C::score_adjustment(r2,.2,.05)<=0);
  lazyipdca::ProductionAtomSeedStopper::Params sp;sp.window=8;sp.min_observations=6;sp.min_atoms=4;
  sp.max_novelty_rate=.2;sp.max_coverage_gain=.01;sp.max_marginal_lift_rel=.01;sp.confirmation_rounds=2;
  lazyipdca::ProductionAtomSeedStopper s(sp);for(int k=0;k<8;++k)s.observe_commit(D::Existing);
  assert(!s.after_round(.20,.005,8).stop);auto q1=s.after_round(.205,.005,8);assert(!q1.stop&&q1.confirmations==1);
  auto q2=s.after_round(.207,.004,8);assert(q2.stop&&q2.confirmations==2);
  std::cout<<"atom_cluster_prod_test PASS\n";
}
