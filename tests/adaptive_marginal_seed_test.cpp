#include "adaptive_marginal_seed.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using lazyipdca::AdaptiveMarginalSeedController;
int main(){
  AdaptiveMarginalSeedController c; double cov=0.0;
  double dc[4]={.08,.06,.05,.04}, lf[4]={.08,.06,.05,.04};
  for(std::size_t k=1;k<=4;k++){cov+=dc[k-1];auto d=c.observe(cov,lf[k-1],k);assert(d.observed&&!d.stop);}
  auto s=c.stats(); assert(s.references_ready); assert(std::fabs(s.coverage_threshold-2.5e-3)<1e-15); assert(std::fabs(s.lift_threshold-1e-3)<1e-15);
  for(std::size_t k=5;k<=7;k++){cov+=1e-4;auto d=c.observe(cov,5e-4,k);assert(!d.stop);} auto d8=c.observe(cov+1e-4,5e-4,8); assert(d8.stop&&c.stats().stop_atom==8);
  AdaptiveMarginalSeedController r; cov=0.0; for(std::size_t k=1;k<=4;k++){cov+=.05;r.observe(cov,.05,k);} for(std::size_t k=5;k<=7;k++){cov+=1e-4;r.observe(cov,5e-4,k);} cov+=.02;auto dh=r.observe(cov,.02,8);assert(!dh.stop&&dh.confirmations==0);auto dup=r.observe(cov,0,8);assert(!dup.observed);auto jump=r.observe(cov+1e-4,5e-4,10);assert(jump.atom_jump==2&&r.stats().multi_atom_jumps==1);
  std::cout<<"adaptive_marginal_seed_test PASS\n";
}
