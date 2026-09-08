#include "lazyipdca/inertia.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace lazyipdca;

static bool close(double a, double b, double tol=1e-12) { return std::fabs(a-b) <= tol; }

int main() {
  int fail = 0;
  AtomSet A;
  A.idx = {{0}};
  A.val = {{1.0}};
  A.beta = {2.0};
  std::vector<double> v{3.0, 4.0}, p;

  build_dca_momentum_force(DcaMomentumMode::Original, v, 1.0, A, 0.2, 2.0,
                           1.0, 0.0, 1.0, p);
  bool t1 = close(p[0],0.0) && close(p[1],0.0);
  std::printf("original force zero: %s\n", t1?"PASS":"FAIL"); fail += !t1;

  build_dca_momentum_force(DcaMomentumMode::InertialEuclidean, v, 1.0, A, 0.2, 2.0,
                           1.0, 0.0, 1.0, p);
  bool t2 = close(p[0],1.2) && close(p[1],1.6);
  std::printf("euclidean force: %s\n", t2?"PASS":"FAIL"); fail += !t2;

  build_dca_momentum_force(DcaMomentumMode::InertialGlobal, v, 1.0, A, 0.2, 1.0,
                           1.0, 0.0, 1.0, p);
  bool t3 = close(p[0],1.8) && close(p[1],0.8);
  std::printf("global metric force: %s\n", t3?"PASS":"FAIL"); fail += !t3;

  build_dca_momentum_force(DcaMomentumMode::InertialAtomic, v, 1.0, A, 0.2, 1.0,
                           0.5, 0.1, 1.0, p);
  // |cos(e1,v)|=3/5, atom weight=.1+.9*.6=.64.
  bool t4 = close(p[0],1.068) && close(p[1],0.4);
  std::printf("atom-wise aligned force: %s\n", t4?"PASS":"FAIL"); fail += !t4;

  // For admissible atom-wise coefficients J <= theta M.  Verify the quadratic
  // inequality on representative directions without assuming orthogonality.
  const double theta=0.3;
  bool t5=true;
  for (int k=0;k<9;++k) {
    std::vector<double> z{(double)(k-4), (double)(2*k-7)};
    build_dca_momentum_force(DcaMomentumMode::InertialAtomic, z, 1.5, A, theta, 1.0,
                             0.8, 0.2, 2.0, p);
    double lhs = dot(z,p);
    double rhs = theta*dca_metric_quad(z,1.5,A);
    if (lhs > rhs + 1e-11 || lhs < -1e-11) t5=false;
  }
  std::printf("atom-wise operator bound: %s\n", t5?"PASS":"FAIL"); fail += !t5;

  bool t6 = std::string(momentum_mode_name(DcaMomentumMode::InertialAtomic)) == "inertiel_atomique" &&
            std::string(acceptance_mode_name(DcaAcceptanceMode::Lyapunov)) == "lyapunov";
  std::printf("public names: %s\n", t6?"PASS":"FAIL"); fail += !t6;

  std::printf(fail ? "INERTIA MODES: %d FAILURES\n" : "INERTIA MODES: ALL PASS\n", fail);
  return fail ? 1 : 0;
}
