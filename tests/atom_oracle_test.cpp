#include "lazyipdca/atoms.hpp"
#include "lazyipdca/problem.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  using namespace lazyipdca;
  Settings st;
  if (!st.atom_oracle_enable || !st.atom_energy_sparsify) {
    std::fprintf(stderr, "adaptive oracle defaults not enabled\n");
    return 1;
  }
  std::vector<double> v = {0.999, 0.03, 0.01, 1e-6, 1e-8};
  std::vector<idx_t> idx;
  std::vector<double> val;
  double energy = 0.0;
  if (!AtomSet::sparsify_energy(v, 5, 0.999, 1, idx, val, &energy)) return 2;
  if (!(energy >= 0.999 && idx.size() < v.size())) {
    std::fprintf(stderr, "energy sparsifier failed: energy=%g k=%zu\n", energy, idx.size());
    return 3;
  }
  double n2 = 0.0; for (double x : val) n2 += x*x;
  if (std::fabs(n2 - 1.0) > 1e-12) return 4;
  std::printf("atom_oracle_test PASS k=%zu energy=%.9f\n", idx.size(), energy);
  return 0;
}
