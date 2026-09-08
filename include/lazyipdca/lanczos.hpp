#pragma once
#include "sparse.hpp"
#include "atoms.hpp"
#include <memory>

namespace lazyipdca {

class MumpsSym;

// Orthogonal projector onto ker(A) via factorized augmented system.
class KerProjector {
 public:
  KerProjector(idx_t n, const SparseCSC& A, int ordering, double delta = 1e-10);
  ~KerProjector();
  void apply(const double* v, double* out) const;  // out = P v
  bool ok() const { return p_ == 0 || ok_; }
 private:
  idx_t n_, p_;
  bool ok_ = false;
  std::vector<double> vals_;
  std::unique_ptr<MumpsSym> mumps_;
};

struct LanczosReport {
  double lmin = 0.0;
  double residual = INF;
  idx_t iters = 0;
  bool converged = false;
  std::vector<double> ritz_low;   // plus petites valeurs de Ritz (profil spectral)
};

// ritz_out (optionnel) recoit jusqu'a ritz_k vecteurs de Ritz associes aux
// plus petites valeurs (ordre croissant), extraits du MEME run de Lanczos.
// Pp == nullptr : Lanczos en ESPACE COMPLET (defaut du zip, garantit K_hat
// PSD via rho_eff) ; Pp != nullptr : projete sur ker(A) (certificat SOC).
LanczosReport projected_lanczos_lmin(const SparseSymUpper& Q, const KerProjector* Pp,
                                     idx_t maxit, double tol, unsigned seed = 42,
                                     const AtomSet* atoms = nullptr,
                                     std::vector<std::vector<double>>* ritz_out = nullptr,
                                     idx_t ritz_k = 1);

}  // namespace lazyipdca
