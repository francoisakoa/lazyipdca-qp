#include "lazyipdca/mumps_solver.hpp"
#include <cstring>
#include <algorithm>

namespace lazyipdca {

namespace {
constexpr int JOB_INIT = -1, JOB_END = -2, JOB_ANALYZE = 1, JOB_FACTORIZE = 2, JOB_SOLVE = 3;
constexpr int USE_COMM_WORLD = -987654;
inline int& ICNTL(DMUMPS_STRUC_C& id, int i) { return id.icntl[i - 1]; }
inline double& CNTL(DMUMPS_STRUC_C& id, int i) { return id.cntl[i - 1]; }
}  // namespace

MumpsSym::MumpsSym() {
  std::memset(&id_, 0, sizeof(id_));
  id_.comm_fortran = USE_COMM_WORLD;
  id_.par = 1;
  id_.sym = 2;  // general symmetric indefinite
  id_.job = JOB_INIT;
  dmumps_c(&id_);
  // silence output
  ICNTL(id_, 1) = -1; ICNTL(id_, 2) = -1; ICNTL(id_, 3) = -1; ICNTL(id_, 4) = 0;
}

MumpsSym::~MumpsSym() {
  id_.job = JOB_END;
  dmumps_c(&id_);
}

void MumpsSym::set_matrix(idx_t n, std::vector<int>&& irn, std::vector<int>&& jcn,
                          idx_t nnz, int ordering, double pivot_tol) {
  n_ = n; nnz_ = nnz;
  irn_ = std::move(irn); jcn_ = std::move(jcn);
  a_.assign(nnz_, 0.0);
  id_.n = (MUMPS_INT)n_;
  id_.nnz = (MUMPS_INT8)nnz_;
  id_.irn = irn_.data();
  id_.jcn = jcn_.data();
  id_.a = a_.data();
  ICNTL(id_, 7) = ordering;   // 5 = METIS (validated default)
  ICNTL(id_, 13) = 1;         // sequential root => reliable inertia INFOG(12)
  ICNTL(id_, 14) = 60;        // workspace headroom (%)
  ICNTL(id_, 24) = 0;         // no null-pivot fixing (would perturb inertia)
  CNTL(id_, 1) = 1e-4;        // seuil de pivotage reduit: les KKT quasi-definis
                              // n'ont pas besoin de 0.01, et un seuil eleve fait
                              // exploser les pivots retardes quand rho est trop
                              // petit (workspace -9). Inertie exacte inchangee.
  ICNTL(id_, 10) = 2;         // 2 pas de raffinement iteratif a la resolution
  CNTL(id_, 3) = 0.0;         // default null pivot threshold
  analyzed_ = false;
}

void MumpsSym::analyze() {
  id_.job = JOB_ANALYZE;
  dmumps_c(&id_);
  if (id_.infog[0] < 0)
    throw std::runtime_error("MUMPS analysis failed, INFOG(1)=" + std::to_string(id_.infog[0]));
  analyzed_ = true;
}

bool MumpsSym::factorize(const double* values) {
  if (!analyzed_) analyze();
  std::memcpy(a_.data(), values, sizeof(double) * (size_t)nnz_);
  for (int attempt = 0; attempt < 6; ++attempt) {
    id_.job = JOB_FACTORIZE;
    dmumps_c(&id_);
    infog1_ = id_.infog[0];
    if (infog1_ >= 0) return true;
    if (infog1_ == -8 || infog1_ == -9 || infog1_ == -14 || infog1_ == -15 ||
        infog1_ == -17 || infog1_ == -20) {           // famille "workspace"
      id_.icntl[13] = id_.icntl[13] <= 0 ? 80 : id_.icntl[13] * 2;  // ICNTL(14)
      if (id_.icntl[13] > 2000) break;   // au-dela: pivots retardes explosifs,
      continue;                          // c'est un defaut de courbure, pas de RAM
    }
    break;
  }
  id_.icntl[13] = 40;  // reset pour la prochaine factorisation
  return false;
}

void MumpsSym::solve(std::vector<double>& rhs) {
  id_.rhs = rhs.data();
  id_.nrhs = 1;
  id_.lrhs = (MUMPS_INT)n_;
  id_.job = JOB_SOLVE;
  dmumps_c(&id_);
  if (id_.infog[0] < 0)
    throw std::runtime_error("MUMPS solve failed, INFOG(1)=" + std::to_string(id_.infog[0]));
}

}  // namespace lazyipdca
