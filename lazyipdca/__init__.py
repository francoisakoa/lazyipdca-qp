"""LazyIPDCA — interior-point DC algorithm for large-scale nonconvex sparse QP.

    min 1/2 x'Qx + q'x + c0
    s.t. A x = b,   cl <= C x <= cu,   l <= x <= u

Q may be indefinite.  The solver separates the DC-model regularization rho
from the numerical KKT-stability regularizer rho_x.  Controlled CONST/DYN/lazy
variants are available for the OMS ablation; DYN uses an explicit full-space
curvature check before committing smaller model shifts.  KKT inertia is an
independent numerical diagnostic, not a substitute for that DC curvature test.
KKT residuals are measured in ORIGINAL coordinates.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from ._lazyipdca import (  # noqa: F401
    Settings, Status, Info, solve_csc,
    DcaMomentumMode, DcaAcceptanceMode, DcaEuclideanGammaMode,
    DcRegularizationMode, RhoInitMethod,
)

__all__ = [
    "solve", "Settings", "Status", "Info",
    "DcaMomentumMode", "DcaAcceptanceMode", "DcaEuclideanGammaMode",
    "DcRegularizationMode", "RhoInitMethod", "configure_atom_strategy",
]

_EMPTY_P = np.zeros(1, dtype=np.int64)
_EMPTY_I = np.zeros(0, dtype=np.int64)
_EMPTY_V = np.zeros(0, dtype=np.float64)


def configure_atom_strategy(settings: Settings, strategy: str = "adaptive") -> Settings:
    """Configure one mutually-exclusive atomic strategy.

    adaptive : pooled activity-aware oracle + energy support compression
    legacy   : historical v0.18.1 atomic insertion path
    scalar   : LAZY scalar regularization with H_k = 0
    auto     : exact legacy seeding, then structural full-coverage AUTO switch
    """
    mode = str(strategy).strip().lower()
    valid = {"adaptive", "legacy", "scalar", "auto"}
    if mode not in valid:
        raise ValueError(
            f"unknown atom strategy {strategy!r}; expected one of {sorted(valid)}"
        )

    # Important for Settings objects that may be reused.
    settings.auto_atoms_enable = False
    settings.auto_portfolio_enable = False

    if mode == "adaptive":
        settings.atoms_enable = True
        settings.atom_oracle_enable = True
        settings.atom_energy_sparsify = True
        settings.atom_oracle_retune_enable = True
        settings.atom_oracle_rhox_rollback = False

    elif mode == "legacy":
        # TRUE LEGACY v0.18.1 atomic policy.
        # Keep user-controlled campaign axes (coverage, max_atoms, support size)
        # untouched, but pin the historical insertion mechanism explicitly.
        settings.atoms_enable = True

        # No modern oracle / compression / retuning.
        settings.atom_oracle_enable = False
        settings.atom_energy_sparsify = False
        settings.atom_oracle_retune_enable = False
        settings.atom_oracle_rhox_rollback = False

        # Historical progressive convexification path.
        # Historical v0.18.1 campaign configuration.
        # The source default was OFF, but the validated legacy campaign
        # explicitly ran with --atom-events --atom-event-every 50.
        settings.atom_event_enable = True
        settings.atom_event_every = 50
        settings.atom_update_every_accepts = 20
        settings.atom_update_every_iters = 10
        settings.atom_selection_enable = False
        settings.atoms_per_update = 1
        settings.atoms_freeze_after_seed = False

        # Sparse-beta recertification is a later experimental mechanism.
        settings.atom_sparse_recertify_beta = False

    elif mode == "scalar":
        settings.atoms_enable = False
        settings.atom_oracle_enable = False
        settings.atom_energy_sparsify = False
        settings.atom_oracle_retune_enable = False
        settings.atom_oracle_rhox_rollback = False

    else:  # auto
        # AUTO starts from EXACTLY the historical legacy policy.
        settings.atoms_enable = True
        settings.atom_oracle_enable = False
        settings.atom_energy_sparsify = False
        settings.atom_oracle_retune_enable = False
        settings.atom_oracle_rhox_rollback = False
        settings.auto_atoms_enable = True
        settings.auto_portfolio_enable = True

    return settings



def _csc_parts(M, rows, cols, upper=False):
    if M is None:
        ptr = np.zeros(cols + 1, dtype=np.int64)
        return ptr, _EMPTY_I, _EMPTY_V
    M = sp.csc_matrix(M)
    if M.shape != (rows, cols):
        raise ValueError(f"expected shape {(rows, cols)}, got {M.shape}")
    if upper:
        M = sp.triu(M, format="csc")
    M.sort_indices()
    return (M.indptr.astype(np.int64), M.indices.astype(np.int64),
            M.data.astype(np.float64))


def solve(Q, q, A=None, b=None, C=None, cl=None, cu=None, l=None, u=None,
          c0=0.0, settings: Settings | None = None, name: str = "qp",
          **kwargs):
    """Solve the QP.  Q, A, C: any scipy.sparse (or dense) matrices.

    Only the upper triangle of Q is used (Q is symmetrized implicitly).
    Infinite entries of cl/cu/l/u encode one-sided or absent bounds.
    Extra keyword arguments are applied to a fresh Settings object,
    e.g. solve(..., verbose=2, eps_stat=1e-8).
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    n = q.size
    p = 0 if A is None else sp.csc_matrix(A).shape[0]
    m = 0 if C is None else sp.csc_matrix(C).shape[0]

    st = settings if settings is not None else Settings()

    # Friendly string aliases for the two A/B axes.  The C++ API remains
    # strongly typed; the high-level Python API accepts the same public names
    # as the CUTEst runner.
    momentum_aliases = {
        "original": DcaMomentumMode.Original,
        "none": DcaMomentumMode.Original,
        "off": DcaMomentumMode.Original,
        "inertiel_euclidien": DcaMomentumMode.InertialEuclidean,
        "euclidien": DcaMomentumMode.InertialEuclidean,
        "euclidean": DcaMomentumMode.InertialEuclidean,
        "inertiel_global": DcaMomentumMode.InertialGlobal,
        "global": DcaMomentumMode.InertialGlobal,
        "inertiel_atomique": DcaMomentumMode.InertialAtomic,
        "atomique": DcaMomentumMode.InertialAtomic,
        "atomic": DcaMomentumMode.InertialAtomic,
    }
    acceptance_aliases = {
        "monotone": DcaAcceptanceMode.Monotone,
        "window": DcaAcceptanceMode.Window,
        "lyapunov": DcaAcceptanceMode.Lyapunov,
    }
    gamma_aliases = {
        "fixed": DcaEuclideanGammaMode.Fixed,
        "rayleigh": DcaEuclideanGammaMode.MetricRayleigh,
        "metric-rayleigh": DcaEuclideanGammaMode.MetricRayleigh,
    }
    regularization_aliases = {
        "legacy": DcRegularizationMode.Legacy,
        "const": DcRegularizationMode.Const,
        "fixed": DcRegularizationMode.Const,
        "dyn": DcRegularizationMode.Dyn,
        "dynamic": DcRegularizationMode.Dyn,
        "lazy": DcRegularizationMode.Lazy,
        "lowrank": DcRegularizationMode.Lazy,
        "low-rank": DcRegularizationMode.Lazy,
    }
    rho_init_aliases = {
        "auto": RhoInitMethod.Auto,
        "user": RhoInitMethod.User,
        "lanczos": RhoInitMethod.Lanczos,
        "gershgorin": RhoInitMethod.Gershgorin,
        "hint": RhoInitMethod.Hint,
    }

    for k, v in kwargs.items():
        if not hasattr(st, k):
            raise TypeError(f"unknown setting '{k}'")
        if isinstance(v, str):
            key = v.lower()
            if k == "dca_momentum":
                if key not in momentum_aliases:
                    raise ValueError(f"unknown dca_momentum '{v}'")
                v = momentum_aliases[key]
            elif k == "dca_acceptance":
                if key not in acceptance_aliases:
                    raise ValueError(f"unknown dca_acceptance '{v}'")
                v = acceptance_aliases[key]
            elif k == "dca_momentum_gamma_mode":
                if key not in gamma_aliases:
                    raise ValueError(f"unknown dca_momentum_gamma_mode '{v}'")
                v = gamma_aliases[key]
            elif k == "regularization_mode":
                if key not in regularization_aliases:
                    raise ValueError(f"unknown regularization_mode '{v}'")
                v = regularization_aliases[key]
            elif k == "rho_init_method":
                if key not in rho_init_aliases:
                    raise ValueError(f"unknown rho_init_method '{v}'")
                v = rho_init_aliases[key]
        setattr(st, k, v)

    Qp, Qi, Qv = _csc_parts(Q, n, n, upper=True)
    Ap, Ai, Av = _csc_parts(A, p, n) if p else (_EMPTY_P, _EMPTY_I, _EMPTY_V)
    Cp, Ci, Cv = _csc_parts(C, m, n) if m else (_EMPTY_P, _EMPTY_I, _EMPTY_V)

    def _vec(v, size, default):
        if v is None:
            return np.full(size, default, dtype=np.float64)
        return np.asarray(v, dtype=np.float64).ravel()

    b_ = _vec(b, p, 0.0)
    cl_ = _vec(cl, m, -np.inf)
    cu_ = _vec(cu, m, np.inf)
    l_ = _vec(l, n, -np.inf)
    u_ = _vec(u, n, np.inf)

    return solve_csc(n, p, m, Qp, Qi, Qv, q, float(c0),
                     Ap, Ai, Av, b_, Cp, Ci, Cv, cl_, cu_, l_, u_, st, name)
