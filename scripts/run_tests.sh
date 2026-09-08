#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD=${BUILD:-$ROOT/build}
[ -d "$BUILD" ] || "$ROOT/scripts/build.sh"
for t in smoke bp_test best_feasible_test inertia_modes_test inertial_acceptance_test rho_policy_test atom_oracle_test; do [ -x "$BUILD/$t" ] && "$BUILD/$t"; done
