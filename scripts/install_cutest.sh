#!/usr/bin/env bash
set -euo pipefail
PREFIX=${CUTEST_PREFIX:-$HOME/opt/cutest}
mkdir -p "$PREFIX"; cd "$PREFIX"
[ -d archdefs/.git ] || git clone https://github.com/ralna/ARCHDefs archdefs
[ -d sifdecode/.git ] || git clone https://github.com/ralna/SIFDecode sifdecode
[ -d cutest/.git ] || git clone https://github.com/ralna/CUTEst cutest
[ -d mastsif/.git ] || git clone https://bitbucket.org/optrove/sif mastsif
export ARCHDEFS="$PREFIX/archdefs" SIFDECODE="$PREFIX/sifdecode" CUTEST="$PREFIX/cutest" MASTSIF="$PREFIX/mastsif" MYARCH=${MYARCH:-pc64.lnx.gfo}
curl -fsSL https://raw.githubusercontent.com/jfowkes/pycutest/v1.8.2/.install_cutest.sh -o "$PREFIX/install_cutest_pycutest_1.8.2.sh"
bash "$PREFIX/install_cutest_pycutest_1.8.2.sh"
cat > "$PREFIX/env.sh" <<EOF
export ARCHDEFS="$ARCHDEFS"
export SIFDECODE="$SIFDECODE"
export CUTEST="$CUTEST"
export MASTSIF="$MASTSIF"
export MYARCH="$MYARCH"
EOF
echo "CUTEst environment written to $PREFIX/env.sh"
