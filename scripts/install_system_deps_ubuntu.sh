#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ gfortran cmake ninja-build git curl wget libopenblas-dev liblapack-dev libeigen3-dev libmumps-seq-dev libmetis-dev python3-dev meson pkg-config
