#!/usr/bin/env bash
set -euo pipefail
ENV_FILE=${1:-environment.yml}
ENV_NAME=${ENV_NAME:-ipdca-qp}
conda env create -f "$ENV_FILE" || conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
echo "Activate with: conda activate $ENV_NAME"
