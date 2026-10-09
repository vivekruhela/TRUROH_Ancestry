#!/usr/bin/env bash
# Site settings written by configure.sh (modules, conda) are loaded first if present.
_TRUROH_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -r "${_TRUROH_ENV_DIR}/site.env" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${_TRUROH_ENV_DIR}/site.env"
  set -u
fi
# Use an already activated environment. Optional modules are selected by the
# caller, then inherited by SLURM jobs; no cohort-specific edits are needed.
# Example on the user's cluster:
# export TRUROH_MODULES="Plink/1.9.10 BCFTOOLS"
if [[ -n "${TRUROH_MODULES:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    if [[ -r /etc/profile.d/modules.sh ]]; then
      source /etc/profile.d/modules.sh
    elif [[ -r /usr/share/lmod/lmod/init/bash ]]; then
      source /usr/share/lmod/lmod/init/bash
    fi
  fi
  if ! type module >/dev/null 2>&1; then
    echo "ERROR: TRUROH_MODULES is set but the module command is unavailable; initialize your cluster modules or use an activated environment." >&2
    return 1
  fi
  read -r -a TRUROH_MODULE_LIST <<< "${TRUROH_MODULES}"
  module load "${TRUROH_MODULE_LIST[@]}"
fi
# To activate conda in batch shells, set the actual installation path explicitly:
# source /your/conda/etc/profile.d/conda.sh
# conda activate truroh-grid1433
export PYTHON_BIN="${PYTHON_BIN:-python3}"
