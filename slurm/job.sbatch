#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
PACKAGE_ROOT="${1:?absolute package root required}"
CONFIG_PATH="${2:?absolute resolved config required}"
STAGE="${3:?stage required}"
CONFIG_SHA256="${4:?frozen config SHA256 required}"
# Never resolve source paths from /var/spool/slurmd or the submit working directory.
cd -- "${PACKAGE_ROOT}"
exec bash "${PACKAGE_ROOT}/run.sh" --config "${CONFIG_PATH}" --expected-config-sha256 "${CONFIG_SHA256}" "${STAGE}"
