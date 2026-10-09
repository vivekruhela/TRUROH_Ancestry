#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Both launchers use the same parser. No cohort is selected implicitly.
# Retain the old positional JSON form as an explicit (never default) selection.
if [[ $# -gt 0 && "${1}" != -* ]]; then
  CONFIG_ARGUMENT="$1"
  shift
  set -- --config "${CONFIG_ARGUMENT}" "$@"
fi
exec bash "${PACKAGE_ROOT}/run.sh" submit "$@"
