#!/usr/bin/env bash
# Interactive configuration wizard: writes config/<project>.json (and optional config/site.env).
set -euo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" "${PACKAGE_ROOT}/src/configure.py" "$@"
