#!/usr/bin/env bash
# Full experiment: Python reconstructions, then the DimWit compile checks.
set -uo pipefail
cd "$(dirname "$0")"

echo "=============================================================="
echo " 1/2  Python reconstructions (plain / plain JAX / jaxtyping)"
echo "=============================================================="
if command -v uv >/dev/null; then
  uv run --quiet pytest -q "$@"
else
  python3 -m pytest -q "$@"
fi
py=$?

echo
echo "=============================================================="
echo " 2/2  DimWit compile checks"
echo "=============================================================="
./scripts/check_scala.sh
sc=$?

echo
[[ $py -eq 0 && $sc -eq 0 ]] && echo "ALL GREEN" || echo "see output above"
exit $(( py != 0 || sc != 0 ))
