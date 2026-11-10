#!/usr/bin/env bash
# tensor_annotations checks for every case that has a tensor_annotations_case.py.
# Deliberately NOT part of run_all.sh: the extra packages are pulled into a throwaway environment via `uv run --with`, so the
# project's lock file and the published results stay untouched.
#
# The tests in each tensor_annotations_case.py run mypy (02 onwards also pyright) on the file
# itself and assert which lines are rejected, then check the run-time behaviour.
set -uo pipefail
cd "$(dirname "$0")/.."

uv run --quiet \
  --with tensor_annotations==2.0.3 --with tensor_annotations_jax_stubs==2.0.3 --with mypy --with pyright \
  python -m pytest -q -p no:cacheprovider "$@" \
  cases/*/tensor_annotations_case.py
