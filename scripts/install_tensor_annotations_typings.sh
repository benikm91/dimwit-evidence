#!/usr/bin/env bash
# Make tensor_annotations' types visible to Pylance/pyright in VS Code.
#
# Installs tensor_annotations + its JAX stubs into .typings/ (git-ignored), NOT into the
# project venv. pyrightconfig.json applies them to tensor_annotations_case.py only: the
# JAX stubs are partial and would otherwise replace JAX's own types in every other file.
# Afterwards run "Python: Restart Language Server" (or reload the window) in VS Code.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf .typings/tensor_annotations
uv pip install --quiet --no-deps --target .typings/tensor_annotations \
  tensor_annotations==2.0.3 tensor_annotations_jax_stubs==2.0.3
echo "installed into .typings/tensor_annotations"
