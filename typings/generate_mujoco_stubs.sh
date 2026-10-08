#!/usr/bin/env bash
# Generate type stubs for MuJoCo's C++ bindings, which ship without any.
# Without them ty reports every mujoco.MjModel, mujoco.mj_step, ... as missing.
#
# Adapted from mjlab (typings/generate_mujoco_stubs.sh, Apache-2.0).
# postprocess_stubs.py then strips the platform-specific parts, so the
# committed stubs are the same whether they were generated on macOS or Linux.
# Rerun this after bumping mujoco.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --no-sync uses the existing .venv (where mujoco is importable) as is.
STUBGEN=(uv run --no-sync --with "pybind11-stubgen==2.5.5" pybind11-stubgen)

"${STUBGEN[@]}" mujoco -o "$SCRIPT_DIR" --ignore-all-errors
# mujoco.viewer needs GLFW, which a headless machine may not have; the
# committed viewer stub is then kept.
"${STUBGEN[@]}" mujoco.viewer -o "$SCRIPT_DIR" --ignore-all-errors \
  || echo "Could not generate mujoco.viewer stubs (no GLFW?), skipping."

uv run --no-sync python "$SCRIPT_DIR/postprocess_stubs.py"
echo "MuJoCo stubs written to $SCRIPT_DIR/mujoco/"
