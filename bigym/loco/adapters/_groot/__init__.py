"""Vendored NVIDIA GR00T-WholeBodyControl G1 Gear-WBC policy runtime.

The two files next to this one are the Python runtime of the official
Balance/Walk policy pair (``decoupled_wbc`` in
<https://github.com/NVlabs/GR00T-WholeBodyControl>, Apache-2.0), copied in
so the benchmark installs without a git dependency or a checkout.
``PROVENANCE.md`` records the upstream commit, the sha256 of the originals
and every edit made here. The ONNX weights live in ``bigym/loco/assets``.
"""

from pathlib import Path

from bigym.loco.adapters._groot.g1_gear_wbc_policy import G1GearWbcPolicy

POLICY_CONFIG_PATH = Path(__file__).resolve().parent / "g1_gear_wbc.yaml"

__all__ = ["G1GearWbcPolicy", "POLICY_CONFIG_PATH"]
