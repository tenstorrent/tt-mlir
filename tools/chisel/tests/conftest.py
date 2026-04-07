# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for Chisel tests.

Adds tools/golden to sys.path so that golden.metrics can be imported
without triggering the full golden package init (which requires MLIR
bindings). The metrics module itself only depends on torch.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_TOOLS_GOLDEN = Path(__file__).resolve().parents[3] / "tools" / "golden"


def _load_metrics_as_golden_submodule():
    """Load tools/golden/metrics.py and register it as golden.metrics."""
    if "golden" not in sys.modules:
        # Create a stub package so 'golden.metrics' is importable
        pkg = types.ModuleType("golden")
        pkg.__path__ = [str(_TOOLS_GOLDEN)]
        pkg.__package__ = "golden"
        sys.modules["golden"] = pkg

    spec = importlib.util.spec_from_file_location(
        "golden.metrics", _TOOLS_GOLDEN / "metrics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["golden.metrics"] = mod
    spec.loader.exec_module(mod)


_load_metrics_as_golden_submodule()
