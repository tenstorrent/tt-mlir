# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for Chisel tests.

Adds tools/golden to sys.path so that golden.metrics can be imported
without triggering the full golden package init (which requires MLIR
bindings). The metrics module itself only depends on torch.

Also loads golden.mapping (and thus GoldenMapTensor / get_golden_function)
by mocking out the dialect imports that are not available in the test
environment (e.g. stablehlo, d2m, sdy, debug).
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_TOOLS_GOLDEN = Path(__file__).resolve().parents[3] / "tools" / "golden"
_TOOLS_CHISEL = Path(__file__).resolve().parents[1]

# Add tools/chisel to sys.path so that 'chisel' package is importable
if str(_TOOLS_CHISEL) not in sys.path:
    sys.path.insert(0, str(_TOOLS_CHISEL))


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


class _AutoAttrModule(types.ModuleType):
    """A module stub where any attribute access returns another _AutoAttrModule."""

    def __getattr__(self, name: str):
        child = _AutoAttrModule(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return _AutoAttrModule(f"{self.__name__}()")


def _stub_missing_dialect(dialect_name: str):
    """Insert an auto-attribute stub for a ttmlir dialect not compiled in."""
    full_name = f"ttmlir.dialects.{dialect_name}"
    if full_name not in sys.modules:
        stub = _AutoAttrModule(full_name)
        sys.modules[full_name] = stub
        # Also register as attribute on the parent ttmlir.dialects module
        if "ttmlir.dialects" in sys.modules:
            setattr(sys.modules["ttmlir.dialects"], dialect_name, stub)


def _load_golden_mapping():
    """
    Load tools/golden/mapping.py as golden.mapping.

    mapping.py imports several ttmlir dialects (stablehlo, d2m, sdy, debug)
    that may not be compiled in the current environment. Stubs are inserted
    for any missing dialect before loading the module.
    """
    if "golden.mapping" in sys.modules:
        return

    # Ensure ttmlir.dialects is importable first
    try:
        import ttmlir.dialects  # noqa: F401
    except ImportError:
        pass

    # Stub out dialects that mapping.py requires but may not be compiled in
    for dialect in ("stablehlo", "d2m", "sdy", "debug"):
        try:
            importlib.import_module(f"ttmlir.dialects.{dialect}")
        except (ImportError, ModuleNotFoundError):
            _stub_missing_dialect(dialect)

    spec = importlib.util.spec_from_file_location(
        "golden.mapping", _TOOLS_GOLDEN / "mapping.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["golden.mapping"] = mod
    spec.loader.exec_module(mod)

    # Expose top-level symbols on the golden package
    golden_pkg = sys.modules.get("golden")
    if golden_pkg is not None:
        for attr in ("GoldenMapTensor", "get_golden_function", "GOLDEN_MAPPINGS"):
            if hasattr(mod, attr):
                setattr(golden_pkg, attr, getattr(mod, attr))


_load_metrics_as_golden_submodule()
_load_golden_mapping()
