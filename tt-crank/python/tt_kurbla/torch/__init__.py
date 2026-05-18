from . import _native  # noqa: F401  — loading the .so runs c10::register_privateuse1_backend("tt")
from ._device import register

import os
from pathlib import Path

_VENDORED_TT_METAL = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "tt-mlir"
    / "third_party"
    / "tt-metal"
    / "src"
    / "tt-metal"
)
# Not an `assert` — `python -O` strips those, and this check is the only thing
# standing between a half-cloned submodule and a baffling runtime error later.
if not _VENDORED_TT_METAL.is_dir():
    raise RuntimeError(
        f"vendored tt-metal not found at {_VENDORED_TT_METAL}; "
        "build at least once so tt-mlir-ep populates the submodule"
    )

os.environ["TT_METAL_HOME"] = str(_VENDORED_TT_METAL)
os.environ["TT_METAL_RUNTIME_ROOT"] = str(_VENDORED_TT_METAL)

register()
