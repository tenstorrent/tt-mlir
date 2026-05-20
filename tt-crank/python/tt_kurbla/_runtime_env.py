# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Runtime-environment helpers shared between tt_kurbla modules.

The Tracy wrapper and the torch backend both need to point tt-metal at a
real on-disk copy of itself (via ``TT_METAL_HOME`` / ``TT_METAL_RUNTIME_ROOT``).
The torch backend already sets these unconditionally at import time; this
module exposes the same logic so the Tracy wrapper (and any future re-export)
can reuse it without growing a circular dependency on the torch package.
"""

import os
from pathlib import Path


def _vendored_tt_metal() -> Path:
    """Path to the tt-metal source tree populated by tt-mlir-ep at build time."""
    # python/tt_kurbla/_runtime_env.py -> repo root -> third_party/...
    return (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "tt-mlir"
        / "third_party"
        / "tt-metal"
        / "src"
        / "tt-metal"
    )


def setup_tt_metal_home() -> None:
    """
    Ensure ``TT_METAL_HOME`` and ``TT_METAL_RUNTIME_ROOT`` point at a valid
    tt-metal tree.

    Resolution order:
      1. If both env vars are already set, validate the path exists. A stale
         override is a hard error — silently falling back would mask whatever
         the user was trying to point at.
      2. Otherwise default to the tt-mlir-ep submodule checkout
         ``third_party/tt-mlir/third_party/tt-metal/src/tt-metal/``.

    Mirrors the behavior of ``python/tt_kurbla/torch/__init__.py`` so that
    importing either module first leaves the environment in the same state.
    """
    user_override = os.getenv("TT_METAL_RUNTIME_ROOT")
    if user_override is not None:
        if not Path(user_override).is_dir():
            raise FileNotFoundError(
                f"TT_METAL_RUNTIME_ROOT is set to {user_override!r}, "
                f"but that directory does not exist."
            )
        # Honor the override for both env vars. The tracy CLI reads
        # TT_METAL_HOME, not TT_METAL_RUNTIME_ROOT, so we need to mirror.
        os.environ.setdefault("TT_METAL_HOME", user_override)
        return

    vendored = _vendored_tt_metal()
    if not vendored.is_dir():
        raise RuntimeError(
            f"vendored tt-metal not found at {vendored}; "
            "build at least once so tt-mlir-ep populates the submodule, "
            "or set TT_METAL_RUNTIME_ROOT explicitly."
        )

    os.environ["TT_METAL_HOME"] = str(vendored)
    os.environ["TT_METAL_RUNTIME_ROOT"] = str(vendored)
