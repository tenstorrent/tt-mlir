# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime-environment helpers shared between tt_crank modules.

The Tracy wrapper and the torch backend both need to point tt-metal at a
real on-disk copy of itself (via ``TT_METAL_HOME`` / ``TT_METAL_RUNTIME_ROOT``).
The torch backend already sets these unconditionally at import time; this
module exposes the same logic so the Tracy wrapper (and any future re-export)
can reuse it without growing a circular dependency on the torch package.
"""

import os
from pathlib import Path


def _wheel_tt_metal() -> Path:
    """Path to the tt-metal runtime tree bundled inside an installed wheel.

    third_party/CMakeLists.txt stages it at ``tt_crank/tt-metal`` (a sibling of
    this file inside the ``tt_crank`` package), so it resolves to
    ``site-packages/tt_crank/tt-metal`` for a pip-installed wheel. Absent in an
    editable/source checkout (nothing installs the payload there).
    """
    return Path(__file__).resolve().parent / "tt-metal"


def _vendored_tt_metal() -> Path:
    """Path to the tt-metal source tree populated by tt-mlir-ep at build time."""
    # python/tt_crank/_runtime_env.py -> repo root -> third_party/...
    return (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "tt-mlir"
        / "third_party"
        / "tt-metal"
        / "src"
        / "tt-metal"
    )


def tt_metal_home() -> Path:
    """
    Resolve the tt-metal tree.

    Resolution order:
      1. ``TT_METAL_RUNTIME_ROOT`` or ``TT_METAL_HOME``, if the user set either.
         Every one that is set must exist — a stale override is a hard error,
         since silently falling back would mask what the user pointed at.
      2. The tt-metal tree bundled inside an installed wheel
         (``tt_crank/tt-metal``). This is the normal pip-install path.
      3. Otherwise the tt-mlir-ep submodule checkout
         ``third_party/tt-mlir/third_party/tt-metal/src/tt-metal/`` (editable dev).
    """
    resolved = None
    for var in ("TT_METAL_RUNTIME_ROOT", "TT_METAL_HOME"):
        value = os.getenv(var)
        if value is None:
            continue
        if not Path(value).is_dir():
            raise FileNotFoundError(
                f"{var} is set to {value!r}, but that directory does not exist."
            )
        resolved = resolved or Path(value)
    if resolved is not None:
        return resolved

    for root in (_wheel_tt_metal(), _vendored_tt_metal()):
        if root.is_dir():
            return root
    raise RuntimeError(
        f"tt-metal runtime tree not found (looked in {_wheel_tt_metal()} and "
        f"{_vendored_tt_metal()}); build at least once so tt-mlir-ep populates "
        "the submodule, or set TT_METAL_RUNTIME_ROOT explicitly."
    )


def setup_tt_metal_home() -> None:
    """
    Ensure ``TT_METAL_HOME`` and ``TT_METAL_RUNTIME_ROOT`` point at a valid
    tt-metal tree, leaving whatever the user already set untouched.

    Mirrors the behavior of ``python/tt_crank/torch/__init__.py`` so that
    importing either module first leaves the environment in the same state.
    """
    # The tracy CLI reads TT_METAL_HOME, not TT_METAL_RUNTIME_ROOT, so both.
    root = str(tt_metal_home())
    os.environ.setdefault("TT_METAL_HOME", root)
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", root)
