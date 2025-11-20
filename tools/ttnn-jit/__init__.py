# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from importlib import import_module
import os

# _build_metadata.py is generated only by wheel build.
try:
    from ttnn_jit._build_metadata import METAL_GIT_SHA
except ImportError:
    METAL_GIT_SHA = None

if METAL_GIT_SHA is not None:
    # Runtime check against {TT_METAL_HOME} git SHA to ensure compatibility.
    metal_home = os.environ.get("TT_METAL_HOME", None)
    if metal_home is None:
        raise RuntimeError("TT_METAL_HOME is not set")
    try:
        import subprocess
        cwd_metal_sha = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=metal_home,
            text=True,
        ).strip()
    except Exception:
        raise RuntimeError("Could not get TT_METAL_HOME git SHA")
    
    if cwd_metal_sha != METAL_GIT_SHA[:8]:
        raise RuntimeError(f"Incompatible tt-metal version detected. Expected {METAL_GIT_SHA[:8]}, got {cwd_metal_sha}. Please use a compatible tt-metal version.")


from ttnn_jit.api import (
    jit,
)
__all__ = [
    "jit",
]
