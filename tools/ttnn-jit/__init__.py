# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess

import ttnn

# _build_metadata.py is generated only by wheel build.
try:
    from ttnn_jit._build_metadata import METAL_GIT_SHA
except ImportError:
    METAL_GIT_SHA = None


def _get_ttnn_pip_sha():
    """Extract SHA from ttnn pip package version string."""
    try:
        output = subprocess.check_output(["pip", "show", "ttnn"], text=True)
        if match := re.search(r"Version:.*\+g([a-f0-9]+)", output):
            return match.group(1)[:8]
    except (subprocess.CalledProcessError, AttributeError):
        pass
    return None


def _get_metal_home_sha():
    """Extract SHA from TT_METAL_HOME git repository."""
    metal_home = os.environ.get("TT_METAL_HOME")
    if not metal_home:
        raise RuntimeError("TT_METAL_HOME is not set")
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=metal_home,
            text=True,
        ).strip()
    except Exception:
        raise RuntimeError("Could not get TT_METAL_HOME git SHA")


def _verify_metal_compatibility():
    """Verify tt-metal compatibility via pip package or TT_METAL_HOME."""
    expected_sha = METAL_GIT_SHA[:8]

    # Try ttnn pip package first (wheel-based installation)
    if actual_sha := _get_ttnn_pip_sha():
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Incompatible ttnn version. Expected SHA {expected_sha}, got {actual_sha}"
            )
        return

    # Fall back to TT_METAL_HOME check (local dev environment)
    actual_sha = _get_metal_home_sha()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Incompatible tt-metal version. Expected SHA {expected_sha}, got {actual_sha}"
        )


if METAL_GIT_SHA is not None:
    _verify_metal_compatibility()


from ttnn_jit.api import (
    jit,
)

__all__ = [
    "jit",
]
