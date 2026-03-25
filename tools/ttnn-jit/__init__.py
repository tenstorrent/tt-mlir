# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import subprocess
import sys


class _TtmlirRedirector(importlib.abc.MetaPathFinder):
    """Redirect bare ``ttmlir.*`` imports to ``ttnn_jit.ttmlir.*``.

    The MLIR Python bindings .so files are compiled with
    ``MLIR_PYTHON_PACKAGE_PREFIX=ttmlir.`` and hardcode C-level imports such as
    ``import_("ttmlir._mlir_libs._mlir.ir")``.  When the ttmlir tree is bundled
    inside the ``ttnn_jit`` wheel, those imports must resolve to the
    already-loaded ``ttnn_jit.ttmlir.*`` modules instead of picking up a second
    copy from the build directory on ``PYTHONPATH``.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname == "ttmlir" or fullname.startswith("ttmlir."):
            return importlib.machinery.ModuleSpec(fullname, _TtmlirLoader())
        return None


class _TtmlirLoader(importlib.abc.Loader):
    def create_module(self, spec):
        redirected = "ttnn_jit." + spec.name
        if redirected in sys.modules:
            return sys.modules[redirected]
        return importlib.import_module(redirected)

    def exec_module(self, module):
        pass


if not any(isinstance(f, _TtmlirRedirector) for f in sys.meta_path):
    sys.meta_path.insert(0, _TtmlirRedirector())

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
