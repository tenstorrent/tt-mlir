# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tracy profiler wrapper module.

This module wraps the original `tracy` module from tt-metal so that we can set up
the environment correctly before delegating to the original module.
"""

import os
import sys
from pathlib import Path

import tt_kurbla
from tt_kurbla._runtime_env import setup_tt_metal_home
from tt_kurbla._wrapper import create_wrapper_redirector, proxy_import

setup_tt_metal_home()

# `tracy` uses `TT_METAL_HOME` environment variable.
os.environ["TT_METAL_HOME"] = os.environ["TT_METAL_RUNTIME_ROOT"]

_original_path = Path(__file__).parent / "_original"
if not _original_path.exists():
    raise RuntimeError(
        f"tracy._original symlink missing at {_original_path}. "
        "Run a tt-kurbla build (./scripts/build) so the bundle-tracy target "
        "creates it."
    )

# Install wrapper redirector.
sys.meta_path.insert(
    0,
    create_wrapper_redirector(
        "tracy",
        original_path=_original_path,
        skip_submodules=("__main__",),
    ),
)

# Import original with proxy for circular import handling.
with proxy_import("tracy"):
    from tracy._original import *

# Override profiler binary path to point to the correct location.
# In wheel: tt_kurbla/bin/
# In dev: build/third_party/tt-mlir-install/bin/
_wheel_bin_dir = Path(tt_kurbla.__file__).parent / "bin"
_dev_bin_dir = (
    Path(__file__).parent / ".." / ".." / "build" / "third_party" / "tt-mlir-install" / "bin"
)

if _wheel_bin_dir.exists() and (_wheel_bin_dir / "capture-release").exists():
    PROFILER_BIN_DIR = _wheel_bin_dir
else:
    PROFILER_BIN_DIR = _dev_bin_dir

# Override profiler artifacts dir to a local .tracy_artifacts folder (by default).
# User can still override it with command line argument when running `tracy`.
PROFILER_ARTIFACTS_DIR = Path(os.getcwd()) / ".tracy_artifacts"
PROFILER_LOGS_DIR = common.generate_logs_folder(PROFILER_ARTIFACTS_DIR)
PROFILER_OUTPUT_DIR = common.generate_reports_folder(PROFILER_ARTIFACTS_DIR)
os.environ["TT_METAL_PROFILER_DIR"] = str(PROFILER_ARTIFACTS_DIR.resolve())
