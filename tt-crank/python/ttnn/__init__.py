# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTNN wrapper module.

This module wraps the original `ttnn` module from tt-metal so that we can set up
the environment correctly before delegating to the original module.
"""

import sys
from pathlib import Path

from tt_kurbla._runtime_env import setup_tt_metal_home
from tt_kurbla._wrapper import create_wrapper_redirector, proxy_import

setup_tt_metal_home()

_original_path = Path(__file__).parent / "_original"
if not _original_path.exists():
    raise RuntimeError(
        f"ttnn._original symlink missing at {_original_path}. "
        "Run a tt-kurbla build (./scripts/build) so the bundle-ttnn target "
        "creates it."
    )

# Install wrapper redirector.
sys.meta_path.insert(
    0,
    create_wrapper_redirector(
        "ttnn",
        original_path=_original_path,
        extensions={"_ttnn": "_ttnn.so"},
    ),
)

# Import original with proxy for circular import handling.
with proxy_import("ttnn"):
    from ttnn._original import *
