# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime front door for the tt-crank execution provider.

``register()`` points ONNX Runtime at the plugin library and, like
``tt_crank.torch``, unregisters it again at interpreter exit so the device mesh
closes before C-runtime teardown. Registering through ONNX Runtime directly
works too, but then the caller must unregister before exit.
"""

import atexit
import os
from pathlib import Path

import onnxruntime

from tt_crank._runtime_env import setup_tt_metal_home

EP_NAME = "TTKurblaExecutionProvider"
_REGISTRATION_NAME = "tt_crank"
_registered = False


def library_path() -> Path:
    """The plugin library: wheel payload if installed, else the tt-mlir build tree
    (BUILD_DIR overrides <tt-mlir>/build)."""
    wheel_lib = Path(__file__).resolve().parents[1] / "lib" / "libtt_crank_ort.so"
    if wheel_lib.exists():
        return wheel_lib
    tt_mlir_root = Path(__file__).resolve().parents[4]
    build_dir = Path(os.environ.get("BUILD_DIR", tt_mlir_root / "build"))
    return build_dir / "tt-crank" / "src" / "onnx" / "libtt_crank_ort.so"


def register() -> None:
    """Register the EP library with ONNX Runtime (idempotent)."""
    global _registered
    if _registered:
        return
    path = library_path()
    if not path.exists():
        raise FileNotFoundError(f"tt-crank ONNX EP library not found: {path}")
    setup_tt_metal_home()
    onnxruntime.register_execution_provider_library(_REGISTRATION_NAME, str(path))
    atexit.register(
        onnxruntime.unregister_execution_provider_library, _REGISTRATION_NAME
    )
    _registered = True
