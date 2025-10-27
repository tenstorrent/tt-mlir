# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simple library tweaks module used to move `TT_METAL_RUNTIME_ROOT` to point to the
mirrored TTMetal tree within the `ttrt` wheel. It is important that
`set_tt_metal_home()` is the _FIRST_ bit of code run in this `ttrt` module.
Thus, this file should only be included in `ttrt/__init__.py` and only run
there. This is a temporary fix, and will need to be cleaned up once TTMetal
drops `TT_METAL_RUNTIME_ROOT` functionality
"""
import importlib.util
import os


def get_ttrt_metal_home_path() -> str:
    """Finds the root of the mirrored TTMetal tree within the `ttrt` wheel"""
    package_name = "ttrt"
    spec = importlib.util.find_spec(package_name)
    package_path = os.path.dirname(spec.origin)
    tt_metal_home = f"{package_path}/runtime"
    return tt_metal_home


def set_tt_metal_home():
    """Sets the environment variable `TT_METAL_RUNTIME_ROOT` to point into the root
    mirrored TTMetal tree within the `ttrt` wheel.
    """
    os.environ["TT_METAL_RUNTIME_ROOT_EXTERNAL"] = os.environ.get(
        "TT_METAL_RUNTIME_ROOT", ""
    )
    os.environ["TT_METAL_RUNTIME_ROOT"] = get_ttrt_metal_home_path()
    os.environ["TT_METAL_HOME"] = os.environ["TT_METAL_RUNTIME_ROOT"]

    new_linker_path = f"{get_ttrt_metal_home_path()}/tests"
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if current_ld_library_path:
        updated_ld_library_path = f"{new_linker_path}:{current_ld_library_path}"
    else:
        updated_ld_library_path = new_linker_path
    os.environ["LD_LIBRARY_PATH"] = updated_ld_library_path
