# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import Optional, Tuple

import pytest
import _ttmlir_runtime as tt_runtime

from chisel.ops import IRModule

from utils import json_string_as_dict


# The chisel runtime registers DebugHooks against a single mesh device, so we
# only ever keep one device open at a time. Tests that want a different mesh
# shape (e.g. multichip n300) reuse the same fixture; on a mesh shape change
# the cached device is torn down and a fresh one is opened with the right
# fabric config. This mirrors `test/python/golden/conftest.py` but trimmed to
# the bits chisel needs.
_current_device = None
_current_mesh_shape: Optional[Tuple[int, ...]] = None


def _open_device(mesh_shape: Tuple[int, ...]):
    global _current_device, _current_mesh_shape

    if _current_device is not None and _current_mesh_shape == mesh_shape:
        return _current_device

    if _current_device is not None:
        tt_runtime.runtime.close_mesh_device(_current_device)
        tt_runtime.runtime.set_fabric_config(tt_runtime.runtime.FabricConfig.DISABLED)
        _current_device = None
        _current_mesh_shape = None

    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    mesh_options.mesh_shape = mesh_shape
    tt_runtime.runtime.set_current_device_runtime(tt_runtime.runtime.DeviceRuntime.TTNN)
    if math.prod(mesh_shape) > 1:
        tt_runtime.runtime.set_fabric_config(tt_runtime.runtime.FabricConfig.FABRIC_1D)
    _current_device = tt_runtime.runtime.open_mesh_device(mesh_options)
    _current_mesh_shape = mesh_shape
    return _current_device


def _close_device():
    global _current_device, _current_mesh_shape
    if _current_device is not None:
        tt_runtime.runtime.close_mesh_device(_current_device)
        tt_runtime.runtime.set_fabric_config(tt_runtime.runtime.FabricConfig.DISABLED)
        _current_device = None
        _current_mesh_shape = None


def _num_available_chips() -> Optional[int]:
    """Best-effort lookup of the number of physical chips on this host.

    Looks for a serialized system descriptor at ``SYSTEM_DESC_PATH`` (env)
    or the canonical ``ttrt-artifacts/system_desc.ttsys`` location. Returns
    None when no descriptor is reachable so callers can fall back to letting
    the runtime decide.
    """
    candidates = []
    env_path = os.environ.get("SYSTEM_DESC_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.append("ttrt-artifacts/system_desc.ttsys")

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            import json
            import re

            sys_desc_bin = tt_runtime.binary.load_system_desc_from_path(path)
            raw = sys_desc_bin.as_json()
            # Flatbuffers emit bare nan/inf which the stdlib json parser rejects.
            raw = re.sub(r"\bnan\b", "NaN", raw)
            raw = re.sub(r"\binf\b", "Infinity", raw)
            sys_desc = json.loads(raw).get("system_desc", {})
            return len(sys_desc.get("chip_desc_indices", []))
        except Exception:
            continue
    return None


@pytest.fixture(scope="function")
def device():
    return _open_device((1, 1))


@pytest.fixture(scope="function")
def multichip_device():
    # n300-style (1, 2) mesh; skip cleanly on hosts that can't satisfy it so
    # the same test file remains collectable on n150 / sim setups.
    required = 2
    num_chips = _num_available_chips()
    if num_chips is not None and num_chips < required:
        pytest.skip(
            f"multichip test requires at least {required} chips, "
            f"system has {num_chips}"
        )
    return _open_device((1, 2))


def pytest_sessionfinish(session):
    _close_device()


@pytest.fixture
def binary(binary_path):
    return tt_runtime.binary.load_binary_from_path(binary_path)


@pytest.fixture
def ir_module(binary):
    mlir_json = json_string_as_dict(binary.get_mlir_as_json())
    functions = [binary.get_program_name(i) for i in range(binary.get_num_programs())]
    return IRModule(mlir_source=mlir_json["source"], functions=functions)


def pytest_addoption(parser):
    parser.addoption(
        "--binary",
        help="Path to a .ttnn file or directory to search recursively.",
    )


def _find_ttnn_files(directory):
    """Recursively discover .ttnn files in a directory, sorted alphabetically."""
    found = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".ttnn"):
                found.append(os.path.join(root, f))
    found.sort()
    return found


def _collect_binary_paths(config):
    p = config.getoption("binary")
    if p is None:
        return []
    if os.path.isdir(p):
        return _find_ttnn_files(p)
    return [p]


def pytest_generate_tests(metafunc):
    if "binary_path" in metafunc.fixturenames:
        paths = _collect_binary_paths(metafunc.config)
        if not paths:
            binary_opt = metafunc.config.getoption("binary", default=None)
            if binary_opt is None:
                pytest.fail(
                    "No binary specified. Use --binary to provide a .ttnn file or directory."
                )
            else:
                pytest.fail(f"No .ttnn flatbuffers found under '{binary_opt}'.")
        metafunc.parametrize("binary_path", paths)
