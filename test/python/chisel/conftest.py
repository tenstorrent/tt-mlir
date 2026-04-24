# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import _ttmlir_runtime as tt_runtime


def pytest_addoption(parser):
    # Guard: --binary may already be registered when this conftest is loaded
    # alongside tools/chisel/tests/conftest.py in a combined pytest session.
    try:
        parser.addoption(
            "--binary",
            help="Path to a .ttnn file or directory to search recursively.",
        )
    except ValueError:
        pass


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


@pytest.fixture(scope="function")
def device():
    """Open a 1x1 TTNN mesh device for chisel device-execution tests."""
    tt_runtime.runtime.set_current_device_runtime(
        tt_runtime.runtime.DeviceRuntime.TTNN
    )
    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    mesh_options.mesh_shape = (1, 1)
    dev = tt_runtime.runtime.open_mesh_device(mesh_options)
    yield dev
    tt_runtime.runtime.close_mesh_device(dev)
