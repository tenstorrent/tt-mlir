# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import ttrt.binary

from chisel.ops import IRModule


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


@pytest.fixture
def binary(binary_path):
    return ttrt.binary.load_binary_from_path(binary_path)


@pytest.fixture
def ir_module(binary):
    mlir_json = ttrt.binary.mlir_as_dict(binary)
    functions = [binary.get_program_name(i) for i in range(binary.get_num_programs())]
    return IRModule(mlir_source=mlir_json["source"], functions=functions)


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
