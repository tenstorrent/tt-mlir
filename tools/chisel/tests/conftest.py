# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--binary-path",
        action="append",
        default=[],
        help="Path(s) to .ttnn binary files for walk order tests. Can be repeated.",
    )


def _collect_binary_paths(config):
    paths = list(config.getoption("binary_path"))
    env = os.environ.get("CHISEL_TEST_BINARY_PATHS", "")
    if env:
        paths.extend(p for p in env.split(":") if p)
    return paths


def pytest_generate_tests(metafunc):
    if "binary_path" in metafunc.fixturenames:
        paths = _collect_binary_paths(metafunc.config)
        if not paths:
            metafunc.parametrize("binary_path", [None])
        else:
            metafunc.parametrize("binary_path", paths)
