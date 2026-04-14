# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--binary",
        action="append",
        default=[],
        help="Path to a .ttnn file or directory to search recursively. Can be repeated.",
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
    paths = []
    for p in config.getoption("binary"):
        if os.path.isdir(p):
            paths.extend(_find_ttnn_files(p))
        else:
            paths.append(p)
    return paths


def pytest_generate_tests(metafunc):
    if "binary_path" in metafunc.fixturenames:
        paths = _collect_binary_paths(metafunc.config)
        if not paths:
            metafunc.parametrize("binary_path", [None])
        else:
            metafunc.parametrize("binary_path", paths)
