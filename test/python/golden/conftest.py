# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttrt.common.api import API as ttrt


def pytest_addoption(parser):
    parser.addoption(
        "--path",
        action="store",
        default=".",
        help="Path to store test artifacts (e.g. flatbuffers and .mlir files)",
    )
    parser.addoption(
        "--sys-desc",
        action="store",
        default="ttrt-artifacts/system_desc.ttsys",
        help="Path to system descriptor",
    )


def pytest_configure(config):
    # Register custom markers
    config.addinivalue_line(
        "markers",
        "skip_target(target): skip test if operation is not supported on the specified target",
    )


def pytest_runtest_setup(item):
    # Skip tests marked with skip_target when the current target matches
    for marker in item.iter_markers(name="skip_target"):
        target_to_skip = marker.args[0]
        # Get the current target from the test's parametrization
        current_target = None
        for param in item.callspec.params.items():
            if param[0] == "target":
                current_target = param[1]
                break

        if current_target == target_to_skip:
            pytest.skip(f"Operation not supported on {target_to_skip} target")
