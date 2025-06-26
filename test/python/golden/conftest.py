# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttrt
from functools import reduce
import operator


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
    parser.addoption(
        "--allow-subset-mesh",
        action="store_true",
        help="Enable running tests whose mesh shapes are a subset of the current device",
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


def filter_valid_mesh_shape(system_desc, params, allow_subset_mesh=False):
    num_chips = reduce(operator.mul, params.get("mesh_shape", [1]), 1)
    if allow_subset_mesh:
        return num_chips <= len(system_desc["chip_desc_indices"])
    else:
        return num_chips == len(system_desc["chip_desc_indices"])


def pytest_collection_modifyitems(config, items):
    valid_items = []
    deselected = []
    system_desc = ttrt.binary.fbb_as_dict(
        ttrt.binary.load_system_desc_from_path(config.option.sys_desc)
    )["system_desc"]

    for item in items:
        # Only check parameterized tests
        if hasattr(item, "callspec"):
            params = item.callspec.params
            if not filter_valid_mesh_shape(
                system_desc, params, allow_subset_mesh=config.option.allow_subset_mesh
            ):
                # Deselect the test case
                deselected.append(item)
                continue
        valid_items.append(item)

    # Update the items list (collected tests)
    items[:] = valid_items

    # Report deselected items to pytest
    if deselected:
        config.hook.pytest_deselected(items=deselected)
