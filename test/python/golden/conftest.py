# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import ttrt
import platform
from functools import reduce
import operator

ALL_BACKENDS = set(["ttnn", "ttmetal", "ttnn-standalone"])
ALL_SYSTEMS = set(["n150", "n300", "llmbox", "tg", "p150", "p300"])


def is_x86_machine():
    machine = platform.machine().lower()
    return machine in ["x86_64", "amd64", "i386", "i686", "x86"]


x86_only = pytest.mark.skipif(
    not is_x86_machine(),
    reason=f"Test requires x86 architecture, but running on {platform.machine()}",
)


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
        "--require-exact-mesh",
        action="store_true",
        help="Require exact mesh shape match with the current device (default allows subset)",
    )
    parser.addoption(
        "--require-opmodel",
        action="store_true",
        help="Require tests to run only if build has opmodel enabled",
    )


def get_board_id(system_desc) -> str:
    arch = system_desc["chip_descs"][0]["arch"]
    num_chips = len(system_desc["chip_desc_indices"])

    match arch, num_chips:
        case "Blackhole", 1:
            return "p150"
        case "Blackhole", 2:
            return "p300"
        case "Wormhole_b0", 1:
            return "n150"
        case "Wormhole_b0", 2:
            return "n300"
        case _:
            raise ValueError(f"Unknown architecture: {arch}")


def filter_valid_mesh_shape(system_desc, params, require_exact_mesh=False):
    num_chips = reduce(operator.mul, params.get("mesh_shape", [1]), 1)
    num_physical_chips = len(system_desc["chip_desc_indices"])
    if require_exact_mesh:
        return num_chips == num_physical_chips
    else:
        return num_chips <= num_physical_chips



def pytest_collection_modifyitems(config, items):
    valid_items = []
    deselected = []
    system_desc = ttrt.binary.fbb_as_dict(
        ttrt.binary.load_system_desc_from_path(config.option.sys_desc)
    )["system_desc"]

    skip_opmodel = pytest.mark.skip(reason="Test requires --require-opmodel flag")
    require_opmodel = config.getoption("--require-opmodel")

    for item in items:
        # Skip optimizer tests if opmodel flag is missing
        if not require_opmodel and "optimizer" in str(item.fspath):
            item.add_marker(skip_opmodel)

        # Only check parameterized tests
        if hasattr(item, "callspec"):
            params = item.callspec.params
            if not filter_valid_mesh_shape(
                system_desc, params, require_exact_mesh=config.option.require_exact_mesh
            ):
                # Deselect the test case
                deselected.append(item)
                continue
        valid_items.append(item)

        # Skip specific target / system combinations

        # Fetch the current target of this test, if any
        current_target = None
        for param in item.callspec.params.items():
            if param[0] == "target":
                current_target = param[1]
                break

        for marker in item.iter_markers(name="skip_config"):
            for platform_config in marker.args:

                # All of the operations we need to do on these are set membership based
                platform_config = set(platform_config)

                reason = marker.kwargs.get("reason", "")

                # Verify this is a valid configuration
                if not platform_config <= ALL_BACKENDS.union(ALL_SYSTEMS):
                    outliers = platform_config - ALL_BACKENDS.union(ALL_SYSTEMS)
                    raise ValueError(
                        f"Invalid skip config: {platform_config}, invalid entries: {outliers}. Please ensure that all entries in the config are members of {ALL_SYSTEMS} or {ALL_BACKENDS}"
                    )

                board_id = get_board_id(system_desc)

                if platform_config <= set([current_target, board_id]):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Operation not supported on following platform/target combination: {platform_config}. {reason}"
                        )
                    )

    # Update the items list (collected tests)
    items[:] = valid_items

    # Report deselected items to pytest
    if deselected:
        config.hook.pytest_deselected(items=deselected)
