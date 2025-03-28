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
        default="",
        help="Path to system descriptor",
    )


# TODO: figure out how to neatly pass this to all tests
@pytest.fixture(autouse=True)
def sys_desc():
    """
    Before any tests are run, query the system so the descriptor is always up to date
    """
    ttrt.initialize_apis()
    args = {"--save-artifacts": True}
    ttrt.Query(args=args)()
