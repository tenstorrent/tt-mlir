# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from util import *


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    directory_name = "ttrt-results"
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")

    yield


def pytest_runtest_teardown(item, nextitem):
    assert (
        check_results(f"ttrt-results/{item.name}.json") == 0
    ), f"one of more tests failed in={item.name}"
