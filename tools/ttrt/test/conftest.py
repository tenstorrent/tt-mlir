# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from util import *
import shutil


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    # save previous ttrt-artifacts directory
    if os.path.exists("ttrt-artifacts"):
        try:
            os.rename("ttrt-artifacts", "save-artifacts")
        except Exception as e:
            print(f"An error occurred while renaming the directory: {e}")

    directory_name = "ttrt-results"
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
        except Exception as e:
            print(f"An error occurred while saving artifacts: {e}")

    yield


def pytest_runtest_teardown(item, nextitem):
    assert (
        check_results(f"ttrt-results/{item.name}.json") == 0
    ), f"one of more tests failed in={item.name}"

    # Remove ttrt-artifacts directory and all of its content and restore previous one
    if os.path.exists("ttrt-artifacts") and os.path.exists("save-artifacts"):
        try:
            shutil.rmtree("ttrt-artifacts")
            os.rename("save-artifacts", "ttrt-artifacts")
        except Exception as e:
            print(f"An error occurred while restoring artifacts: {e}")
