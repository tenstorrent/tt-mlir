# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
try:
    import ttrt
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "Error: runtime python tests require ttrt to built and installed. Please run `cmake --build build`"
    )
import ttrt.runtime
from ttrt.common.api import API
from .utils import Helper
import pytest


@pytest.fixture(autouse=True, scope="module")
def initialize():
    API.initialize_apis()
    ttrt.runtime.set_current_device_runtime(ttrt.runtime.DeviceRuntime.TTNN)


@pytest.fixture(scope="module")
def helper():
    helper = Helper()
    yield helper
