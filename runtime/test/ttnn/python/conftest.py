# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
try:
    import ttrt
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "Error: runtime python tests require ttrt to built and installed. Please run `cmake --build build`"
    )
import torch
import ttrt.runtime
from ttrt.common.api import API
import pytest


@pytest.fixture(autouse=True)
def initialize():
    API.initialize_apis()
    ttrt.runtime.set_current_device_runtime(ttrt.runtime.DeviceRuntime.TTNN)
    torch.manual_seed(27)
    ttrt.runtime.DebugStats.get().clear()
    yield
    ttrt.runtime.DebugStats.get().clear()
