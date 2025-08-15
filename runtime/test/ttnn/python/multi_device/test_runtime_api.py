# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import ttrt
import ttrt.runtime
from ttrt.common.util import *


@pytest.mark.parametrize(
    "mesh_shape",
    [[1, 1], [1, 2]],
)
@pytest.mark.parametrize(
    "mesh_offset",
    [[0, 0]],
)
@pytest.mark.parametrize(
    "num_hw_cqs",
    [1, 2],
)
@pytest.mark.parametrize(
    "enable_program_cache",
    [False, True],
)
@pytest.mark.parametrize(
    "l1_small_size",
    [0, 1024, 2048],
)
@pytest.mark.parametrize(
    "trace_region_size",
    [0, 1024, 2048],
)
def test_open_mesh_device(
    helper,
    mesh_shape,
    mesh_offset,
    num_hw_cqs,
    enable_program_cache,
    l1_small_size,
    trace_region_size,
):
    num_devices = ttrt.runtime.get_num_available_devices()
    assert num_devices == 2, f"Expected 2 devices, got {num_devices}"
    options = ttrt.runtime.MeshDeviceOptions()
    options.mesh_shape = mesh_shape
    options.mesh_offset = mesh_offset
    options.num_hw_cqs = num_hw_cqs
    options.enable_program_cache = enable_program_cache
    options.l1_small_size = l1_small_size
    options.trace_region_size = trace_region_size
    device = ttrt.runtime.open_mesh_device(options)
    device_ids = device.get_device_ids()
    assert device_ids == list(range(math.prod(mesh_shape)))
    assert device.get_mesh_shape() == mesh_shape
    assert device.get_num_hw_cqs() == num_hw_cqs
    assert device.is_program_cache_enabled() == enable_program_cache
    assert device.get_l1_small_size() == l1_small_size
    assert device.get_trace_region_size() == trace_region_size
    assert device.get_num_dram_channels() != 0
    assert device.get_dram_size_per_channel() != 0
    assert device.get_l1_size_per_core() != 0
    ttrt.runtime.close_mesh_device(device)
