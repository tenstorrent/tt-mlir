# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *

from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    get_input_spec,
    create_multi_device_input,
    create_grouped_multi_device_input,
    create_replicated_multi_device_input,
    run_and_verify_output_shards,
)

MESH_SHAPE = [1, 8]
NUM_DEVICES = MESH_SHAPE[0] * MESH_SHAPE[1]
FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/llmbox/cpu_hoisting/Output"
)


def initialize_helper(helper: Helper, request):
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()


def test_cpu_hoisted_add_sharded(helper: Helper, request):
    """Test CPU-hoisted add with both inputs sharded across all eight chips.

    Every shard carries distinct data, so the hoisted function runs once per
    shard, concurrently, and each output shard must match its own sum.
    """
    assert ttrt.runtime.get_num_available_devices() == 8

    initialize_helper(helper, request)
    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_shards = create_multi_device_input(shape, dtype, MESH_SHAPE)
    input1, input1_shards = create_multi_device_input(shape, dtype, MESH_SHAPE)

    expected = [
        torch.add(input0_shards[i], input1_shards[i]) for i in range(NUM_DEVICES)
    ]

    debug_stats = ttrt.runtime.DebugStats.get()
    debug_stats.clear()

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    # Every shard's input tuple is distinct, so the hoisted function must
    # have executed once per shard.
    assert debug_stats.get_stat("CpuOpHoistedRuns") == NUM_DEVICES

    helper.teardown()


def test_cpu_hoisted_add_replicated_inputs(helper: Helper, request):
    """Test CPU-hoisted add with both inputs replicated across the mesh.

    All eight shards present the same host buffers, so the hoisted function
    sees one distinct input tuple and takes the run-once dedupe path; every
    output shard must equal the same sum.
    """
    assert ttrt.runtime.get_num_available_devices() == 8

    initialize_helper(helper, request)
    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_torch = create_replicated_multi_device_input(
        shape, dtype, MESH_SHAPE
    )
    input1, input1_torch = create_replicated_multi_device_input(
        shape, dtype, MESH_SHAPE
    )

    expected = [torch.add(input0_torch, input1_torch) for _ in range(NUM_DEVICES)]

    debug_stats = ttrt.runtime.DebugStats.get()
    debug_stats.clear()

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    assert debug_stats.get_stat("CpuOpHoistedRuns") == 1

    helper.teardown()


def test_cpu_hoisted_add_grouped_shards(helper: Helper, request):
    """Test CPU-hoisted add where shards duplicate a small set of buffers.

    Both inputs cycle through two distinct host buffers, so the eight shards
    form exactly two dedupe groups. The hoisted function must run once per
    group, and each output shard must match the sum for its own group --
    alternating expected values catch any shard-to-group mapping mixups.
    """
    assert ttrt.runtime.get_num_available_devices() == 8

    initialize_helper(helper, request)
    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    num_distinct = 2
    input0, input0_shards = create_grouped_multi_device_input(
        shape, dtype, MESH_SHAPE, num_distinct
    )
    input1, input1_shards = create_grouped_multi_device_input(
        shape, dtype, MESH_SHAPE, num_distinct
    )

    expected = [
        torch.add(input0_shards[i], input1_shards[i]) for i in range(NUM_DEVICES)
    ]

    debug_stats = ttrt.runtime.DebugStats.get()
    debug_stats.clear()

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    assert debug_stats.get_stat("CpuOpHoistedRuns") == num_distinct

    helper.teardown()


def test_cpu_hoisted_add_replicated_and_sharded(helper: Helper, request):
    """Test CPU-hoisted add with one sharded and one replicated input.

    The sharded input makes every per-shard input tuple distinct, so the
    hoisted function runs once per shard (concurrently), while the replicated
    input contributes the same buffer to each tuple.
    """
    assert ttrt.runtime.get_num_available_devices() == 8

    initialize_helper(helper, request)
    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_shards = create_multi_device_input(shape, dtype, MESH_SHAPE)
    input1, input1_torch = create_replicated_multi_device_input(
        shape, dtype, MESH_SHAPE
    )

    expected = [torch.add(input0_shards[i], input1_torch) for i in range(NUM_DEVICES)]

    debug_stats = ttrt.runtime.DebugStats.get()
    debug_stats.clear()

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    assert debug_stats.get_stat("CpuOpHoistedRuns") == NUM_DEVICES

    helper.teardown()
