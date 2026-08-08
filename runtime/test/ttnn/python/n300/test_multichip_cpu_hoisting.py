# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from .constants import FLATBUFFER_BASE_PATH

from ..utils import (
    Helper,
    DeviceContext,
    get_input_spec,
    create_multi_device_input,
    create_replicated_multi_device_input,
    create_single_device_input,
    run_and_verify_output_shards,
)

MESH_SHAPE = [1, 2]


def test_cpu_hoisted_add(helper: Helper, request):
    """Test CPU-hoisted add with both inputs sharded."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_shards = create_multi_device_input(shape, dtype, MESH_SHAPE)
    input1, input1_shards = create_multi_device_input(shape, dtype, MESH_SHAPE)

    expected = [torch.add(input0_shards[i], input1_shards[i]) for i in range(2)]

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    helper.teardown()


def test_cpu_hoisted_add_mixed_inputs(helper: Helper, request):
    """Test CPU-hoisted add with one sharded and one non-sharded input."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_shards = create_multi_device_input(shape, dtype, MESH_SHAPE)
    input1, input1_torch = create_single_device_input(shape, dtype)

    # Non-sharded input is broadcast to all shards.
    expected = [torch.add(input0_shards[i], input1_torch) for i in range(2)]

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    helper.teardown()


def test_cpu_hoisted_add_replicated_inputs(helper: Helper, request):
    """Test CPU-hoisted add with both inputs replicated across the mesh.

    Both inputs present the same host buffer at every mesh coordinate, so the
    hoisted function sees one distinct input tuple and takes the run-once
    dedupe path; every output shard must equal the same sum.
    """
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_torch = create_replicated_multi_device_input(
        shape, dtype, MESH_SHAPE
    )
    input1, input1_torch = create_replicated_multi_device_input(
        shape, dtype, MESH_SHAPE
    )

    expected = [torch.add(input0_torch, input1_torch) for _ in range(2)]

    debug_stats = ttrt.runtime.DebugStats.get()
    debug_stats.clear()

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    # Both shards present the same input buffers, so the hoisted function
    # must have executed exactly once.
    assert debug_stats.get_stat("CpuOpHoistedRuns") == 1

    helper.teardown()


def test_cpu_hoisted_add_replicated_and_sharded(helper: Helper, request):
    """Test CPU-hoisted add with one sharded and one replicated input.

    The sharded input makes every per-shard input tuple distinct, so the
    hoisted function runs once per shard (concurrently), while the replicated
    input contributes the same buffer to each tuple.
    """
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program = helper.binary.get_program(0)
    shape, dtype = get_input_spec(program, 0)

    input0, input0_shards = create_multi_device_input(shape, dtype, MESH_SHAPE)
    input1, input1_torch = create_replicated_multi_device_input(
        shape, dtype, MESH_SHAPE
    )

    expected = [torch.add(input0_shards[i], input1_torch) for i in range(2)]

    debug_stats = ttrt.runtime.DebugStats.get()
    debug_stats.clear()

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        run_and_verify_output_shards(helper, mesh_device, [input0, input1], expected)

    # The sharded input makes each shard's input tuple distinct, so the
    # hoisted function must have executed once per shard.
    assert debug_stats.get_stat("CpuOpHoistedRuns") == 2

    helper.teardown()
