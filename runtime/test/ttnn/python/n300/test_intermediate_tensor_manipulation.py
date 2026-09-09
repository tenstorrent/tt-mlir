# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from .constants import FLATBUFFER_BASE_PATH

from ..utils import (
    Helper,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    get_torch_output_container,
    get_to_layout_inputs,
    get_runtime_tensor_from_torch,
)

MESH_SHAPE = [1, 2]


def get_torch_tensor(tensor: ttrt.runtime.Tensor):
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    if rt_dtype is not ttrt.runtime.DataType.Float32:
        raise ValueError(f"Unsupported data type: {rt_dtype}")
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=torch.float32)
    return torch_tensor.reshape(shape)


def update_multi_device_tensor(
    program_context, tensor_ref, dst_shards, per_shard_torch_tensors, alive_refs
):
    """Replace a multi-device pool entry with per-shard torch data.

    The torch tensors are kept alive via the caller-provided ``alive_refs`` list
    because the runtime borrows their underlying buffers.
    """
    contiguous_shards = [t.contiguous() for t in per_shard_torch_tensors]
    alive_refs.extend(contiguous_shards)

    template_shard = dst_shards[0]
    shape = template_shard.get_shape()
    stride = template_shard.get_stride()
    dtype = template_shard.get_dtype()
    item_size = contiguous_shards[0].element_size()

    multi_device_tensor = ttrt.runtime.create_multi_device_host_tensor(
        [t.data_ptr() for t in contiguous_shards],
        shape,
        stride,
        item_size,
        dtype,
        {},
        MESH_SHAPE,
    )
    ttrt.runtime.update_tensor_in_pool(program_context, tensor_ref, multi_device_tensor)


def identity(binary, program_context, op_context):
    return


def is_callback_enabled():
    debug_stats = str(ttrt.runtime.DebugStats.get())
    return debug_stats != "DebugStats Disabled"


def make_linear_postop(expected_per_shard, replacement_per_shard, alive_refs):
    """Build a postop callback that, on the ``ttnn.linear`` op:

    - retrieves the per-shard intermediate tensor from the pool;
    - asserts each shard equals ``expected_per_shard[i]`` (a scalar);
    - updates the same pool entry twice, where the second replacement uses
      ``replacement_per_shard[i]``.
    """

    def postop(binary, program_context, op_context):
        debug_op_str = ttrt.runtime.get_op_debug_str(op_context)
        if "ttnn.linear" not in debug_op_str:
            return

        output_refs = ttrt.runtime.get_op_output_refs(op_context)
        if not output_refs:
            return
        tensor_ref: ttrt.runtime.TensorRef = output_refs[0]

        tensor = ttrt.runtime.retrieve_tensor_from_pool(program_context, tensor_ref)
        if tensor is None:
            return

        shards = ttrt.runtime.to_host(tensor, untilize=True)

        assert shards is not None

        assert len(shards) == len(
            expected_per_shard
        ), f"Expected {len(expected_per_shard)} shards, got {len(shards)}"

        per_shard_torch_shape = None
        per_shard_torch_dtype = None
        for i, shard in enumerate(shards):
            t = get_torch_tensor(shard)
            per_shard_torch_shape = t.shape
            per_shard_torch_dtype = t.dtype
            expected = torch.full_like(t, expected_per_shard[i])
            assert torch.allclose(t, expected), (
                f"Shard {i}: expected all {expected_per_shard[i]}, "
                f"got mean {t.mean().item()}"
            )

        replacement_shards = [
            torch.full(
                per_shard_torch_shape,
                replacement_per_shard[i],
                dtype=per_shard_torch_dtype,
            )
            for i in range(len(shards))
        ]
        first_replacement_shards = [torch.zeros_like(t) for t in replacement_shards]
        update_multi_device_tensor(
            program_context,
            tensor_ref,
            shards,
            first_replacement_shards,
            alive_refs,
        )
        update_multi_device_tensor(
            program_context, tensor_ref, shards, replacement_shards, alive_refs
        )

    return postop


def test_replicated_intermidate_tensor_manipulation(helper: Helper, request):
    """Linear+add with all four inputs replicated across both chips.

    Each chip sees identical data, so both retrieved shards must match the
    same scalar. After replacing the intermediate with all-ones on both
    chips the final output is all-twos.
    """
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear_replicated.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="linear_replicated", expected_num_inputs=4, compute_golden=None
    )
    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    # Replicated host inputs (all ones).
    # Per-chip linear: ones(10x10) @ ones(10x10) + ones(10) = 11.
    inputs_torch = [
        torch.ones((10, 10), dtype=torch.float32),  # input
        torch.ones((10, 10), dtype=torch.float32),  # weight
        torch.ones((10,), dtype=torch.float32),  # bias
        torch.ones((10, 10), dtype=torch.float32),  # extra
    ]

    alive_refs = []
    hooks = ttrt.runtime.DebugHooks.get(
        identity,
        make_linear_postop(
            expected_per_shard=[11.0, 11.0],
            replacement_per_shard=[1.0, 1.0],
            alive_refs=alive_refs,
        ),
    )
    if hooks is None:
        return

    with DeviceContext(mesh_shape=MESH_SHAPE) as device:
        runtime_inputs = [get_runtime_tensor_from_torch(t) for t in inputs_torch]
        runtime_inputs_with_layout = get_to_layout_inputs(
            device, runtime_inputs, helper.binary, 0
        )
        output_torch = get_torch_output_container(test_runner.program)
        output = test_runner.run_program(device, runtime_inputs_with_layout)
        ttrt.runtime.memcpy(output_torch.data_ptr(), output)
        # After update: per-chip linear = 1, plus replicated extra = 1 -> 2.
        assert torch.allclose(output_torch, torch.full_like(output_torch, 2.0))

    ttrt.runtime.unregister_hooks()
    helper.teardown()


def test_sharded_intermidate_tensor_manipulation(helper: Helper, request):
    """Linear+add where weight, bias, and extra are split across the mesh.

    Each chip sees a different slice of weight/bias/extra, so the two
    intermediate shards must hold different values. The test replaces them
    with chip-specific scalars and verifies the aggregated output reflects
    those replacements.
    """
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear_sharded.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="linear_sharded", expected_num_inputs=4, compute_golden=None
    )
    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    # Build host-shape inputs whose left/right halves seed the two chips.
    #   chip 0 sees the [:, :10] / [:10] half, chip 1 sees the [:, 10:] / [10:] half.
    input_t = torch.ones((10, 10), dtype=torch.float32)  # replicated

    weight_t = torch.empty((10, 20), dtype=torch.float32)
    weight_t[:, :10] = 1.0  # chip 0 weight
    weight_t[:, 10:] = 2.0  # chip 1 weight

    bias_t = torch.empty((20,), dtype=torch.float32)
    bias_t[:10] = 1.0  # chip 0 bias
    bias_t[10:] = 2.0  # chip 1 bias

    extra_t = torch.empty((10, 20), dtype=torch.float32)
    extra_t[:, :10] = 3.0  # chip 0 extra
    extra_t[:, 10:] = 4.0  # chip 1 extra

    inputs_torch = [input_t, weight_t, bias_t, extra_t]

    # Per-chip linear:
    #   chip 0: ones @ ones + ones = 10 + 1 = 11
    #   chip 1: ones @ twos + twos = 20 + 2 = 22
    expected_per_shard = [11.0, 22.0]
    # After update -> add per-chip extra -> aggregate along dim 1:
    #   chip 0: 5 + 3 = 8        -> output[:, :10]  = 8
    #   chip 1: 7 + 4 = 11       -> output[:, 10:] = 11
    replacement_per_shard = [5.0, 7.0]

    alive_refs = []
    hooks = ttrt.runtime.DebugHooks.get(
        identity,
        make_linear_postop(
            expected_per_shard=expected_per_shard,
            replacement_per_shard=replacement_per_shard,
            alive_refs=alive_refs,
        ),
    )
    if hooks is None:
        return

    with DeviceContext(mesh_shape=MESH_SHAPE) as device:
        runtime_inputs = [get_runtime_tensor_from_torch(t) for t in inputs_torch]
        runtime_inputs_with_layout = get_to_layout_inputs(
            device, runtime_inputs, helper.binary, 0
        )
        output_torch = get_torch_output_container(test_runner.program)
        output = test_runner.run_program(device, runtime_inputs_with_layout)
        ttrt.runtime.memcpy(output_torch.data_ptr(), output)

        expected_out = torch.empty_like(output_torch)
        expected_out[:, :10] = 8.0
        expected_out[:, 10:] = 11.0
        assert torch.allclose(output_torch, expected_out)

    ttrt.runtime.unregister_hooks()
    helper.teardown()
