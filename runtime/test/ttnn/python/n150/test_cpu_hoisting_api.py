# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    Storage,
    assert_pcc,
    get_runtime_tensor_from_torch,
    get_to_layout_inputs,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/cpu_hoisting/Output"
)

SHAPE = (32, 32)

DTYPE = torch.bfloat16
CPU_OP_DTYPE = torch.float32


def is_debug_enabled():
    debug_stats = str(ttrt.runtime.DebugStats.get())
    return debug_stats != "DebugStats Disabled"


def host_tensor_to_torch(rt_tensor):
    shape = rt_tensor.get_shape()
    dtype = ttrt_datatype_to_torch_dtype(rt_tensor.get_dtype())
    torch_tensor = torch.zeros(shape, dtype=dtype)
    ttrt.runtime.memcpy(torch_tensor.data_ptr(), rt_tensor)
    return torch_tensor


def test_invoke_cpu_op(helper: Helper, request):
    """Re-invoke a CPU-hoisted dylib out-of-band via invoke_cpu_op.

    A debug hook captures the live program/op context for the CpuOp during
    normal program execution, then within the callback invoke_cpu_op is
    called with caller-supplied host inputs that differ from the program's
    pool inputs. The output is compared against a torch reference to confirm
    the dylib really ran on the supplied tensors.
    """
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "cpu_hoisted_add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program_inputs = [
        torch.randn(SHAPE, dtype=DTYPE),
        torch.randn(SHAPE, dtype=DTYPE),
    ]

    # Different from program_inputs on purpose: proves the dylib actually
    # ran on the caller-supplied tensors and not on the pool tensors.
    custom_inputs_torch = [
        torch.randn(SHAPE, dtype=CPU_OP_DTYPE),
        torch.randn(SHAPE, dtype=CPU_OP_DTYPE),
    ]
    expected_output = custom_inputs_torch[0] + custom_inputs_torch[1]

    captured = {"output": None, "cpu_op_count": 0}

    def post_op(binary, program_context, op_context):
        debug_str = ttrt.runtime.get_op_debug_str(op_context)
        if "cpu_hoist_call" not in debug_str:
            return

        captured["cpu_op_count"] += 1

        # Owned storage so invoke_cpu_op can read the host buffers without
        # the caller having to keep custom_inputs_torch pinned.
        runtime_inputs = [
            get_runtime_tensor_from_torch(t, storage=Storage.Owned)
            for t in custom_inputs_torch
        ]

        rt_outputs = ttrt.runtime.invoke_cpu_op(
            program_context, op_context, runtime_inputs
        )
        assert len(rt_outputs) == 1, f"expected 1 output, got {len(rt_outputs)}"
        captured["output"] = host_tensor_to_torch(rt_outputs[0])

    hooks = ttrt.runtime.DebugHooks.get(post_op=post_op)
    assert hooks is not None, "Failed to register hooks"

    with DeviceContext(mesh_shape=[1, 1]) as device:
        runtime_inputs = [get_runtime_tensor_from_torch(t) for t in program_inputs]
        runtime_inputs_with_layout = get_to_layout_inputs(
            device, runtime_inputs, helper.binary, 0
        )
        ttrt.runtime.submit(device, helper.binary.fbb, 0, runtime_inputs_with_layout)

    ttrt.runtime.unregister_hooks()

    assert (
        captured["cpu_op_count"] == 1
    ), f"post_op saw {captured['cpu_op_count']} CpuOp(s), expected 1"
    assert captured["output"] is not None
    assert_pcc(captured["output"], expected_output)

    helper.teardown()
