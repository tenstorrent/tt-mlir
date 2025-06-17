# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    assert_pcc,
    get_torch_inputs,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/trace/Output"
)


def get_inputs_and_golden(device, helper, program, program_index):
    inputs_torch = get_torch_inputs(program)
    inputs_runtime = [
        get_runtime_tensor_from_torch(torch_input) for torch_input in inputs_torch
    ]
    input_layouts = [
        ttrt.runtime.get_layout(
            executable=helper.binary.fbb, program_index=program_index, input_index=i
        )
        for i in range(len(inputs_runtime))
    ]
    inputs_runtime_with_layout = [
        ttrt.runtime.to_layout(rt_input, device, layout, True)
        for rt_input, layout in zip(inputs_runtime, input_layouts)
    ]

    golden = (inputs_torch[0] @ inputs_torch[1]) + inputs_torch[2]

    return inputs_runtime_with_layout, golden


def run_program_and_compare_golden(
    device, helper, program, program_index, inputs, golden
):

    output_torch = get_torch_output_container(program)

    output = ttrt.runtime.submit(device, helper.binary.fbb, program_index, inputs)[0]

    output = ttrt.runtime.to_host(output, untilize=True)[0]
    ttrt.runtime.memcpy(output_torch.data_ptr(), output)
    assert_pcc(output_torch, golden)


@pytest.mark.parametrize("num_loops", [5])
def test_trace_matmul_add_no_consteval(helper: Helper, request, num_loops):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_add_no_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program_index = 0
    program: Binary.Program = helper.binary.get_program(program_index)
    assert not program.is_private()

    assert program.num_inputs() == 3

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1], enable_program_cache=True, trace_region_size=16384
    ) as device:

        for i in range(num_loops):
            inputs_runtime_with_layout, golden = get_inputs_and_golden(
                device, helper, program, program_index
            )
            # First execute, should be a trace cache miss
            # Subsequent executes should be trace cache hit and execute trace
            run_program_and_compare_golden(
                device,
                helper,
                program,
                program_index,
                inputs_runtime_with_layout,
                golden,
            )
            assert debug_stats.get_stat("TraceCacheMiss") == 1
            assert debug_stats.get_stat("CapturedTrace") == 1
            assert debug_stats.get_stat("ExecutedTrace") == i

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("num_loops", [5])
def test_trace_matmul_add_with_consteval(helper: Helper, request, num_loops):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_add_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    program_index = 0
    program: Binary.Program = helper.binary.get_program(program_index)
    assert not program.is_private()

    assert program.num_inputs() == 3

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1], enable_program_cache=True, trace_region_size=16384
    ) as device:

        inputs_runtime_with_layout, golden = get_inputs_and_golden(
            device, helper, program, program_index
        )

        for i in range(num_loops):
            # First execute, should be a trace cache miss and consteval cache miss
            # Subsequent executes should be consteval and trace cache hit
            run_program_and_compare_golden(
                device,
                helper,
                program,
                program_index,
                inputs_runtime_with_layout,
                golden,
            )
            assert debug_stats.get_stat("TraceCacheMiss") == 1
            assert debug_stats.get_stat("CapturedTrace") == 1
            assert debug_stats.get_stat("ExecutedTrace") == i
            assert debug_stats.get_stat("ConstEvalCacheMiss") == 1
            assert debug_stats.get_stat("ConstEvalCacheHit") == i

        ttrt.runtime.DebugStats.get().clear()

        inputs_runtime_with_layout, golden = get_inputs_and_golden(
            device, helper, program, program_index
        )

        for i in range(num_loops):
            # First execute should be a consteval cache miss because we've updated the inputs
            # Subsequent executes should be consteval cache hits
            # Trace cache should not be affected
            run_program_and_compare_golden(
                device,
                helper,
                program,
                program_index,
                inputs_runtime_with_layout,
                golden,
            )
            assert debug_stats.get_stat("TraceCacheMiss") == 0
            assert debug_stats.get_stat("CapturedTrace") == 0
            assert debug_stats.get_stat("ExecutedTrace") == i + 1
            assert debug_stats.get_stat("ConstEvalCacheMiss") == 1
            assert debug_stats.get_stat("ConstEvalCacheHit") == i

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()
