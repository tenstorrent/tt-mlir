# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import torch.nn.functional as F
import ttrt
import ttrt.runtime
from ttrt.common.util import *
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    assert_pcc,
    get_torch_output_container,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/trace/Output"
)


@pytest.mark.parametrize("num_loops", [5])
def test_trace_matmul_multiply_no_consteval(helper: Helper, request, num_loops):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_no_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="matmul_multiply",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Matmul multiply trace test",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1], enable_program_cache=True, trace_region_size=80000
    ) as device:

        for i in range(num_loops):
            # First execute, should be a trace cache miss
            # Subsequent executes should be trace cache hit and execute trace
            inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(
                device
            )
            test_runner.run_program_and_compare_golden(
                device, inputs_runtime_with_layout, golden
            )
            assert debug_stats.get_stat("TraceCacheMiss") == 1
            assert debug_stats.get_stat("CapturedTrace") == 1
            assert debug_stats.get_stat("ExecutedTrace") == i

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("num_loops", [5])
def test_trace_matmul_multiply_with_consteval(helper: Helper, request, num_loops):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()
    test_config = ProgramTestConfig(
        name="matmul_multiply",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Matmul multiply trace test",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)
    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1], enable_program_cache=True, trace_region_size=80000
    ) as device:

        inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(device)

        for i in range(num_loops):
            # First execute, should be a trace cache miss and consteval cache miss
            # Subsequent executes should be consteval and trace cache hit
            test_runner.run_program_and_compare_golden(
                device,
                inputs_runtime_with_layout,
                golden,
            )
            assert debug_stats.get_stat("TraceCacheMiss") == 1
            assert debug_stats.get_stat("CapturedTrace") == 1
            assert debug_stats.get_stat("ExecutedTrace") == i
            assert debug_stats.get_stat("ConstEvalCacheMiss") == 1
            assert debug_stats.get_stat("ConstEvalCacheHit") == i

        ttrt.runtime.DebugStats.get().clear()

        inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(device)

        for i in range(num_loops):
            # First execute should be a consteval cache miss because we've updated the inputs
            # Subsequent executes should be consteval cache hits
            # Trace cache should not be affected
            test_runner.run_program_and_compare_golden(
                device,
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


def mnist_linear_logits_golden(inputs):
    [input_tensor, weight1, bias1, weight2, bias2] = inputs
    # First linear layer: input @ weight1 + bias1
    # Shape: (1, 784) @ (784, 256) -> (1, 256)
    hidden = torch.matmul(input_tensor, weight1)
    hidden = torch.add(hidden, bias1)

    # ReLU activation
    hidden = torch.relu(hidden)

    # Second linear layer: hidden @ weight2 + bias2
    # Shape: (1, 256) @ (256, 10) -> (1, 10)
    output = torch.matmul(hidden, weight2)
    output = torch.add(output, bias2)

    return output


@pytest.mark.parametrize("num_loops", [16])
def test_mnist_linear_logits(helper: Helper, request, num_loops):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "mnist_linear_logits.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()
    test_config = ProgramTestConfig(
        name="mnist_linear_logits",
        expected_num_inputs=5,
        compute_golden=mnist_linear_logits_golden,
        description="mnist linear logits trace test",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    debug_stats = ttrt.runtime.DebugStats.get()

    output_torch = get_torch_output_container(test_runner.program)

    with DeviceContext(
        mesh_shape=[1, 1], enable_program_cache=True, trace_region_size=80000
    ) as device:

        inputs_runtime_with_layout, golden = test_runner.get_inputs_and_golden(device)
        # Warm up the device
        test_runner.run_program_and_compare_golden(
            device, inputs_runtime_with_layout, golden
        )
        assert debug_stats.get_stat("TraceCacheMiss") == 1
        assert debug_stats.get_stat("CapturedTrace") == 1
        assert debug_stats.get_stat("ExecutedTrace") == 0

        start_time = time.perf_counter() * 1000
        for i in range(num_loops - 1):
            output = test_runner.run_program(device, inputs_runtime_with_layout)
            assert debug_stats.get_stat("TraceCacheMiss") == 1
            assert debug_stats.get_stat("CapturedTrace") == 1
            assert debug_stats.get_stat("ExecutedTrace") == i + 1
        end_time = time.perf_counter() * 1000
        ttrt.runtime.memcpy(output_torch.data_ptr(), output)

    assert_pcc(output_torch, golden)
    print(
        f"{request.node.name} Executing {num_loops} loops time elapsed: {end_time - start_time} ms"
    )

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()
