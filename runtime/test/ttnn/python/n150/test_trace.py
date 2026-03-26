# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest
import os
import torch
import ttrt
import ttrt.runtime
from ..utils import (
    DeviceContext,
    ProgramTestRunner,
    assert_pcc,
    get_torch_output_container,
    get_flatbuffer_base_path,
    load_binary,
)

FLATBUFFER_BASE_PATH = get_flatbuffer_base_path("Runtime", "TTNN", "n150", "trace")


@pytest.mark.parametrize("num_loops", [5])
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_matmul_multiply_no_consteval(num_loops, trace_region_size):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_no_consteval.mlir.tmp.ttnn"
    )
    binary = load_binary(binary_path)

    test_runner = ProgramTestRunner(
        binary,
        0,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
    )

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        for i in range(num_loops):
            # First execute, should be a trace cache miss
            # Subsequent executes should be trace cache hit and execute trace
            inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
                device
            )
            test_runner.run_program_and_compare_golden(
                device, inputs_runtime_with_layout, golden
            )
            assert debug_stats.get_stat("TraceCacheMiss") == 1
            assert debug_stats.get_stat("CapturedTrace") == 1
            assert debug_stats.get_stat("ExecutedTrace") == i


@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_memory_overwrite_multi_graph(trace_region_size):
    """
    This test verifies that the two traced graphs do not overwrite each other's memory.
    Device allocations, after a trace is captured, are (in general) not safe - there are no guarantees that the
    newly allocated memory does not overlap with memory used by the previously captured trace. If the overlap
    happens then the trace replay may corrupt the newly allocated memory (e.g. by writing its intermediates to it).

    We handle this case in runtime by tracking all of the captures - whenever we have a new capture, we bump
    trace cache generation id. Then, when we want to replay a cached trace, we check the generation id of the cached trace
    against the current generation id of the cache. In case of mismatch, we re-capture & re-cache the trace to ensure that
    we won't overlap with any of the new allocations that happened since the last time we captured the trace.
    """
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_consteval.mlir.tmp.ttnn"
    )
    first_binary = load_binary(binary_path)

    first_bin_runner = ProgramTestRunner(
        first_binary,
        0,
    )

    victim_binary = load_binary(binary_path)
    victim_runner = ProgramTestRunner(
        victim_binary,
        0,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
    )

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        # Execute & capture the first graph.
        (
            pressure_inputs,
            _,
            _pressure_inputs_torch,
        ) = first_bin_runner.get_inputs_and_golden(device)
        first_bin_runner.run_program(device, pressure_inputs)

        # Run the first graph few times - confirm that we're actually executing the trace.
        for i in range(3):
            first_bin_runner.run_program(device, pressure_inputs)
            assert debug_stats.get_stat("TraceCacheMiss") == 1
            assert debug_stats.get_stat("ExecutedTrace") == i + 1

        # Reset the stats.
        debug_stats.clear()

        # Now enters the second graph.
        # These inputs are allocated after we've captured the first trace.
        # Hence, they are vulnerable to getting overwritten if we would naively
        # replay the first trace.
        (
            victim_inputs,
            victim_golden,
            _victim_inputs_torch,
        ) = victim_runner.get_inputs_and_golden(device)

        # Capture victim trace, golden check should pass.
        victim_runner.run_program_and_compare_golden(
            device, victim_inputs, victim_golden
        )

        # Replay both traces in a loop, verifying that the outputs of the victim graph
        # stay correct - i.e. we do NOT overwrite any of the input tensors of the victim
        # graph.
        loop_count = 32
        for i in range(loop_count):
            first_bin_runner.run_program(device, pressure_inputs)
            victim_runner.run_program_and_compare_golden(
                device, victim_inputs, victim_golden
            )

        # The first graph should have been recaptured once.
        assert debug_stats.get_stat("TraceStaleRecapture") == 1
        # We have two misses, one for the initial capture of the victim graph and one for the re-capture of the first graph.
        assert debug_stats.get_stat("TraceCacheMiss") == 2
        # We should hit the cache for both graphs each time we run the loop, except for the first recapture.
        assert debug_stats.get_stat("ExecutedTrace") == 2 * loop_count - 1


@pytest.mark.parametrize("num_loops", [5])
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_matmul_multiply_with_consteval(num_loops, trace_region_size):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_consteval.mlir.tmp.ttnn"
    )
    binary = load_binary(binary_path)

    test_runner = ProgramTestRunner(
        binary,
        0,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
    )
    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

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

        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

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
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_mnist_linear_logits(request, num_loops, trace_region_size):
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "mnist_linear_logits.mlir.tmp.ttnn"
    )
    binary = load_binary(binary_path)

    test_runner = ProgramTestRunner(
        binary,
        0,
        compute_golden=mnist_linear_logits_golden,
    )

    debug_stats = ttrt.runtime.DebugStats.get()

    output_torch = get_torch_output_container(test_runner.program)

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )
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
