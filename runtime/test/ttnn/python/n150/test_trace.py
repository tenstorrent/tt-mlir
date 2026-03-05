# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import random
from enum import Enum
import pytest
import os
import torch
import ttrt
import ttrt.runtime
from ttrt.common.util import Binary, FileManager, Logger
from ..utils import (
    TT_MLIR_HOME,
    Helper,
    DeviceContext,
    ProgramTestConfig,
    ProgramTestRunner,
    Storage,
    assert_pcc,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/trace/Output"
)


@pytest.mark.parametrize("num_loops", [5])
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_matmul_multiply_no_consteval(
    helper: Helper, request, num_loops, trace_region_size
):
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
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        for i in range(num_loops):
            # First execute captures directly. Subsequent executes replay.
            inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
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


@pytest.mark.parametrize("trace_region_size", [0])
def test_trace_memory_corruption_multi_graph(
    helper: Helper, request, trace_region_size
):
    # Load the same single-matmul binary as two separate Binary objects so
    # the runtime treats them as independent programs with separate traces.
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "single_matmul.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    pressure_config = ProgramTestConfig(
        name="pressure",
        expected_num_inputs=2,
        # No golden needed — we only care about the side-effect of trace replay
        compute_golden=None,
        description="Graph whose trace replay corrupts victim memory",
    )
    pressure_runner = ProgramTestRunner(pressure_config, helper.binary, 0)

    victim_binary = Binary(Logger(), FileManager(Logger()), binary_path)
    victim_config = ProgramTestConfig(
        name="victim",
        expected_num_inputs=2,
        compute_golden=lambda inputs: inputs[0] @ inputs[1],
        description="Graph whose parameters get corrupted by trace replay",
    )
    victim_runner = ProgramTestRunner(victim_config, victim_binary, 0)

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        # Capture pressure trace
        # Keep pressure_inputs_torch alive to prevent GC from freeing
        # the host memory backing the borrowed runtime tensors (issue #6254)
        pressure_inputs, _, pressure_inputs_torch = (
            pressure_runner.get_inputs_and_golden(device)
        )
        pressure_runner.run_program(device, pressure_inputs)

        # Allocate victim inputs AFTER pressure trace is captured,
        # so parameters land in memory the pressure trace may overwrite
        victim_inputs, victim_golden, victim_inputs_torch = (
            victim_runner.get_inputs_and_golden(device)
        )

        # Capture victim trace, golden check should pass
        victim_runner.run_program_and_compare_golden(
            device, victim_inputs, victim_golden
        )

        # Replay both traces in a loop — pressure replay may corrupt
        # victim's parameter memory, causing the victim golden check to fail
        for i in range(32):
            pressure_runner.run_program(device, pressure_inputs)
            victim_runner.run_program_and_compare_golden(
                device, victim_inputs, victim_golden
            )

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("trace_region_size", [0])
def test_trace_late_arriving_traced_graph(
    helper: Helper, request, trace_region_size
):
    """Pressure graph is fully captured and replaying BEFORE victim arrives.

    This is the "late-arriving graph" scenario: pressure's trace was captured
    without knowledge of victim's memory footprint. When victim allocates its
    slots, they may land at addresses that pressure's trace uses for
    intermediates. Pressure replay then corrupts victim's slots.
    """
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "single_matmul.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    pressure_config = ProgramTestConfig(
        name="pressure",
        expected_num_inputs=2,
        compute_golden=None,
        description="Graph fully captured before victim arrives",
    )
    pressure_runner = ProgramTestRunner(pressure_config, helper.binary, 0)

    victim_binary = Binary(Logger(), FileManager(Logger()), binary_path)
    victim_config = ProgramTestConfig(
        name="victim",
        expected_num_inputs=2,
        compute_golden=lambda inputs: inputs[0] @ inputs[1],
        description="Late-arriving graph whose slots may overlap pressure intermediates",
    )
    victim_runner = ProgramTestRunner(victim_config, victim_binary, 0)

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        # Fully capture pressure: warmup → capture → several replays
        pressure_inputs, _, pressure_inputs_torch = (
            pressure_runner.get_inputs_and_golden(device)
        )
        for _ in range(5):
            pressure_runner.run_program(device, pressure_inputs)

        # NOW victim arrives — pressure is already fully captured
        victim_inputs, victim_golden, victim_inputs_torch = (
            victim_runner.get_inputs_and_golden(device)
        )
        # Warmup + capture victim
        victim_runner.run_program(device, victim_inputs)
        victim_runner.run_program(device, victim_inputs)

        # Replay both in a loop — pressure replay may corrupt victim slots
        for _ in range(32):
            pressure_runner.run_program(device, pressure_inputs)
            victim_runner.run_program_and_compare_golden(
                device, victim_inputs, victim_golden
            )

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("num_loops", [5])
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_matmul_multiply_with_consteval(
    helper: Helper, request, num_loops, trace_region_size
):
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
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        for i in range(num_loops):
            # First execute captures directly. Subsequent executes replay.
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
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_mnist_linear_logits(helper: Helper, request, num_loops, trace_region_size):
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
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )
        # First call captures directly
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
