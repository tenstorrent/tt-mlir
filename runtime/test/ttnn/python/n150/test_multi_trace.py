# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-trace test: exercises the scenario where two different trace graphs
are executed on the same device. This is the key scenario that triggers
the bug where one trace's replay overwrites another trace's input slots
or intermediates on DRAM.

The test interleaves execution of two different programs (each with their
own trace) and verifies that outputs remain correct across iterations.
"""

import pytest
import torch
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


def _create_runners(helper, request):
    """Create both program runners, compatible with both old and new ProgramTestRunner API."""
    binary_path_a = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_no_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path_a), f"Binary not found: {binary_path_a}"

    binary_path_b = os.path.join(
        FLATBUFFER_BASE_PATH, "add_relu_no_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path_b), f"Binary not found: {binary_path_b}"

    helper.initialize(request.node.name, binary_path_a)
    helper.check_constraints()
    binary_a = helper.binary

    binary_b = Binary(Logger(), FileManager(Logger()), binary_path_b)
    binary_b.check_version()

    config_a = ProgramTestConfig(
        name="matmul_multiply",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Program A: matmul + multiply",
    )

    config_b = ProgramTestConfig(
        name="add_relu",
        expected_num_inputs=2,
        compute_golden=lambda inputs: torch.relu(inputs[0] + inputs[1]),
        description="Program B: add + relu",
    )

    runner_a = ProgramTestRunner(config_a, binary_a, 0)
    runner_b = ProgramTestRunner(config_b, binary_b, 0)

    return runner_a, runner_b


@pytest.mark.parametrize("num_loops", [5])
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_multi_trace_interleaved(
    helper: Helper, request, num_loops, trace_region_size
):
    """
    Execute two different traced programs on the same device in interleaved
    fashion. This exercises the scenario where:
      1. Program A captures its trace (intermediates at some DRAM addresses)
      2. Program B captures its trace (intermediates may overlap with A's)
      3. Program A replays its trace (writes to recorded addresses, may corrupt B)
      4. Program B replays its trace (writes to recorded addresses, may corrupt A)
    """
    runner_a, runner_b = _create_runners(helper, request)

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        for i in range(num_loops):
            # Execute program A with fresh inputs each iteration
            inputs_a, golden_a, _ = runner_a.get_inputs_and_golden(device)
            runner_a.run_program_and_compare_golden(device, inputs_a, golden_a)

            # Execute program B with fresh inputs each iteration
            inputs_b, golden_b, _ = runner_b.get_inputs_and_golden(device)
            runner_b.run_program_and_compare_golden(device, inputs_b, golden_b)

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("num_loops", [5])
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_multi_trace_sequential_then_replay(
    helper: Helper, request, num_loops, trace_region_size
):
    """
    Capture both traces first, then replay them multiple times.
    This is the most direct reproduction of the collision:
      - Iteration 1: A captures, B captures
      - Iterations 2+: A replays, B replays (collision if intermediates overlap)
    """
    runner_a, runner_b = _create_runners(helper, request)

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        # --- Phase 1: Capture both traces (first execution of each) ---
        inputs_a, golden_a, torch_a = runner_a.get_inputs_and_golden(device)
        runner_a.run_program_and_compare_golden(device, inputs_a, golden_a)

        inputs_b, golden_b, torch_b = runner_b.get_inputs_and_golden(device)
        runner_b.run_program_and_compare_golden(device, inputs_b, golden_b)

        # --- Phase 2: Replay both traces multiple times with new inputs ---
        # This is where the collision manifests: replaying trace A writes to
        # DRAM addresses that were recorded during capture, which may now
        # overlap with trace B's intermediates (and vice versa).
        for i in range(num_loops):
            inputs_a, golden_a, torch_a = runner_a.get_inputs_and_golden(device)
            runner_a.run_program_and_compare_golden(device, inputs_a, golden_a)

            inputs_b, golden_b, torch_b = runner_b.get_inputs_and_golden(device)
            runner_b.run_program_and_compare_golden(device, inputs_b, golden_b)

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("num_loops", [5])
@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_multi_trace_same_inputs_replay(
    helper: Helper, request, num_loops, trace_region_size
):
    """
    Same as above but keeps inputs constant across replays. This isolates
    the memory corruption from input version tracking: if the output changes
    despite same inputs, it's a memory corruption issue.
    """
    runner_a, runner_b = _create_runners(helper, request)

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        # Generate inputs once
        inputs_a, golden_a, torch_a = runner_a.get_inputs_and_golden(device)
        inputs_b, golden_b, torch_b = runner_b.get_inputs_and_golden(device)

        for i in range(num_loops):
            # Execute A then B with the same inputs every iteration
            # If outputs degrade over iterations, it's memory corruption
            runner_a.run_program_and_compare_golden(device, inputs_a, golden_a)
            runner_b.run_program_and_compare_golden(device, inputs_b, golden_b)

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()
