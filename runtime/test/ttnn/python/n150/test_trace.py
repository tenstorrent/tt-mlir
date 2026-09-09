# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
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
    assert_pcc,
    get_torch_output_container,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/trace/Output"
)


def _read_device_output(runner, device_output):
    """Read a held device output tensor into a fresh torch container."""
    output_host = ttrt.runtime.to_host(device_output, untilize=True, blocking=True)[0]
    output_torch = get_torch_output_container(runner.program)
    ttrt.runtime.memcpy(output_torch.data_ptr(), output_host)
    return output_torch


@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_recapture_slot_reuse_arena(helper: Helper, request, trace_region_size):
    """
    Regression test for trace output corruption caused by a stale-trace recapture
    reallocating its persistent I/O slots.

    Since trace is always writing to same addresses during replay (these addresses
    are allocated during capture and immediately freed), any persistent allocation
    that happens after a trace has been captured is not safe since it can be
    overwritten by the trace's replay (if it is allocated in the same region of
    memory as the trace's intermediates).

    This test targets a particular wrong-result scenario in which, recaptured trace
    allocated its output slot in the region of memory used by another trace's
    intermediates.

    We can define the following terminology:
    - "safe" hole - is a gap in the memory space that was freed at some point and
      not used by any of the traces during replay.
    - "unsafe" hole - is a gap in the memory space that was freed but it IS USED
      by some trace replay (zone(s) where the trace's intermediates were captured).

    Test is constructed such that we force any re-allocations during trace
    recapture to be allocated in the "unsafe" hole:
    1. Allocate a dummy large tensor which will occupy large portion of the
    bottom space in DRAM.
    2. Push graph A's params and capture it. All allocations will be above the
    dummy large tensor allocation.
    3. Free the large tensor - this will leave a large unallocated gap at the bottom
    of the DRAM space.
    4. Capture graph B - the graph is constructed such that it will leave only
    one "unsafe" hole after the capture.
    5. Recapture A - anything allocated during recapture will go to the "unsafe"
    zone. Save the A's output tensor (which will be on device).
    6. Replay graph B - anything allocated by the graph A's recapture will get
    overwritten by this replay.
    7. Read back the saved graph A's output - it should pass the accuracy check.

    """
    a_binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_no_consteval.mlir.tmp.ttnn"
    )
    b_binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "large_neg_no_consteval.mlir.tmp.ttnn"
    )
    for path in (a_binary_path, b_binary_path):
        assert os.path.exists(path), f"Binary file not found: {path}"
    helper.initialize(request.node.name, a_binary_path)
    helper.check_constraints()

    a_config = ProgramTestConfig(
        name="matmul_multiply",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Traced graph whose recapture must reuse its slots",
    )
    a_runner = ProgramTestRunner(a_config, helper.binary, 0)

    b_binary = Binary(Logger(), FileManager(Logger()), b_binary_path)
    b_binary.check_version()
    b_binary.check_system_desc(helper.query)
    b_config = ProgramTestConfig(
        name="large_neg",
        expected_num_inputs=1,
        compute_golden=lambda inputs: -inputs[0],
        description="Traced graph whose replay rewrites the arena",
    )
    b_runner = ProgramTestRunner(b_config, b_binary, 0)

    debug_stats = ttrt.runtime.DebugStats.get()
    debug_stats.clear()

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:
        # 1. Large allocation first - it must be the very first device allocation so that
        # freeing it later opens a region below everything else.
        large_tensor_shape = [4140, 4096]
        large_tensor_stride = [4096, 1]
        large_tensor_layout = ttrt.runtime.test.get_dram_interleaved_row_major_layout(
            Binary.Program.to_data_type(torch.bfloat16)
        )
        large_tensor = ttrt.runtime.create_empty_tensor(
            device, large_tensor_layout, large_tensor_shape, large_tensor_stride, 2
        )

        # A's parameters are uploaded to device here - above the large tensor allocation,
        # like everything else from now on. B's input is host-side only.
        a_inputs, a_golden, _a_inputs_torch = a_runner.get_inputs_and_golden(device)
        b_inputs, b_golden, _b_inputs_torch = b_runner.get_inputs_and_golden(device)

        # 2. Capture A above the large tensor.
        a_output = a_runner.submit_program(device, a_inputs)
        assert_pcc(_read_device_output(a_runner, a_output), a_golden)
        assert debug_stats.get_stat("TraceCacheMiss") == 1
        assert debug_stats.get_stat("CapturedTrace") == 1

        # 3. Free the large tensor - open the arena.
        ttrt.runtime.deallocate_tensor(large_tensor, force=True)

        # 4. Capture B in the arena. A's trace is now stale.
        b_runner.run_program_and_compare_golden(device, b_inputs, b_golden)
        assert debug_stats.get_stat("TraceCacheMiss") == 2
        assert debug_stats.get_stat("CapturedTrace") == 2

        # 5. Rerun A: stale trace, gets recaptured. Slots must be reused.
        a_output_recaptured = a_runner.submit_program(device, a_inputs)
        assert debug_stats.get_stat("TraceStaleRecapture") == 1
        assert_pcc(_read_device_output(a_runner, a_output_recaptured), a_golden)

        # 6. Replay B (rewrites the arena), then re-read A's output. A
        # reallocated output slot would live inside the arena and now hold
        # B's intermediate data.
        b_runner.run_program_and_compare_golden(device, b_inputs, b_golden)
        assert debug_stats.get_stat("ExecutedTrace") == 1
        assert_pcc(_read_device_output(a_runner, a_output_recaptured), a_golden)

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


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

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_memory_overwrite_multi_graph(helper: Helper, request, trace_region_size):
    """
    This test verifies that the two traced graphs do not overwrite each other's memory.
    Device allocations, after a trace is captured, are (in general) not safe - there are no guarantees that the
    newly allocated memory does not overlap with memory used by the previously captured trace. If the overlap
    happens then the trace replay may corrupt the newly allocated memory (e.g. by writing its intermediates to it).

    We handle this case in runtime by tracking all of the captures - whenever we have a new capture, we bump
    trace cache generation id. Then, when we want to replay a cached trace, we check the generation id of the cached trace
    against the current generation id of the cache. In case of mismatch, we re-capture the trace to ensure that
    we won't overlap with any of the new allocations that happened since the last time we captured the trace.
    The cache entry and its persistent input/output slots stay in place across a re-capture; only the
    device-side trace is replaced. That is what keeps a re-capture from moving any buffer, and therefore
    from invalidating the other graph's trace and triggering an endless re-capture cycle.
    """
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    first_bin_config = ProgramTestConfig(
        name="first_graph",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Graph whose trace replay can corrupt victim memory",
    )
    first_bin_runner = ProgramTestRunner(first_bin_config, helper.binary, 0)

    victim_binary = Binary(Logger(), FileManager(Logger()), binary_path)
    victim_config = ProgramTestConfig(
        name="victim",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Graph whose parameters can get corrupted by trace replay",
    )
    victim_runner = ProgramTestRunner(victim_config, victim_binary, 0)

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:

        # Execute & capture the first graph.
        (
            pressure_inputs,
            pressure_golden,
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

        # The first graph should have been recaptured once, and exactly once: a
        # recapture reuses its persistent slots, so it cannot invalidate the
        # victim graph's trace and force it to be recaptured in turn.
        assert debug_stats.get_stat("TraceStaleRecapture") == 1
        # Only one miss: the initial capture of the victim graph. A stale
        # recapture is not a cache miss - the entry (and its slots) stay in the
        # cache and only the device-side trace is replaced.
        assert debug_stats.get_stat("TraceCacheMiss") == 1
        # We should hit the cache for both graphs each time we run the loop, except for the first recapture.
        assert debug_stats.get_stat("ExecutedTrace") == 2 * loop_count - 1

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()


@pytest.mark.parametrize("trace_region_size", [0, 80000])
def test_trace_recapture_uses_fresh_inputs(helper: Helper, request, trace_region_size):
    """
    Verifies that a re-capture consumes freshly supplied inputs and publishes its
    output slots. The first graph's trace is made stale by capturing a second
    graph (which bumps the trace cache generation), then the first graph is run
    with inputs it has never seen: a re-capture that failed to refresh the input
    slots, or failed to publish the output slots, would produce a stale result
    and fail the golden check.
    """
    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "matmul_multiply_consteval.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    first_config = ProgramTestConfig(
        name="first_graph",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Graph whose stale trace gets re-captured",
    )
    first_runner = ProgramTestRunner(first_config, helper.binary, 0)

    second_binary = Binary(Logger(), FileManager(Logger()), binary_path)
    second_config = ProgramTestConfig(
        name="second_graph",
        expected_num_inputs=3,
        compute_golden=lambda inputs: ((inputs[0] @ inputs[1]) * inputs[2]),
        description="Graph whose capture makes the first graph's trace stale",
    )
    second_runner = ProgramTestRunner(second_config, second_binary, 0)

    debug_stats = ttrt.runtime.DebugStats.get()

    with DeviceContext(
        mesh_shape=[1, 1],
        enable_program_cache=True,
        trace_region_size=trace_region_size,
    ) as device:
        # Capture the first graph's trace.
        inputs, golden, _ = first_runner.get_inputs_and_golden(device)
        first_runner.run_program_and_compare_golden(device, inputs, golden)

        # Capture the second graph's trace; this bumps the cache generation and
        # makes the first graph's trace stale.
        (
            second_inputs,
            second_golden,
            _second_inputs_torch,
        ) = second_runner.get_inputs_and_golden(device)
        second_runner.run_program_and_compare_golden(
            device, second_inputs, second_golden
        )

        # Run the first graph with brand-new inputs. This takes the re-capture
        # path, so the golden check fails unless the re-capture refreshed the
        # input slots and published the output slots.
        (
            fresh_inputs,
            fresh_golden,
            _fresh_inputs_torch,
        ) = first_runner.get_inputs_and_golden(device)
        first_runner.run_program_and_compare_golden(device, fresh_inputs, fresh_golden)

        assert debug_stats.get_stat("TraceStaleRecapture") == 1
        # Two misses: the initial capture of each graph. The re-capture is not a
        # miss - the entry and its slots stay in the cache.
        assert debug_stats.get_stat("TraceCacheMiss") == 2

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

    assert_pcc(output_torch, golden, threshold=0.98)
    print(
        f"{request.node.name} Executing {num_loops} loops time elapsed: {end_time - start_time} ms"
    )

    ttrt.runtime.DebugStats.get().clear()
    helper.teardown()
