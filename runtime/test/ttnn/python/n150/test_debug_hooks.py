# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
    ProgramTestConfig,
    ProgramTestRunner,
    get_runtime_tensor_from_torch,
    get_to_layout_inputs,
)

FLATBUFFER_BASE_PATH = (
    f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/n150/tensor_manipulation/Output"
)


class CallbackTracker:
    """Helper class to track callback invocations."""

    def __init__(self):
        self.pre_execution_count = 0
        self.post_execution_count = 0
        self.pre_op_count = 0
        self.post_op_count = 0
        self.pre_op_locations = []
        self.post_op_locations = []
        self.execution_contexts = []

    def reset(self):
        """Reset all counters and tracked data."""
        self.pre_execution_count = 0
        self.post_execution_count = 0
        self.pre_op_count = 0
        self.post_op_count = 0
        self.pre_op_locations = []
        self.post_op_locations = []
        self.execution_contexts = []

    def pre_execution_callback(self, binary, program_context):
        """Called before program execution starts."""
        self.pre_execution_count += 1
        self.execution_contexts.append(("pre_exec", binary, program_context))

    def post_execution_callback(self, binary, program_context):
        """Called after program execution completes."""
        self.post_execution_count += 1
        self.execution_contexts.append(("post_exec", binary, program_context))

    def pre_op_callback(self, binary, program_context, op_context):
        """Called before each operation execution."""
        self.pre_op_count += 1
        # Extract operation location if available
        try:
            loc = op_context.get_loc_info()
            self.pre_op_locations.append(loc)
        except:
            self.pre_op_locations.append("unknown")

    def post_op_callback(self, binary, program_context, op_context):
        """Called after each operation execution."""
        self.post_op_count += 1
        # Extract operation location if available
        try:
            loc = op_context.get_loc_info()
            self.post_op_locations.append(loc)
        except:
            self.post_op_locations.append("unknown")


def is_debug_enabled():
    """Check if debug hooks are enabled in the runtime."""
    debug_stats = str(ttrt.runtime.DebugStats.get())
    return debug_stats != "DebugStats Disabled"


def test_all_four_callbacks(helper: Helper, request):
    """Test that all four callback hooks work correctly."""
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Test all four callback hooks",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)
    tracker = CallbackTracker()

    # Register all four callbacks
    hooks = ttrt.runtime.DebugHooks.get(
        pre_op=tracker.pre_op_callback,
        post_op=tracker.post_op_callback,
        pre_exec=tracker.pre_execution_callback,
        post_exec=tracker.post_execution_callback,
    )

    assert hooks is not None, "Failed to register hooks"

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        # Run the program
        output = test_runner.run_program(device, inputs_runtime_with_layout)

        # Verify all callbacks were called
        assert (
            tracker.pre_execution_count == 1
        ), f"Pre-execution callback called {tracker.pre_execution_count} times, expected 1"
        assert (
            tracker.post_execution_count == 1
        ), f"Post-execution callback called {tracker.post_execution_count} times, expected 1"
        assert (
            tracker.pre_op_count > 0
        ), f"Pre-op callback never called (expected at least 1 operation)"
        assert (
            tracker.post_op_count > 0
        ), f"Post-op callback never called (expected at least 1 operation)"
        assert (
            tracker.pre_op_count == tracker.post_op_count
        ), f"Pre-op count {tracker.pre_op_count} != Post-op count {tracker.post_op_count}"

        # Verify execution contexts were captured
        assert len(tracker.execution_contexts) == 2, "Expected 2 execution contexts"
        assert tracker.execution_contexts[0][0] == "pre_exec"
        assert tracker.execution_contexts[1][0] == "post_exec"

    # Cleanup
    ttrt.runtime.unregister_hooks()
    helper.teardown()


def test_selective_callbacks(helper: Helper, request):
    """Test registering only some callbacks (not all four)."""
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Test selective callback registration",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)
    tracker = CallbackTracker()

    # Register only pre_op and post_exec callbacks
    hooks = ttrt.runtime.DebugHooks.get(
        pre_op=tracker.pre_op_callback, post_exec=tracker.post_execution_callback
    )

    assert hooks is not None, "Failed to register hooks"

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        # Run the program
        output = test_runner.run_program(device, inputs_runtime_with_layout)

        # Verify only registered callbacks were called
        assert (
            tracker.pre_execution_count == 0
        ), "Pre-execution callback should not be called"
        assert (
            tracker.post_execution_count == 1
        ), "Post-execution callback should be called once"
        assert tracker.pre_op_count > 0, "Pre-op callback should be called"
        assert tracker.post_op_count == 0, "Post-op callback should not be called"

    # Cleanup
    ttrt.runtime.unregister_hooks()
    helper.teardown()


def test_unregister_hooks(helper: Helper, request):
    """Test that unregister_hooks properly clears all callbacks."""
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Test unregister hooks functionality",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)
    tracker = CallbackTracker()

    # Register all callbacks
    hooks = ttrt.runtime.DebugHooks.get(
        pre_op=tracker.pre_op_callback,
        post_op=tracker.post_op_callback,
        pre_exec=tracker.pre_execution_callback,
        post_exec=tracker.post_execution_callback,
    )

    assert hooks is not None, "Failed to register hooks"

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        # First run with callbacks
        output1 = test_runner.run_program(device, inputs_runtime_with_layout)

        # Verify callbacks were called
        first_pre_exec_count = tracker.pre_execution_count
        first_post_exec_count = tracker.post_execution_count
        first_pre_op_count = tracker.pre_op_count
        first_post_op_count = tracker.post_op_count

        assert first_pre_exec_count == 1
        assert first_post_exec_count == 1
        assert first_pre_op_count > 0
        assert first_post_op_count > 0

        # Unregister hooks
        ttrt.runtime.unregister_hooks()

        # Second run without callbacks
        output2 = test_runner.run_program(device, inputs_runtime_with_layout)

        # Verify callbacks were NOT called after unregister
        assert (
            tracker.pre_execution_count == first_pre_exec_count
        ), "Pre-execution callback called after unregister"
        assert (
            tracker.post_execution_count == first_post_exec_count
        ), "Post-execution callback called after unregister"
        assert (
            tracker.pre_op_count == first_pre_op_count
        ), "Pre-op callback called after unregister"
        assert (
            tracker.post_op_count == first_post_op_count
        ), "Post-op callback called after unregister"

    helper.teardown()


def test_multiple_registrations(helper: Helper, request):
    """Test what happens when callbacks are registered multiple times."""
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Test multiple callback registrations",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)
    tracker1 = CallbackTracker()
    tracker2 = CallbackTracker()

    # First registration
    hooks1 = ttrt.runtime.DebugHooks.get(
        pre_exec=tracker1.pre_execution_callback,
        post_exec=tracker1.post_execution_callback,
    )

    assert hooks1 is not None, "Failed to register first hooks"

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        # Run with first callbacks
        output1 = test_runner.run_program(device, inputs_runtime_with_layout)

        # Verify first callbacks were called
        assert tracker1.pre_execution_count == 1
        assert tracker1.post_execution_count == 1
        assert tracker2.pre_execution_count == 0
        assert tracker2.post_execution_count == 0

        # Second registration (should replace the first)
        hooks2 = ttrt.runtime.DebugHooks.get(
            pre_exec=tracker2.pre_execution_callback,
            post_exec=tracker2.post_execution_callback,
        )

        assert hooks2 is not None, "Failed to register second hooks"

        # Run with second callbacks
        output2 = test_runner.run_program(device, inputs_runtime_with_layout)

        # Verify second callbacks were called, first were not called again
        assert (
            tracker1.pre_execution_count == 1
        ), "First tracker should not be called after re-registration"
        assert (
            tracker1.post_execution_count == 1
        ), "First tracker should not be called after re-registration"
        assert (
            tracker2.pre_execution_count == 1
        ), "Second tracker should be called after re-registration"
        assert (
            tracker2.post_execution_count == 1
        ), "Second tracker should be called after re-registration"

    # Cleanup
    ttrt.runtime.unregister_hooks()
    helper.teardown()


def test_partial_re_registration(helper: Helper, request):
    """Test registering some callbacks, then adding more without unregistering."""
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Test partial callback re-registration",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)
    tracker = CallbackTracker()

    # First: Register only execution callbacks
    hooks1 = ttrt.runtime.DebugHooks.get(
        pre_exec=tracker.pre_execution_callback,
        post_exec=tracker.post_execution_callback,
    )

    assert hooks1 is not None, "Failed to register execution hooks"

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        # Run with only execution callbacks
        output1 = test_runner.run_program(device, inputs_runtime_with_layout)

        assert tracker.pre_execution_count == 1
        assert tracker.post_execution_count == 1
        assert tracker.pre_op_count == 0
        assert tracker.post_op_count == 0

        # Second: Add operation callbacks (should merge with existing)
        hooks2 = ttrt.runtime.DebugHooks.get(
            pre_op=tracker.pre_op_callback, post_op=tracker.post_op_callback
        )

        assert hooks2 is not None, "Failed to register operation hooks"

        # Run with all callbacks
        output2 = test_runner.run_program(device, inputs_runtime_with_layout)

        # Both execution and operation callbacks should be active
        assert (
            tracker.pre_execution_count == 2
        ), "Pre-execution callback should still be active"
        assert (
            tracker.post_execution_count == 2
        ), "Post-execution callback should still be active"
        assert tracker.pre_op_count > 0, "Pre-op callback should now be active"
        assert tracker.post_op_count > 0, "Post-op callback should now be active"

    # Cleanup
    ttrt.runtime.unregister_hooks()
    helper.teardown()


def test_callback_exception_handling(helper: Helper, request):
    """Test behavior when callbacks raise exceptions."""
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Test callback exception handling",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    def failing_callback(binary, program_context):
        """A callback that raises an exception."""
        raise RuntimeError("Intentional callback failure")

    # Register a callback that will fail
    hooks = ttrt.runtime.DebugHooks.get(pre_exec=failing_callback)

    assert hooks is not None, "Failed to register hooks"

    with DeviceContext(mesh_shape=[1, 1]) as device:
        inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
            device
        )

        # The exception from the callback should propagate
        with pytest.raises(RuntimeError, match="Intentional callback failure"):
            output = test_runner.run_program(device, inputs_runtime_with_layout)

    # Cleanup
    ttrt.runtime.unregister_hooks()
    helper.teardown()


def test_callback_with_multiple_programs(helper: Helper, request):
    """Test that callbacks work correctly across multiple program executions."""
    if not is_debug_enabled():
        pytest.skip("Debug hooks not enabled (TT_RUNTIME_DEBUG=0)")

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "linear.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="binary_ops",
        expected_num_inputs=4,
        compute_golden=lambda inputs: (inputs[0] + inputs[1])
        * ((inputs[1] + inputs[2]) - (inputs[2] + inputs[3])),
        description="Test callbacks across multiple executions",
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)
    tracker = CallbackTracker()

    # Register callbacks
    hooks = ttrt.runtime.DebugHooks.get(
        pre_exec=tracker.pre_execution_callback,
        post_exec=tracker.post_execution_callback,
    )

    assert hooks is not None, "Failed to register hooks"

    num_executions = 5

    with DeviceContext(mesh_shape=[1, 1]) as device:
        for i in range(num_executions):
            inputs_runtime_with_layout, golden, _ = test_runner.get_inputs_and_golden(
                device
            )
            output = test_runner.run_program(device, inputs_runtime_with_layout)

        # Verify callbacks were called correct number of times
        assert (
            tracker.pre_execution_count == num_executions
        ), f"Pre-execution callback called {tracker.pre_execution_count} times, expected {num_executions}"
        assert (
            tracker.post_execution_count == num_executions
        ), f"Post-execution callback called {tracker.post_execution_count} times, expected {num_executions}"

    # Cleanup
    ttrt.runtime.unregister_hooks()
    helper.teardown()
