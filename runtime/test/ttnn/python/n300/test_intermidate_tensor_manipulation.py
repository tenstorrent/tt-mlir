# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
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
    assert_pcc,
    get_torch_inputs,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
    get_to_layout_inputs,
)

MESH_SHAPE = [1, 2]

op_trace = []


def get_tensors_info(program_context, tensor_ref):
    """Retrieve per-device shapes and dtypes for a tensor ref from the pool.
    Uses retrieve_tensors_from_pool which supports multi-device tensors."""
    try:
        tensors = ttrt.runtime.retrieve_tensors_from_pool(
            program_context, tensor_ref
        )
        if not tensors:
            return None
        return [
            {"shape": t.get_shape(), "dtype": str(t.get_dtype())}
            for t in tensors
        ]
    except RuntimeError:
        return None


def preop_hook(binary, program_context, op_context):
    debug_str = ttrt.runtime.get_op_debug_str(op_context)
    try:
        input_refs = ttrt.runtime.get_op_input_refs(op_context, program_context)
        input_infos = [
            get_tensors_info(program_context, ref) for ref in input_refs
        ]
    except RuntimeError:
        input_infos = []
    op_trace.append(
        {"stage": "pre", "op": debug_str, "inputs": input_infos, "output": None}
    )


def postop_hook(binary, program_context, op_context):
    debug_str = ttrt.runtime.get_op_debug_str(op_context)
    output_info = None
    try:
        output_ref = ttrt.runtime.get_op_output_ref(op_context, program_context)
        if output_ref is not None:
            output_info = get_tensors_info(program_context, output_ref)
    except RuntimeError:
        pass
    op_trace.append(
        {"stage": "post", "op": debug_str, "inputs": None, "output": output_info}
    )


def test_multichip_add(helper: Helper, request):
    """Run a truly multichip graph (distribute_tensor -> add -> aggregate_tensor)
    on a 1x2 mesh and verify the full output matches the golden."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "multichip_add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    test_config = ProgramTestConfig(
        name="multichip_add",
        expected_num_inputs=2,
        compute_golden=lambda inputs: torch.add(inputs[0], inputs[1]),
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        inputs, golden, torch_inputs = test_runner.get_inputs_and_golden(
            mesh_device, borrow=False
        )
        test_runner.run_program_and_compare_golden(mesh_device, inputs, golden)

    helper.teardown()


def test_multichip_add_op_trace(helper: Helper, request):
    """Run the multichip graph with debug hooks to inspect tensor shapes
    and counts before/after each op in the runtime execution."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "multichip_add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    op_trace.clear()
    hooks = ttrt.runtime.DebugHooks.get(preop_hook, postop_hook)

    if hooks is None:
        return

    program = helper.binary.get_program(0)
    torch_inputs = get_torch_inputs(program)
    runtime_inputs = [
        get_runtime_tensor_from_torch(torch_input) for torch_input in torch_inputs
    ]

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        runtime_inputs_with_layout = get_to_layout_inputs(
            mesh_device, runtime_inputs, helper.binary, 0
        )

        output = ttrt.runtime.submit(
            mesh_device, helper.binary.fbb, 0, runtime_inputs_with_layout
        )[0]

        output_host = ttrt.runtime.to_host(output, untilize=True)[0]
        output_torch = get_torch_output_container(program)
        ttrt.runtime.memcpy(output_torch.data_ptr(), output_host)
        golden = torch.add(torch_inputs[0], torch_inputs[1])
        assert_pcc(output_torch, golden, threshold=0.99)

    ttrt.runtime.unregister_hooks()

    print("\n=== Multichip Op Trace ===")
    for entry in op_trace:
        stage = entry["stage"]
        op = entry["op"]
        if stage == "pre":
            print(f"[PRE ] {op}")
            for idx, per_device in enumerate(entry["inputs"]):
                if per_device is None:
                    print(f"        input[{idx}]: N/A")
                else:
                    shapes = [d["shape"] for d in per_device]
                    print(
                        f"        input[{idx}]: {len(per_device)} device(s), "
                        f"shapes={shapes}"
                    )
        else:
            out = entry["output"]
            if out is None:
                print(f"[POST] {op}")
                print("        output: N/A")
            else:
                shapes = [d["shape"] for d in out]
                print(f"[POST] {op}")
                print(
                    f"        output: {len(out)} device(s), shapes={shapes}"
                )
    print("=== End Trace ===\n")

    assert len(op_trace) > 0, "No ops were traced"

    helper.teardown()


def update_device_tensor(program_context, tensor_ref, dst_tensor, src_tensor):
    """Replace a tensor in the pool with new data.
    Works for both single-device and multi-device tensors: the C++ side
    calls to_device(src, dstTensor.device()) which handles mesh devices."""
    data_ptr = src_tensor.data_ptr()
    shape = dst_tensor.get_shape()
    stride = dst_tensor.get_stride()
    dtype = dst_tensor.get_dtype()
    size = torch.numel(src_tensor)
    tensor = ttrt.runtime.create_owned_host_tensor(
        data_ptr, shape, stride, size, dtype
    )
    ttrt.runtime.update_tensor_in_pool(program_context, tensor_ref, tensor)


def make_postop_override_add(override_value=10.0):
    """Create a postop hook that replaces the ttnn.add output with a constant."""

    def postop_override(binary, program_context, op_context):
        debug_str = ttrt.runtime.get_op_debug_str(op_context)
        if "ttnn.add" not in debug_str:
            return

        tensor_ref = ttrt.runtime.get_op_output_ref(op_context, program_context)
        if tensor_ref is None:
            return

        tensors = ttrt.runtime.retrieve_tensors_from_pool(
            program_context, tensor_ref
        )
        if not tensors:
            return

        per_device_shape = tensors[0].get_shape()
        replacement = torch.full(per_device_shape, override_value, dtype=torch.float32)
        update_device_tensor(program_context, tensor_ref, tensors[0], replacement)

    return postop_override


def test_multichip_add_update_output(helper: Helper, request):
    """After ttnn.add, replace its multi-device output with all 10s.
    The aggregate_tensor should then produce a 64x128 tensor of all 10s."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "multichip_add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    override_val = 10.0
    hooks = ttrt.runtime.DebugHooks.get(
        lambda b, pc, oc: None,
        make_postop_override_add(override_val),
    )

    if hooks is None:
        return

    program = helper.binary.get_program(0)
    torch_inputs = get_torch_inputs(program)
    runtime_inputs = [
        get_runtime_tensor_from_torch(torch_input) for torch_input in torch_inputs
    ]

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        runtime_inputs_with_layout = get_to_layout_inputs(
            mesh_device, runtime_inputs, helper.binary, 0
        )

        output = ttrt.runtime.submit(
            mesh_device, helper.binary.fbb, 0, runtime_inputs_with_layout
        )[0]

        output_host = ttrt.runtime.to_host(output, untilize=True)[0]
        output_torch = get_torch_output_container(program)
        ttrt.runtime.memcpy(output_torch.data_ptr(), output_host)

        expected = torch.full_like(output_torch, override_val)
        print(f"\nOutput sample (first row): {output_torch[0, :8]}")
        print(f"Expected: all {override_val}")
        assert torch.allclose(output_torch, expected, atol=0.1), (
            f"Expected all {override_val}, got min={output_torch.min()}, "
            f"max={output_torch.max()}"
        )

    ttrt.runtime.unregister_hooks()
    helper.teardown()


def test_multichip_add_mixed(helper: Helper, request):
    """Run a multichip graph where arg0 is sharded and arg1 is replicated,
    then verify the aggregated output matches the golden."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "multichip_add_mixed.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    def compute_golden(inputs):
        # inputs[0] is 64x128 (sharded on dim 1), inputs[1] is 64x64 (replicated)
        # Each device gets: arg0_shard[:, half] + arg1, then shards are concatenated.
        half = inputs[0].shape[1] // 2
        left = inputs[0][:, :half] + inputs[1]
        right = inputs[0][:, half:] + inputs[1]
        return torch.cat([left, right], dim=1)

    test_config = ProgramTestConfig(
        name="multichip_add_mixed",
        expected_num_inputs=2,
        compute_golden=compute_golden,
    )

    test_runner = ProgramTestRunner(test_config, helper.binary, 0)

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        inputs, golden, torch_inputs = test_runner.get_inputs_and_golden(
            mesh_device, borrow=False
        )
        test_runner.run_program_and_compare_golden(mesh_device, inputs, golden)

    helper.teardown()


def test_multichip_add_mixed_op_trace(helper: Helper, request):
    """Run the mixed-shard graph with debug hooks to compare tensor shapes
    between the sharded and replicated paths."""
    assert ttrt.runtime.get_num_available_devices() == 2

    binary_path = os.path.join(
        FLATBUFFER_BASE_PATH, "multichip_add_mixed.mlir.tmp.ttnn"
    )
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    op_trace.clear()
    hooks = ttrt.runtime.DebugHooks.get(preop_hook, postop_hook)

    if hooks is None:
        return

    program = helper.binary.get_program(0)
    torch_inputs = get_torch_inputs(program)
    runtime_inputs = [
        get_runtime_tensor_from_torch(torch_input) for torch_input in torch_inputs
    ]

    with DeviceContext(mesh_shape=MESH_SHAPE) as mesh_device:
        runtime_inputs_with_layout = get_to_layout_inputs(
            mesh_device, runtime_inputs, helper.binary, 0
        )

        output = ttrt.runtime.submit(
            mesh_device, helper.binary.fbb, 0, runtime_inputs_with_layout
        )[0]

        output_host = ttrt.runtime.to_host(output, untilize=True)[0]
        output_torch = get_torch_output_container(program)
        ttrt.runtime.memcpy(output_torch.data_ptr(), output_host)

        half = torch_inputs[0].shape[1] // 2
        golden = torch.cat(
            [
                torch_inputs[0][:, :half] + torch_inputs[1],
                torch_inputs[0][:, half:] + torch_inputs[1],
            ],
            dim=1,
        )
        assert_pcc(output_torch, golden, threshold=0.99)

    ttrt.runtime.unregister_hooks()

    print("\n=== Mixed Shard/Replicate Op Trace ===")
    for entry in op_trace:
        stage = entry["stage"]
        op = entry["op"]
        if stage == "pre":
            print(f"[PRE ] {op}")
            for idx, per_device in enumerate(entry["inputs"]):
                if per_device is None:
                    print(f"        input[{idx}]: N/A")
                else:
                    shapes = [d["shape"] for d in per_device]
                    print(
                        f"        input[{idx}]: {len(per_device)} device(s), "
                        f"shapes={shapes}"
                    )
        else:
            out = entry["output"]
            if out is None:
                print(f"[POST] {op}")
                print("        output: N/A")
            else:
                shapes = [d["shape"] for d in out]
                print(f"[POST] {op}")
                print(
                    f"        output: {len(out)} device(s), shapes={shapes}"
                )
    print("=== End Trace ===\n")

    assert len(op_trace) > 0, "No ops were traced"

    helper.teardown()
