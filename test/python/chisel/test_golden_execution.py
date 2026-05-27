# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Execute every TTNN op in a flatbuffer through the golden system on meta
tensors (shape/dtype only) and verify each output matches the MLIR op type.
Ops without a registered golden are skipped, not failed.
"""
import pytest
import torch

from chisel.executor import (
    build_role_keyed_inputs,
    execute_golden,
)
from chisel.op_configs import get_op_names_no_golden
from chisel.ops import (
    get_inplace_vals,
    get_op_inputs,
    get_op_outputs,
    is_tensor_value,
)
from golden import GoldenMapTensor, get_chisel_golden_function
from golden.mapping import mlir_type_to_torch_dtype
from ttmlir.dialects import quant
from utils import iterate_programs

# Ops chisel cannot validate (no golden, structural mismatch, etc.) - sourced
# from op_configs.register_defaults so the skip list is a single source of
# truth shared with the runtime callback.
_SKIPPED_OPS: frozenset[str] = get_op_names_no_golden()


def _has_quantized_type(op) -> bool:
    for value in list(op.operands) + list(op.results):
        if not is_tensor_value(value):
            continue
        if quant.QuantizedType.isinstance(value.type.element_type):
            return True
    return False


@pytest.fixture(autouse=True)
def _default_meta_device():
    # Goldens may internally allocate tensors (e.g. SDPA causal mask,
    # ttnn.arange/full/ones/zeros). Running them on the meta device keeps
    # the test allocation-free and avoids per-golden device plumbing.
    with torch.device("meta"):
        yield


# Skipped until goldens are merged with the builder; until then the test is
# not stable enough to run in CI.
@pytest.mark.skip(reason="Disabled until goldens are merged with the builder.")
def test_golden_execution(subtests, ir_module, binary, binary_path):
    asm_state = ir_module.get_asm_state()

    for _prog_idx, prog_name in iterate_programs(binary):
        for op in ir_module.get_function_ops(prog_name):
            with subtests.test(prog=prog_name, op=op.name):
                if _has_quantized_type(op):
                    pytest.skip(
                        f"quantized tensor types not yet supported (op: {op.name})"
                    )

                if (
                    get_chisel_golden_function(type(op.opview)) is None
                    and op.name in _SKIPPED_OPS
                ):
                    pytest.skip(f"{op.name}: no golden registered")

                inputs = {}
                for operand in get_op_inputs(op):
                    name = operand.get_name(asm_state)
                    shape = list(operand.type.shape)
                    try:
                        dtype = mlir_type_to_torch_dtype(operand.type.element_type)
                    except TypeError as e:
                        pytest.skip(f"{op.name}: unsupported operand dtype ({e})")
                    tensor = torch.empty(shape, dtype=dtype, device="meta")
                    inputs[name] = GoldenMapTensor({0: tensor}, (1, 1))

                golden_inputs = build_role_keyed_inputs(op.opview, inputs, asm_state)
                try:
                    result = execute_golden(op.opview, golden_inputs)
                except TypeError as e:
                    pytest.skip(
                        f"{op.name}: unsupported dtype during golden execution ({e})"
                    )

                op_outputs = get_op_outputs(op)
                for i, op_out in enumerate(op_outputs):
                    expected_shape = list(op_out.type.shape)
                    assert list(result[i].shape) == expected_shape, (
                        f"output[{i}] shape mismatch: got {list(result[i].shape)}, expected {expected_shape}"
                        f"\nbinary: {binary_path}"
                    )
                    try:
                        expected_dtype = mlir_type_to_torch_dtype(
                            op_out.type.element_type
                        )
                    except TypeError as e:
                        pytest.skip(f"{op.name}: unsupported dtype ({e})")
                    assert result[i].dtype == expected_dtype, (
                        f"output[{i}] dtype mismatch: got {result[i].dtype}, expected {expected_dtype}"
                        f"\nbinary: {binary_path}"
                    )

                inplace_vals = get_inplace_vals(op.opview)
                for i, (out, val) in enumerate(
                    zip(result[len(op_outputs) :], inplace_vals)
                ):
                    expected_shape = list(val.type.shape)
                    assert list(out.shape) == expected_shape, (
                        f"inplace[{i}] shape mismatch: got {list(out.shape)}, expected {expected_shape}"
                        f"\nbinary: {binary_path}"
                    )
                    try:
                        expected_dtype = mlir_type_to_torch_dtype(val.type.element_type)
                    except TypeError as e:
                        pytest.skip(f"{op.name}: unsupported dtype ({e})")
                    assert out.dtype == expected_dtype, (
                        f"inplace[{i}] dtype mismatch: got {out.dtype}, expected {expected_dtype}"
                        f"\nbinary: {binary_path}"
                    )
