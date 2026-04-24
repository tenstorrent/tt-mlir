# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Execute every TTNN op in a flatbuffer binary through the golden system using
meta tensors (no actual data, shape/dtype only). Ops without a golden
implementation are skipped (not failed). Verifies that each golden output
matches the expected shape and dtype from the MLIR op type. Reports executed
vs skipped counts.
"""
import pytest
import torch

from chisel.executor import execute_golden
from chisel.ops import get_op_inputs, get_op_outputs
from golden import is_non_executable_op
from golden.mapping import mlir_type_to_torch_dtype as _mlir_type_to_torch_dtype


_DTYPE_MAP = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "i32": torch.int32,
    "i16": torch.int16,
    "i8": torch.int8,
    "i64": torch.int64,
    "i1": torch.bool,
    "ui8": torch.uint8,
    "ui16": torch.uint16,
    "ui32": torch.uint32,
    "ui64": torch.uint64,
}


def _element_type_to_torch_dtype(element_type) -> torch.dtype:
    try:
        return _mlir_type_to_torch_dtype(element_type)
    except TypeError:
        return torch.float32


def _iterate_programs(binary):
    """Yield (index, name) for each program in the binary."""
    for i in range(binary.get_num_programs()):
        yield i, binary.get_program_name(i)


@pytest.mark.no_device
def test_golden_execution(subtests, ir_module, binary, mlir_source_path):
    """Execute each TTNN op on the meta device; verify output shape and dtype."""
    for _prog_idx, prog_name in _iterate_programs(binary):
        asm_state = ir_module.get_asm_state()

        for op in ir_module.get_function_ops(prog_name):
            with subtests.test(prog=prog_name, op=op.name):
                if is_non_executable_op(type(op.opview)):
                    pytest.skip(f"golden not applicable for {type(op.opview).__name__}")

                if op.name in ("ttnn.quantize", "ttnn.requantize"):
                    pytest.skip("quantize/requantize golden not supported")

                inputs = {}
                for operand in get_op_inputs(op):
                    name = operand.get_name(asm_state)
                    shape = list(operand.type.shape)
                    dtype = _element_type_to_torch_dtype(operand.type.element_type)
                    inputs[name] = torch.empty(shape, dtype=dtype, device="meta")

                result = execute_golden(op.opview, ir_module, inputs)
                if result is None:
                    pytest.skip(f"no golden for {type(op.opview).__name__}")

                op_outputs = get_op_outputs(op)
                if op_outputs:
                    expected_shape = list(op_outputs[0].type.shape)
                    expected_dtype = _element_type_to_torch_dtype(
                        op_outputs[0].type.element_type
                    )
                    first = result[0]
                    assert list(first.shape) == expected_shape, (
                        f"shape mismatch: got {list(first.shape)}, expected {expected_shape}"
                        f"\nMLIR source: {mlir_source_path}"
                    )
                    assert first.dtype == expected_dtype, (
                        f"dtype mismatch: got {first.dtype}, expected {expected_dtype}"
                        f"\nMLIR source: {mlir_source_path}"
                    )
