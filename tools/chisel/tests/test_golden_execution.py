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
from ttmlir.ir import OpOperandList, Value

from chisel.executor import execute_golden
from chisel.ops import get_op_outputs
from golden import GoldenMapTensor, is_non_executable_op
from golden.mapping import mlir_type_to_torch_dtype


def _meta_input(value: Value) -> GoldenMapTensor:
    shape = list(value.type.shape)
    dtype = mlir_type_to_torch_dtype(value.type.element_type)
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    return GoldenMapTensor({0: tensor}, (1, 1))


@pytest.mark.no_device
def test_golden_execution(subtests, ir_module, binary, mlir_source_path):
    """Execute each TTNN op on the meta device; verify output shape and dtype."""
    for i in range(binary.get_num_programs()):
        prog_name = binary.get_program_name(i)

        for op in ir_module.get_function_ops(prog_name):
            with subtests.test(prog=prog_name, op=op.name):
                if is_non_executable_op(type(op.opview)):
                    pytest.skip(f"golden not applicable for {type(op.opview).__name__}")

                if op.name in ("ttnn.quantize", "ttnn.requantize"):
                    pytest.skip("quantize/requantize golden not supported")

                operand_names = getattr(type(op.opview), "OPERAND_NAMES", None)
                if operand_names is None:
                    pytest.skip(f"no OPERAND_NAMES for {type(op.opview).__name__}")

                inputs = {}
                for name in operand_names:
                    accessor = getattr(op.opview, name, None)
                    if accessor is None:
                        inputs[name] = None
                    elif isinstance(accessor, OpOperandList):
                        inputs[name] = [_meta_input(v) for v in accessor]
                    elif isinstance(accessor, Value):
                        inputs[name] = _meta_input(accessor)
                    else:
                        pytest.skip(
                            f"unsupported {name} accessor type "
                            f"{type(accessor).__name__}"
                        )

                result = execute_golden(op.opview, inputs)
                if result is None:
                    pytest.skip(f"no golden for {type(op.opview).__name__}")
                    return

                op_outputs = get_op_outputs(op)
                if op_outputs:
                    expected_shape = list(op_outputs[0].type.shape)
                    expected_dtype = mlir_type_to_torch_dtype(
                        op_outputs[0].type.element_type
                    )
                    assert list(result.shape) == expected_shape, (
                        f"shape mismatch: got {list(result.shape)}, expected {expected_shape}"
                        f"\nMLIR source: {mlir_source_path}"
                    )
                    assert result.dtype == expected_dtype, (
                        f"dtype mismatch: got {result.dtype}, expected {expected_dtype}"
                        f"\nMLIR source: {mlir_source_path}"
                    )
