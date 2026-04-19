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
import ttrt.binary

from chisel.executor import execute_golden
from chisel.ops import IRModule, get_op_inputs, get_op_outputs


_DTYPE_MAP = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "i32": torch.int32,
    "i16": torch.int16,
    "i8": torch.int8,
    "i64": torch.int64,
    "i1": torch.bool,
}


def _element_type_to_torch_dtype(element_type) -> torch.dtype:
    return _DTYPE_MAP.get(str(element_type), torch.float32)


@pytest.fixture
def binary(binary_path):
    return ttrt.binary.load_binary_from_path(binary_path)


@pytest.fixture
def ir_module(binary):
    mlir_json = ttrt.binary.mlir_as_dict(binary)
    functions = [binary.get_program_name(i) for i in range(binary.get_num_programs())]
    return IRModule(mlir_source=mlir_json["source"], functions=functions)


def _iterate_programs(binary):
    """Yield (index, name) for each program in the binary."""
    for i in range(binary.get_num_programs()):
        yield i, binary.get_program_name(i)


def test_golden_execution(ir_module, binary):
    """Execute each TTNN op on the meta device; verify output shape and dtype."""
    executed = 0
    skipped = 0
    failed = []

    for _prog_idx, prog_name in _iterate_programs(binary):
        asm_state = ir_module.get_asm_state(prog_name)

        for op in ir_module.get_function_ops(prog_name):
            inputs = {}
            for operand in get_op_inputs(op):
                name = operand.get_name(asm_state)
                shape = list(operand.type.shape)
                dtype = _element_type_to_torch_dtype(operand.type.element_type)
                inputs[name] = torch.empty(shape, dtype=dtype, device="meta")

            try:
                result = execute_golden(op, ir_module, prog_name, inputs)
            except RuntimeError:
                skipped += 1
                continue
            except Exception as e:
                failed.append((prog_name, op.name, str(e)))
                continue

            op_outputs = get_op_outputs(op)
            if op_outputs:
                expected_shape = list(op_outputs[0].type.shape)
                expected_dtype = _element_type_to_torch_dtype(
                    op_outputs[0].type.element_type
                )
                if list(result.shape) != expected_shape:
                    failed.append(
                        (
                            prog_name,
                            op.name,
                            f"shape mismatch: got {list(result.shape)}, expected {expected_shape}",
                        )
                    )
                    continue
                if result.dtype != expected_dtype:
                    failed.append(
                        (
                            prog_name,
                            op.name,
                            f"dtype mismatch: got {result.dtype}, expected {expected_dtype}",
                        )
                    )
                    continue

            executed += 1

    total = executed + skipped + len(failed)
    print(
        f"\nGolden execution: {executed} executed, {skipped} skipped, "
        f"{len(failed)} failed (total {total})"
    )

    if failed:
        details = "\n".join(
            f"  [{prog}/{op_name}]: {err}" for prog, op_name, err in failed
        )
        pytest.fail(f"{len(failed)} op(s) raised unexpected errors:\n{details}")
