# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hardware-free test: walk_binary + IRModule shape cross-validation.

For every op in every program, zips MLIR-level operands/results with the
runtime TensorRefs and asserts that reported shapes agree. Each op runs
as a pytest subtest so failures are reported individually.
"""
from _ttmlir_runtime import runtime as rt

from chisel.ops import get_op_inputs, get_op_outputs


# scale/zero_point operands are stored as FB attribute structs, not TensorRefs
_SKIP_OPS = {"ttnn.quantize", "ttnn.dequantize", "ttnn.requantize"}


def test_walk_binary_shapes_match_ir(binary, ir_module, subtests):
    """TensorRef shapes from walk_binary must match shapes in the IR module."""

    for prog_idx in range(binary.get_num_programs()):
        prog_name = binary.get_program_name(prog_idx)
        mlir_ops = iter(ir_module.get_function_ops(prog_name))

        def _check(_bin, prog_ctx, op_ctx, _mlir_ops=mlir_ops):
            mlir_op = next(_mlir_ops)

            if mlir_op.name in _SKIP_OPS:
                return

            with subtests.test(op=mlir_op.name):
                mlir_inputs = get_op_inputs(mlir_op)
                input_refs = rt.get_op_input_refs(op_ctx, prog_ctx)
                assert len(mlir_inputs) == len(input_refs), (
                    f"{mlir_op.name}: input count mismatch "
                    f"(IR={len(mlir_inputs)}, FB={len(input_refs)})"
                )
                for val, ref in zip(mlir_inputs, input_refs):
                    ir_shape = list(val.type.shape)
                    fb_shape = ref.get_shape()
                    assert ir_shape == fb_shape, (
                        f"{mlir_op.name} input shape mismatch: IR={ir_shape} FB={fb_shape}"
                    )

                mlir_outputs = get_op_outputs(mlir_op)
                output_refs = rt.get_op_output_refs(op_ctx, prog_ctx)
                assert len(mlir_outputs) == len(output_refs), (
                    f"{mlir_op.name}: output count mismatch "
                    f"(IR={len(mlir_outputs)}, FB={len(output_refs)})"
                )
                for val, ref in zip(mlir_outputs, output_refs):
                    ir_shape = list(val.type.shape)
                    fb_shape = ref.get_shape()
                    assert ir_shape == fb_shape, (
                        f"{mlir_op.name} output shape mismatch: IR={ir_shape} FB={fb_shape}"
                    )

        rt.walk_binary(binary, prog_idx, _check)
