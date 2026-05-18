# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import _ttmlir_runtime as tt_runtime

from chisel.op_configs import get_op_names_no_golden
from chisel.ops import get_op_inputs, get_op_outputs
from utils import iterate_programs

_SKIPPED_OPS: frozenset[str] = get_op_names_no_golden()


def test_walk_program_shapes_match_ir(binary, ir_module, subtests):
    """TensorRef shapes from walk_program must match shapes in the IR module."""

    for prog_idx, prog_name in iterate_programs(binary):
        mlir_ops = iter(ir_module.get_function_ops(prog_name))

        def _check(op_ctx):
            mlir_op = next(mlir_ops)

            if mlir_op.name in _SKIPPED_OPS:
                return

            with subtests.test(op=mlir_op.name, side="inputs"):
                mlir_inputs = get_op_inputs(mlir_op)
                input_refs = tt_runtime.runtime.get_op_input_refs(op_ctx)
                assert len(mlir_inputs) == len(input_refs), (
                    f"{mlir_op.name}: input count mismatch "
                    f"(IR={len(mlir_inputs)}, FB={len(input_refs)})"
                )
                for val, ref in zip(mlir_inputs, input_refs):
                    ir_shape = list(val.type.shape)
                    fb_shape = ref.get_shape()
                    assert (
                        ir_shape == fb_shape
                    ), f"{mlir_op.name} input shape mismatch: IR={ir_shape} FB={fb_shape}"

            with subtests.test(op=mlir_op.name, side="outputs"):
                mlir_outputs = get_op_outputs(mlir_op)
                output_refs = tt_runtime.runtime.get_op_output_refs(op_ctx)
                assert len(mlir_outputs) == len(output_refs), (
                    f"{mlir_op.name}: output count mismatch "
                    f"(IR={len(mlir_outputs)}, FB={len(output_refs)})"
                )
                for val, ref in zip(mlir_outputs, output_refs):
                    ir_shape = list(val.type.shape)
                    fb_shape = ref.get_shape()
                    assert (
                        ir_shape == fb_shape
                    ), f"{mlir_op.name} output shape mismatch: IR={ir_shape} FB={fb_shape}"

        tt_runtime.runtime.walk_program(binary, prog_idx, _check)
