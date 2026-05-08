# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import _ttmlir_runtime as tt_runtime

from chisel.ops import get_op_inputs, get_op_outputs


# For now skipping quantization OPS
_SKIP_OPS = {"ttnn.quantize", "ttnn.dequantize", "ttnn.requantize"}

# Ops whose `getOpInputRefs` coverage is missing or wrong.
_BAD_INPUT_REF_OPS: set[str] = {
    "ttnn.clamp_tensor",
    "ttnn.conv2d",
    "ttnn.conv3d",
    "ttnn.conv_transpose2d",
    "ttnn.embedding_bw",
    "ttnn.nlp_create_qkv_heads_decode",
    "ttnn.paged_update_cache",
    "ttnn.scaled_dot_product_attention",
    "ttnn.scaled_dot_product_attention_decode",
    "ttnn.split_query_key_value_and_split_heads",
}

# Ops whose `getOpOutputRefs` coverage is missing or wrong.
_BAD_OUTPUT_REF_OPS: set[str] = {
    "func.call",
    "ttcore.load_cached",
    "ttnn.batch_norm_training",
    "ttnn.fill_cache",
    "ttnn.generic",
    "ttnn.max_pool2d_with_indices",
    "ttnn.nlp_create_qkv_heads_decode",
    "ttnn.paged_fill_cache",
    "ttnn.paged_update_cache",
    "ttnn.update_cache",
    "ttnn.sort",
    "ttnn.split_query_key_value_and_split_heads",
}


def test_walk_program_shapes_match_ir(binary, ir_module, subtests):
    """TensorRef shapes from walk_program must match shapes in the IR module."""

    for prog_idx in range(binary.get_num_programs()):
        prog_name = binary.get_program_name(prog_idx)
        mlir_ops = iter(ir_module.get_function_ops(prog_name))

        def _check(op_ctx):
            mlir_op = next(mlir_ops)

            if mlir_op.name in _SKIP_OPS:
                return

            with subtests.test(op=mlir_op.name, side="inputs"):
                if mlir_op.name in _BAD_INPUT_REF_OPS:
                    pytest.xfail(
                        f"{mlir_op.name}: getOpInputRefs not properly implemented"
                    )
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
                if mlir_op.name in _BAD_OUTPUT_REF_OPS:
                    pytest.xfail(
                        f"{mlir_op.name}: getOpOutputRefs not properly implemented"
                    )
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
