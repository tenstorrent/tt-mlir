# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hardware-free test: walk_binary + TensorRef.get_shape / get_dtype.

Iterates every op in every program of a compiled .ttnn flatbuffer and
validates that TensorRef metadata (shape, dtype) is readable without
device access or tensor allocation.
"""
import ttrt.runtime as rt


def test_walk_binary_shape_dtype(binary):
    """walk_binary fires a callback per op; get_shape/dtype read flatbuffer metadata."""
    collected = []

    def _collect(_bin, prog_ctx, op_ctx):
        for ref in rt.get_op_input_refs(op_ctx, prog_ctx):
            shape = ref.get_shape()
            dtype = ref.get_dtype()
            assert isinstance(
                shape, list
            ), f"get_shape() should return list, got {type(shape)}"
            assert len(shape) > 0, "get_shape() returned empty list"
            assert dtype is not None
            collected.append(("in", shape, dtype))
        for ref in rt.get_op_output_refs(op_ctx, prog_ctx):
            shape = ref.get_shape()
            dtype = ref.get_dtype()
            assert isinstance(shape, list)
            assert len(shape) > 0
            assert dtype is not None
            collected.append(("out", shape, dtype))

    for prog_idx in range(binary.get_num_programs()):
        rt.walk_binary(binary, prog_idx, _collect)

    assert len(collected) > 0, (
        "Expected at least one TensorRef across all programs — "
        "binary may have no ops with inputs/outputs"
    )
