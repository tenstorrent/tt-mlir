# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Element-type support: every supported dtype matches the CPU golden."""

import numpy as np
import pytest

import tt_crank.onnx as tt_onnx

_STRUCTURAL = [
    tt_onnx.node("Transpose", ["a"], ["t"], perm=[1, 0]),
    tt_onnx.node("Concat", ["t", "t"], ["y"], axis=0),
]

# case -> (nodes, input dtype, output dtype, input count, atol)
_CASES = {
    "add_i32": (
        [tt_onnx.node("Add", ["a", "b"], ["y"])],
        tt_onnx.DType.I32,
        tt_onnx.DType.I32,
        2,
        0.0,
    ),
    "sub_i64": (
        [tt_onnx.node("Sub", ["a", "b"], ["y"])],
        tt_onnx.DType.I64,
        tt_onnx.DType.I64,
        2,
        0.0,
    ),
    "mul_i32": (
        [tt_onnx.node("Mul", ["a", "b"], ["y"])],
        tt_onnx.DType.I32,
        tt_onnx.DType.I32,
        2,
        0.0,
    ),
    "add_f64": (
        [tt_onnx.node("Add", ["a", "b"], ["y"])],
        tt_onnx.DType.F64,
        tt_onnx.DType.F64,
        2,
        2e-2,
    ),
    "neg_i32": (
        [tt_onnx.node("Neg", ["a"], ["y"])],
        tt_onnx.DType.I32,
        tt_onnx.DType.I32,
        1,
        0.0,
    ),
    "structural_i32": (_STRUCTURAL, tt_onnx.DType.I32, tt_onnx.DType.I32, 1, 0.0),
    "structural_bool": (_STRUCTURAL, tt_onnx.DType.BOOL, tt_onnx.DType.BOOL, 1, 0.0),
    "cast_f32_i32": (
        [tt_onnx.node("Cast", ["a"], ["y"], to=tt_onnx.DType.I32)],
        tt_onnx.DType.F32,
        tt_onnx.DType.I32,
        1,
        0.0,
    ),
    "cast_i32_f32": (
        [tt_onnx.node("Cast", ["a"], ["y"], to=tt_onnx.DType.F32)],
        tt_onnx.DType.I32,
        tt_onnx.DType.F32,
        1,
        1e-5,
    ),
    "cast_f32_i64": (
        [tt_onnx.node("Cast", ["a"], ["y"], to=tt_onnx.DType.I64)],
        tt_onnx.DType.F32,
        tt_onnx.DType.I64,
        1,
        0.0,
    ),
}


def _random(shape, onnx_dtype) -> np.ndarray:
    dtype = tt_onnx.numpy_dtype(onnx_dtype)
    if dtype == np.bool_:
        return tt_onnx.randint(*shape, low=0, high=2, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        return tt_onnx.randint(*shape, dtype=dtype)
    return tt_onnx.randn(*shape, dtype=dtype) * 5


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_dtype_support(case: str) -> None:
    nodes, in_dtype, out_dtype, n_inputs, atol = _CASES[case]
    inputs = {name: _random((16, 16), in_dtype) for name in "ab"[:n_inputs]}
    model = tt_onnx.model(
        nodes,
        [tt_onnx.input(name, [16, 16], in_dtype) for name in inputs],
        [tt_onnx.output("y", None, out_dtype)],
    )
    tt_onnx.assert_tt_matches_cpu(model, inputs, atol=atol, rtol=atol)


@pytest.mark.parametrize("op", ["MatMul", "Div"])
def test_integer_matmul_div_fail_session_creation(op: str) -> None:
    # ttnn matmul / ttir.div are float-only; the builder rejects integer operands at compile.
    model = tt_onnx.model(
        [tt_onnx.node(op, ["a", "b"], ["y"])],
        [
            tt_onnx.input("a", [16, 16], tt_onnx.DType.I32),
            tt_onnx.input("b", [16, 16], tt_onnx.DType.I32),
        ],
        [tt_onnx.output("y", None, tt_onnx.DType.I32)],
    )
    with pytest.raises(Exception, match="integer operands not supported"):
        tt_onnx.session(model, allow_cpu_fallback=True)
