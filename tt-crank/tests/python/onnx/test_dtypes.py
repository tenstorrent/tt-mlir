# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Element-type support: every supported dtype matches the CPU golden."""

import numpy as np
import pytest
from onnx import helper

from ort_ep_utils import (
    DType,
    assert_tt_matches_cpu,
    make_model,
    randint,
    randn,
    session_on_tt,
    vi,
)

_STRUCTURAL = [
    helper.make_node("Transpose", ["a"], ["t"], perm=[1, 0]),
    helper.make_node("Concat", ["t", "t"], ["y"], axis=0),
]

# case -> (nodes, input dtype, output dtype, input count, atol)
_CASES = {
    "add_i32": (
        [helper.make_node("Add", ["a", "b"], ["y"])],
        DType.I32,
        DType.I32,
        2,
        0.0,
    ),
    "sub_i64": (
        [helper.make_node("Sub", ["a", "b"], ["y"])],
        DType.I64,
        DType.I64,
        2,
        0.0,
    ),
    "mul_i32": (
        [helper.make_node("Mul", ["a", "b"], ["y"])],
        DType.I32,
        DType.I32,
        2,
        0.0,
    ),
    "add_f64": (
        [helper.make_node("Add", ["a", "b"], ["y"])],
        DType.F64,
        DType.F64,
        2,
        2e-2,
    ),
    "neg_i32": ([helper.make_node("Neg", ["a"], ["y"])], DType.I32, DType.I32, 1, 0.0),
    "structural_i32": (_STRUCTURAL, DType.I32, DType.I32, 1, 0.0),
    "structural_bool": (_STRUCTURAL, DType.BOOL, DType.BOOL, 1, 0.0),
    "cast_f32_i32": (
        [helper.make_node("Cast", ["a"], ["y"], to=DType.I32)],
        DType.F32,
        DType.I32,
        1,
        0.0,
    ),
    "cast_i32_f32": (
        [helper.make_node("Cast", ["a"], ["y"], to=DType.F32)],
        DType.I32,
        DType.F32,
        1,
        1e-5,
    ),
    "cast_f32_i64": (
        [helper.make_node("Cast", ["a"], ["y"], to=DType.I64)],
        DType.F32,
        DType.I64,
        1,
        0.0,
    ),
}


def _random(shape, onnx_dtype) -> np.ndarray:
    dtype = helper.tensor_dtype_to_np_dtype(onnx_dtype)
    if dtype == np.bool_:
        return randint(*shape, low=0, high=2, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        return randint(*shape, dtype=dtype)
    return randn(*shape, dtype=dtype) * 5


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_dtype_support(case: str) -> None:
    nodes, in_dtype, out_dtype, n_inputs, atol = _CASES[case]
    inputs = {name: _random((16, 16), in_dtype) for name in "ab"[:n_inputs]}
    model = make_model(
        nodes,
        [vi(name, [16, 16], in_dtype) for name in inputs],
        [vi("y", None, out_dtype)],
    )
    assert_tt_matches_cpu(model, inputs, atol=atol, rtol=atol)


@pytest.mark.parametrize("op", ["MatMul", "Div"])
def test_integer_matmul_div_fail_session_creation(op: str) -> None:
    # ttnn matmul / ttir.div are float-only; the builder rejects integer operands at compile.
    model = make_model(
        [helper.make_node(op, ["a", "b"], ["y"])],
        [vi("a", [16, 16], DType.I32), vi("b", [16, 16], DType.I32)],
        [vi("y", None, DType.I32)],
    )
    with pytest.raises(Exception, match="integer operands not supported"):
        session_on_tt(model, allow_cpu_fallback=True)
