# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gemm and MatMul: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_onnx


@pytest.mark.parametrize(
    "attrs,a_shape,b_shape,c_shape",
    [
        ({"transB": 1}, (8, 32), (64, 32), (64,)),  # linear fast path
        ({"transB": 1}, (8, 32), (64, 32), None),
        ({"alpha": 0.5, "beta": 2.0, "transA": 1}, (32, 8), (32, 64), (8, 64)),
        ({}, (8, 32), (32, 64), None),
    ],
)
def test_gemm(attrs: dict, a_shape, b_shape, c_shape) -> None:
    inputs = ["a", "b"] + (["c"] if c_shape else [])
    initializers = [tt_onnx.make_tensor("b", tt_onnx.randn(*b_shape))]
    if c_shape:
        initializers.append(tt_onnx.make_tensor("c", tt_onnx.randn(*c_shape)))
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Gemm", inputs, ["y"], **attrs)],
        [tt_onnx.vi("a", list(a_shape))],
        [tt_onnx.vi("y", None)],
        initializers,
    )
    tt_onnx.assert_tt_matches_cpu(
        model, {"a": tt_onnx.randn(*a_shape)}, atol=0.1, rtol=0.1
    )


def test_gemm_beta_zero_ignores_c() -> None:
    # BLAS convention (and ORT CPU): beta=0 never reads C, so NaNs in C must not propagate.
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Gemm", ["a", "b", "c"], ["y"], beta=0.0)],
        [tt_onnx.vi("a", [8, 32])],
        [tt_onnx.vi("y", None)],
        [
            tt_onnx.make_tensor("b", tt_onnx.randn(32, 64)),
            tt_onnx.make_tensor("c", np.full((8, 64), np.nan, dtype=np.float32)),
        ],
    )
    (y,) = tt_onnx.run_on_tt(tt_onnx.session_on_tt(model), {"a": tt_onnx.randn(8, 32)})
    assert not np.isnan(y).any()


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((2, 32, 64), (2, 64, 32)),  # batched
        ((32, 64), (2, 64, 32)),  # 2-D lhs against batched rhs: batch comes from rhs
    ],
)
def test_matmul(a_shape, b_shape) -> None:
    model = tt_onnx.make_model(
        [tt_onnx.make_node("MatMul", ["a", "b"], ["y"])],
        [tt_onnx.vi("a", list(a_shape)), tt_onnx.vi("b", list(b_shape))],
        [tt_onnx.vi("y", None)],
    )
    tt_onnx.assert_tt_matches_cpu(
        model,
        {"a": tt_onnx.randn(*a_shape), "b": tt_onnx.randn(*b_shape)},
        atol=0.1,
        rtol=0.1,
    )
