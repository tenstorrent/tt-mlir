# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Gemm and MatMul: tt EP vs CPU golden."""

import numpy as np
import pytest
from onnx import helper

from ort_ep_utils import (
    assert_tt_matches_cpu,
    make_model,
    make_tensor,
    randn,
    run_on_tt,
    session_on_tt,
    vi,
)


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
    initializers = [make_tensor("b", randn(*b_shape))]
    if c_shape:
        initializers.append(make_tensor("c", randn(*c_shape)))
    model = make_model(
        [helper.make_node("Gemm", inputs, ["y"], **attrs)],
        [vi("a", list(a_shape))],
        [vi("y", None)],
        initializers,
    )
    assert_tt_matches_cpu(model, {"a": randn(*a_shape)}, atol=0.1, rtol=0.1)


def test_gemm_beta_zero_ignores_c() -> None:
    # BLAS convention (and ORT CPU): beta=0 never reads C, so NaNs in C must not propagate.
    model = make_model(
        [helper.make_node("Gemm", ["a", "b", "c"], ["y"], beta=0.0)],
        [vi("a", [8, 32])],
        [vi("y", None)],
        [
            make_tensor("b", randn(32, 64)),
            make_tensor("c", np.full((8, 64), np.nan, dtype=np.float32)),
        ],
    )
    (y,) = run_on_tt(session_on_tt(model), {"a": randn(8, 32)})
    assert not np.isnan(y).any()


@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        ((2, 32, 64), (2, 64, 32)),  # batched
        ((32, 64), (2, 64, 32)),  # 2-D lhs against batched rhs: batch comes from rhs
    ],
)
def test_matmul(a_shape, b_shape) -> None:
    model = make_model(
        [helper.make_node("MatMul", ["a", "b"], ["y"])],
        [vi("a", list(a_shape)), vi("b", list(b_shape))],
        [vi("y", None)],
    )
    assert_tt_matches_cpu(
        model, {"a": randn(*a_shape), "b": randn(*b_shape)}, atol=0.1, rtol=0.1
    )
