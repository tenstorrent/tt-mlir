# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Elementwise ops: tt EP vs CPU golden."""

import numpy as np
import pytest

from ort_ep_utils import (
    assert_tt_matches_cpu,
    make_model,
    make_node,
    make_tensor,
    randn,
    vi,
)


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div"])
def test_binary(op: str) -> None:
    model = make_model(
        [make_node(op, ["a", "b"], ["y"])],
        [vi("a", [32, 32]), vi("b", [32, 32])],
        [vi("y", [32, 32])],
    )
    b = randn(32, 32)
    b[np.abs(b) < 0.1] = 1.0  # keep Div away from tiny divisors
    assert_tt_matches_cpu(model, {"a": randn(32, 32), "b": b})


def test_binary_broadcast() -> None:
    model = make_model(
        [make_node("Add", ["a", "b"], ["y"])],
        [vi("a", [4, 32, 32]), vi("b", [32])],
        [vi("y", [4, 32, 32])],
    )
    assert_tt_matches_cpu(model, {"a": randn(4, 32, 32), "b": randn(32)})


@pytest.mark.parametrize("op", ["Relu", "Sigmoid", "Exp", "Neg", "Log"])
def test_unary(op: str) -> None:
    model = make_model(
        [make_node(op, ["x"], ["y"])], [vi("x", [32, 32])], [vi("y", [32, 32])]
    )
    x = randn(32, 32)
    if op == "Log":
        x = np.abs(x) + 0.1
    assert_tt_matches_cpu(model, {"x": x})


@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (0.0, 6.0)])
def test_clip(lo: float, hi: float) -> None:
    # Bounds as scalar constant inputs (opset >= 11); (0, 6) is ReLU6.
    model = make_model(
        [make_node("Clip", ["x", "lo", "hi"], ["y"])],
        [vi("x", [1, 8, 16, 16])],
        [vi("y", None)],
        [make_tensor("lo", np.float32(lo)), make_tensor("hi", np.float32(hi))],
    )
    assert_tt_matches_cpu(model, {"x": randn(1, 8, 16, 16) * 4})


def test_softmax() -> None:
    model = make_model(
        [make_node("Softmax", ["x"], ["y"], axis=-1)],
        [vi("x", [8, 64])],
        [vi("y", [8, 64])],
    )
    assert_tt_matches_cpu(model, {"x": randn(8, 64) * 2})
