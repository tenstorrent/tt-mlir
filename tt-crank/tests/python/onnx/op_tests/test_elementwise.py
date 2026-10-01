# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Elementwise ops: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_crank.onnx as tt_onnx


@pytest.mark.parametrize("op", ["Add", "Sub", "Mul", "Div"])
def test_binary(op: str) -> None:
    model = tt_onnx.model(
        [tt_onnx.node(op, ["a", "b"], ["y"])],
        [tt_onnx.input("a", [32, 32]), tt_onnx.input("b", [32, 32])],
        [tt_onnx.output("y", [32, 32])],
    )
    b = tt_onnx.randn(32, 32)
    b[np.abs(b) < 0.1] = 1.0  # keep Div away from tiny divisors
    tt_onnx.assert_tt_matches_cpu(model, {"a": tt_onnx.randn(32, 32), "b": b})


def test_binary_broadcast() -> None:
    model = tt_onnx.model(
        [tt_onnx.node("Add", ["a", "b"], ["y"])],
        [tt_onnx.input("a", [4, 32, 32]), tt_onnx.input("b", [32])],
        [tt_onnx.output("y", [4, 32, 32])],
    )
    tt_onnx.assert_tt_matches_cpu(
        model, {"a": tt_onnx.randn(4, 32, 32), "b": tt_onnx.randn(32)}
    )


@pytest.mark.parametrize("op", ["Relu", "Sigmoid", "Exp", "Neg", "Log"])
def test_unary(op: str) -> None:
    model = tt_onnx.model(
        [tt_onnx.node(op, ["x"], ["y"])],
        [tt_onnx.input("x", [32, 32])],
        [tt_onnx.output("y", [32, 32])],
    )
    x = tt_onnx.randn(32, 32)
    if op == "Log":
        x = np.abs(x) + 0.1
    tt_onnx.assert_tt_matches_cpu(model, {"x": x})


@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (0.0, 6.0)])
def test_clip(lo: float, hi: float) -> None:
    # Bounds as scalar constant inputs (opset >= 11); (0, 6) is ReLU6.
    model = tt_onnx.model(
        [tt_onnx.node("Clip", ["x", "lo", "hi"], ["y"])],
        [tt_onnx.input("x", [1, 8, 16, 16])],
        [tt_onnx.output("y", None)],
        [
            tt_onnx.constant("lo", np.float32(lo)),
            tt_onnx.constant("hi", np.float32(hi)),
        ],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(1, 8, 16, 16) * 4})


def test_softmax() -> None:
    model = tt_onnx.model(
        [tt_onnx.node("Softmax", ["x"], ["y"], axis=-1)],
        [tt_onnx.input("x", [8, 64])],
        [tt_onnx.output("y", [8, 64])],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(8, 64) * 2})
