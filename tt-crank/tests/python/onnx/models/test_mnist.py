# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MNIST-style MLP (Gemm -> Relu -> Gemm) on the TT EP, the ONNX twin of torch/models/test_mnist_forward.py."""

import pytest

import tt_crank.onnx as tt_onnx


def _mnist_linear(feat: int, hidden: int, classes: int) -> bytes:
    return tt_onnx.model(
        [
            tt_onnx.node("Gemm", ["x", "w1", "b1"], ["h"], transB=1),
            tt_onnx.node("Relu", ["h"], ["r"]),
            tt_onnx.node("Gemm", ["r", "w2", "b2"], ["y"], transB=1),
        ],
        [tt_onnx.input("x", ["N", feat])],
        [tt_onnx.output("y", ["N", classes])],
        [
            tt_onnx.constant("w1", tt_onnx.randn(hidden, feat) / feat**0.5),
            tt_onnx.constant("b1", tt_onnx.randn(hidden) / 10),
            tt_onnx.constant("w2", tt_onnx.randn(classes, hidden) / hidden**0.5),
            tt_onnx.constant("b2", tt_onnx.randn(classes) / 10),
        ],
    )


@pytest.mark.parametrize(
    "batch,feat,hidden,classes",
    [
        (32, 32 * 32, 128, 32),  # tile-aligned baseline
        (64, 28 * 28, 128, 10),  # real MNIST shapes
        (1, 28 * 28, 128, 10),  # batch=1 single-sample inference
    ],
    ids=["tile_aligned", "real_mnist_batch64", "real_mnist_batch1"],
)
def test_mnist_forward(batch: int, feat: int, hidden: int, classes: int) -> None:
    model = _mnist_linear(feat, hidden, classes)
    tt_onnx.assert_tt_matches_cpu(
        model, {"x": tt_onnx.randn(batch, feat)}, free_dims={"N": batch}
    )
