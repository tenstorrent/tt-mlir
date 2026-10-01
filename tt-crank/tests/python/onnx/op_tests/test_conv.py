# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Conv and BatchNormalization: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_crank.onnx as tt_onnx


@pytest.mark.parametrize(
    "attrs",
    [
        {"pads": [1, 1, 1, 1]},
        {"strides": [2, 2], "auto_pad": "SAME_UPPER"},
        {"pads": [1, 0, 2, 1]},  # asymmetric: explicit zero pre-pad
        {"group": 4, "pads": [1, 1, 1, 1]},
    ],
)
def test_conv2d(attrs: dict) -> None:
    groups = attrs.get("group", 1)
    model = tt_onnx.model(
        [tt_onnx.node("Conv", ["x", "w", "b"], ["y"], **attrs)],
        [tt_onnx.input("x", [1, 4, 16, 16])],
        [tt_onnx.output("y", None)],
        [
            tt_onnx.constant("w", tt_onnx.randn(8, 4 // groups, 3, 3)),
            tt_onnx.constant("b", tt_onnx.randn(8)),
        ],
    )
    tt_onnx.assert_tt_matches_cpu(
        model, {"x": tt_onnx.randn(1, 4, 16, 16)}, atol=0.1, rtol=0.1
    )


def test_batch_norm() -> None:
    model = tt_onnx.model(
        [
            tt_onnx.node(
                "BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], epsilon=1e-3
            )
        ],
        [tt_onnx.input("x", [2, 8, 16, 16])],
        [tt_onnx.output("y", None)],
        [
            tt_onnx.constant("s", tt_onnx.randn(8)),
            tt_onnx.constant("b", tt_onnx.randn(8)),
            tt_onnx.constant("m", tt_onnx.randn(8)),
            tt_onnx.constant("v", np.abs(tt_onnx.randn(8)) + 0.5),
        ],
    )
    tt_onnx.assert_tt_matches_cpu(
        model, {"x": tt_onnx.randn(2, 8, 16, 16)}, atol=5e-2, rtol=5e-2
    )
