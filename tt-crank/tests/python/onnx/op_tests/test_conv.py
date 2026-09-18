# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Conv and BatchNormalization: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_onnx


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
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Conv", ["x", "w", "b"], ["y"], **attrs)],
        [tt_onnx.vi("x", [1, 4, 16, 16])],
        [tt_onnx.vi("y", None)],
        [
            tt_onnx.make_tensor("w", tt_onnx.randn(8, 4 // groups, 3, 3)),
            tt_onnx.make_tensor("b", tt_onnx.randn(8)),
        ],
    )
    tt_onnx.assert_tt_matches_cpu(
        model, {"x": tt_onnx.randn(1, 4, 16, 16)}, atol=0.1, rtol=0.1
    )


def test_batch_norm() -> None:
    model = tt_onnx.make_model(
        [
            tt_onnx.make_node(
                "BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], epsilon=1e-3
            )
        ],
        [tt_onnx.vi("x", [2, 8, 16, 16])],
        [tt_onnx.vi("y", None)],
        [
            tt_onnx.make_tensor("s", tt_onnx.randn(8)),
            tt_onnx.make_tensor("b", tt_onnx.randn(8)),
            tt_onnx.make_tensor("m", tt_onnx.randn(8)),
            tt_onnx.make_tensor("v", np.abs(tt_onnx.randn(8)) + 0.5),
        ],
    )
    tt_onnx.assert_tt_matches_cpu(
        model, {"x": tt_onnx.randn(2, 8, 16, 16)}, atol=5e-2, rtol=5e-2
    )
