# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MaxPool and GlobalAveragePool: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_onnx


@pytest.mark.parametrize(
    "attrs",
    [
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
        {"kernel_shape": [2, 2], "pads": [1, 0, 0, 1]},  # asymmetric: -inf pre-pad
        {
            "kernel_shape": [3, 3],
            "strides": [3, 3],
            "pads": [2, 2, 2, 2],
        },  # pads > kernel/2: -inf pre-pad
    ],
)
def test_max_pool(attrs: dict) -> None:
    model = tt_onnx.make_model(
        [tt_onnx.make_node("MaxPool", ["x"], ["y"], **attrs)],
        [tt_onnx.vi("x", [1, 4, 16, 16])],
        [tt_onnx.vi("y", None)],
    )
    # All-negative input: any zero-fill of padding would show up as a wrong maximum.
    tt_onnx.assert_tt_matches_cpu(
        model, {"x": -np.abs(tt_onnx.randn(1, 4, 16, 16)) - 0.5}
    )


def test_global_average_pool() -> None:
    model = tt_onnx.make_model(
        [tt_onnx.make_node("GlobalAveragePool", ["x"], ["y"])],
        [tt_onnx.vi("x", [2, 8, 16, 16])],
        [tt_onnx.vi("y", None)],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(2, 8, 16, 16)})
