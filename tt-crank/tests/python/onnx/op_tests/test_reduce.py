# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ReduceMean: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_crank.onnx as tt_onnx


@pytest.mark.parametrize("keepdims", [0, 1])
def test_reduce_mean_axes_input(keepdims: int) -> None:
    # opset >= 18: axes are an int64 constant input.
    model = tt_onnx.make_model(
        [tt_onnx.make_node("ReduceMean", ["x", "axes"], ["y"], keepdims=keepdims)],
        [tt_onnx.vi("x", [2, 8, 4, 4])],
        [tt_onnx.vi("y", None)],
        [tt_onnx.make_tensor("axes", np.array([-1, -2], dtype=np.int64))],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(2, 8, 4, 4)})


def test_reduce_mean_axes_attr() -> None:
    # opset < 18: axes are an attribute.
    model = tt_onnx.make_model(
        [tt_onnx.make_node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=1)],
        [tt_onnx.vi("x", [2, 8, 4, 4])],
        [tt_onnx.vi("y", None)],
        opset=13,
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(2, 8, 4, 4)})
