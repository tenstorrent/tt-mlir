# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Slice and Gather: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_crank.onnx as tt_onnx


def _i64(name: str, values) -> object:
    return tt_onnx.constant(name, np.array(values, dtype=np.int64))


@pytest.mark.parametrize(
    "starts,ends,axes,steps",
    [
        ([0], [1], [1], None),
        ([1], [3], [2], [1]),
        ([0], [2**31], [3], [2]),  # open-ended, step 2
        ([-8], [-1], [1], None),  # negative start/end
    ],
)
def test_slice(starts, ends, axes, steps) -> None:
    inputs = ["x", "starts", "ends", "axes"]
    initializers = [_i64("starts", starts), _i64("ends", ends), _i64("axes", axes)]
    if steps is not None:
        inputs.append("steps")
        initializers.append(_i64("steps", steps))
    model = tt_onnx.model(
        [tt_onnx.node("Slice", inputs, ["y"])],
        [tt_onnx.input("x", [1, 16, 16, 16])],
        [tt_onnx.output("y", None)],
        initializers,
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(1, 16, 16, 16)})


@pytest.mark.parametrize("axis,index", [(3, 0), (1, 5), (2, -1)])
def test_gather_scalar_index(axis: int, index: int) -> None:
    model = tt_onnx.model(
        [tt_onnx.node("Gather", ["x", "idx"], ["y"], axis=axis)],
        [tt_onnx.input("x", [1, 8, 16, 12])],
        [tt_onnx.output("y", None)],
        [_i64("idx", index)],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(1, 8, 16, 12)})
