# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reshape, Transpose, Flatten, Concat, Identity: tt EP vs CPU golden."""

import numpy as np
import pytest

import tt_onnx


@pytest.mark.parametrize("target", [[8, -1, 32], [4, 8, 32], [0, -1]])
def test_reshape(target) -> None:
    # Constant target incl. ONNX's -1 (infer) and 0 (copy the input dim).
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Reshape", ["x", "shape"], ["y"])],
        [tt_onnx.vi("x", [4, 8, 32])],
        [tt_onnx.vi("y", None)],
        [tt_onnx.make_tensor("shape", np.array(target, dtype=np.int64))],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(4, 8, 32)})


def test_transpose_flatten_identity() -> None:
    model = tt_onnx.make_model(
        [
            tt_onnx.make_node("Transpose", ["x"], ["t"], perm=[1, 0, 2]),
            tt_onnx.make_node("Flatten", ["t"], ["f"], axis=1),
            tt_onnx.make_node("Identity", ["f"], ["y"]),
        ],
        [tt_onnx.vi("x", [4, 8, 32])],
        [tt_onnx.vi("y", None)],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(4, 8, 32)})


def test_concat() -> None:
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Concat", ["a", "b", "c"], ["y"], axis=1)],
        [tt_onnx.vi(n, [4, 8]) for n in "abc"],
        [tt_onnx.vi("y", None)],
    )
    tt_onnx.assert_tt_matches_cpu(model, {n: tt_onnx.randn(4, 8) for n in "abc"})


@pytest.mark.xfail(strict=True, reason="Squeeze/Unsqueeze not implemented")
def test_squeeze_unsqueeze() -> None:
    model = tt_onnx.make_model(
        [
            tt_onnx.make_node("Unsqueeze", ["x", "axes"], ["u"]),
            tt_onnx.make_node("Squeeze", ["u", "axes"], ["y"]),
        ],
        [tt_onnx.vi("x", [4, 8, 32])],
        [tt_onnx.vi("y", None)],
        [tt_onnx.make_tensor("axes", np.array([0], dtype=np.int64))],
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(4, 8, 32)})


@pytest.mark.xfail(
    strict=True, reason="Squeeze not implemented (opset-9 attribute forms)"
)
def test_old_opset_attribute_forms() -> None:
    # Opset 9: Clip bounds and Squeeze axes are attributes; a bound-less Clip is an identity.
    model = tt_onnx.make_model(
        [
            tt_onnx.make_node("Clip", ["x"], ["c"], min=-0.5, max=0.5),
            tt_onnx.make_node("Clip", ["c"], ["c2"]),
            tt_onnx.make_node("Squeeze", ["c2"], ["y"], axes=[0]),
        ],
        [tt_onnx.vi("x", [1, 16, 16])],
        [tt_onnx.vi("y", None)],
        opset=9,
    )
    tt_onnx.assert_tt_matches_cpu(model, {"x": tt_onnx.randn(1, 16, 16) * 2})
