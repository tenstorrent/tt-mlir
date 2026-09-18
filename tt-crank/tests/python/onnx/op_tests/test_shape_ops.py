# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reshape, Transpose, Flatten, Concat, Identity: tt EP vs CPU golden."""

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


@pytest.mark.parametrize("target", [[8, -1, 32], [4, 8, 32], [0, -1]])
def test_reshape(target) -> None:
    # Constant target incl. ONNX's -1 (infer) and 0 (copy the input dim).
    model = make_model(
        [make_node("Reshape", ["x", "shape"], ["y"])],
        [vi("x", [4, 8, 32])],
        [vi("y", None)],
        [make_tensor("shape", np.array(target, dtype=np.int64))],
    )
    assert_tt_matches_cpu(model, {"x": randn(4, 8, 32)})


def test_transpose_flatten_identity() -> None:
    model = make_model(
        [
            make_node("Transpose", ["x"], ["t"], perm=[1, 0, 2]),
            make_node("Flatten", ["t"], ["f"], axis=1),
            make_node("Identity", ["f"], ["y"]),
        ],
        [vi("x", [4, 8, 32])],
        [vi("y", None)],
    )
    assert_tt_matches_cpu(model, {"x": randn(4, 8, 32)})


def test_concat() -> None:
    model = make_model(
        [make_node("Concat", ["a", "b", "c"], ["y"], axis=1)],
        [vi(n, [4, 8]) for n in "abc"],
        [vi("y", None)],
    )
    assert_tt_matches_cpu(model, {n: randn(4, 8) for n in "abc"})


@pytest.mark.xfail(strict=True, reason="Squeeze/Unsqueeze not implemented")
def test_squeeze_unsqueeze() -> None:
    model = make_model(
        [
            make_node("Unsqueeze", ["x", "axes"], ["u"]),
            make_node("Squeeze", ["u", "axes"], ["y"]),
        ],
        [vi("x", [4, 8, 32])],
        [vi("y", None)],
        [make_tensor("axes", np.array([0], dtype=np.int64))],
    )
    assert_tt_matches_cpu(model, {"x": randn(4, 8, 32)})


@pytest.mark.xfail(
    strict=True, reason="Squeeze not implemented (opset-9 attribute forms)"
)
def test_old_opset_attribute_forms() -> None:
    # Opset 9: Clip bounds and Squeeze axes are attributes; a bound-less Clip is an identity.
    model = make_model(
        [
            make_node("Clip", ["x"], ["c"], min=-0.5, max=0.5),
            make_node("Clip", ["c"], ["c2"]),
            make_node("Squeeze", ["c2"], ["y"], axes=[0]),
        ],
        [vi("x", [1, 16, 16])],
        [vi("y", None)],
        opset=9,
    )
    assert_tt_matches_cpu(model, {"x": randn(1, 16, 16) * 2})
