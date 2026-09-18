# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ReduceMean: tt EP vs CPU golden."""

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


@pytest.mark.parametrize("keepdims", [0, 1])
def test_reduce_mean_axes_input(keepdims: int) -> None:
    # opset >= 18: axes are an int64 constant input.
    model = make_model(
        [make_node("ReduceMean", ["x", "axes"], ["y"], keepdims=keepdims)],
        [vi("x", [2, 8, 4, 4])],
        [vi("y", None)],
        [make_tensor("axes", np.array([-1, -2], dtype=np.int64))],
    )
    assert_tt_matches_cpu(model, {"x": randn(2, 8, 4, 4)})


def test_reduce_mean_axes_attr() -> None:
    # opset < 18: axes are an attribute.
    model = make_model(
        [make_node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=1)],
        [vi("x", [2, 8, 4, 4])],
        [vi("y", None)],
        opset=13,
    )
    assert_tt_matches_cpu(model, {"x": randn(2, 8, 4, 4)})
