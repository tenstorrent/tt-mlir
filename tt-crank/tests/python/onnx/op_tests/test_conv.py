# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Conv and BatchNormalization: tt EP vs CPU golden."""

import numpy as np
import pytest
from onnx import helper

from ort_ep_utils import assert_tt_matches_cpu, make_model, make_tensor, randn, vi


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
    model = make_model(
        [helper.make_node("Conv", ["x", "w", "b"], ["y"], **attrs)],
        [vi("x", [1, 4, 16, 16])],
        [vi("y", None)],
        [make_tensor("w", randn(8, 4 // groups, 3, 3)), make_tensor("b", randn(8))],
    )
    assert_tt_matches_cpu(model, {"x": randn(1, 4, 16, 16)}, atol=0.1, rtol=0.1)


def test_batch_norm() -> None:
    model = make_model(
        [
            helper.make_node(
                "BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], epsilon=1e-3
            )
        ],
        [vi("x", [2, 8, 16, 16])],
        [vi("y", None)],
        [
            make_tensor("s", randn(8)),
            make_tensor("b", randn(8)),
            make_tensor("m", randn(8)),
            make_tensor("v", np.abs(randn(8)) + 0.5),
        ],
    )
    assert_tt_matches_cpu(model, {"x": randn(2, 8, 16, 16)}, atol=5e-2, rtol=5e-2)
