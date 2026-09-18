# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MaxPool and GlobalAveragePool: tt EP vs CPU golden."""

import numpy as np
import pytest

from ort_ep_utils import assert_tt_matches_cpu, make_model, make_node, randn, vi


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
    model = make_model(
        [make_node("MaxPool", ["x"], ["y"], **attrs)],
        [vi("x", [1, 4, 16, 16])],
        [vi("y", None)],
    )
    # All-negative input: any zero-fill of padding would show up as a wrong maximum.
    assert_tt_matches_cpu(model, {"x": -np.abs(randn(1, 4, 16, 16)) - 0.5})


def test_global_average_pool() -> None:
    model = make_model(
        [make_node("GlobalAveragePool", ["x"], ["y"])],
        [vi("x", [2, 8, 16, 16])],
        [vi("y", None)],
    )
    assert_tt_matches_cpu(model, {"x": randn(2, 8, 16, 16)})
