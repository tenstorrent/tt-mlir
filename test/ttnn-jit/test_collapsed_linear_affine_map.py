# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from ttnn_jit._src.utils import _get_collapsed_linear_affine_map
from ttmlir.ir import *


DIM_COLLAPSE_PARAMS = [
    ([2, 3, 64, 128], [1, 1], [(0, -1)], 0),
    ([1, 2, 32], [1, 1], [(0, -1)], 1)
]


def expected_map(index, rank, ctx):
    map_list = []
    if index == 0:
        map_list = [
            ((AffineDimExpr.get(0, ctx) * 192) + (AffineDimExpr.get(1, ctx) * 64)) + AffineDimExpr.get(2, ctx),
            AffineDimExpr.get(3, ctx)
        ]
    elif index == 1:
        map_list = [
            ((AffineDimExpr.get(0, ctx) * 2) + (AffineDimExpr.get(1, ctx))),
            AffineDimExpr.get(2, ctx)
        ]
    return AffineMap.get(rank, 0, map_list, ctx)


@pytest.mark.parametrize("shape, grid_shape, collapse_intervals, idx", DIM_COLLAPSE_PARAMS)
def test_collapsed_affine_map(shape, grid_shape, collapse_intervals, idx):

    context = Context()
    rank = len(shape)
    true_map = expected_map(idx, rank, context)

    result_map = _get_collapsed_linear_affine_map(context, shape, grid_shape, collapse_intervals)

    print("Expected Map: ", true_map)
    print("Result Map: ", result_map)

    assert result_map == true_map