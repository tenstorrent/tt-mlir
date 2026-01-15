# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Callable, List, Optional, Tuple, Union

import ttmlir
from ttmlir.dialects import ttcore
from ttmlir.ir import *


def test_grid_attr():
    ctx = ttmlir.Context()
    shape = [1, 2, 3]
    grid_attr = ttcore.ir.GridAttr.get(ctx, shape)

    assert isinstance(grid_attr, ttcore.ir.GridAttr)
    assert list(grid_attr.shape) == shape


@pytest.mark.parametrize(
    "reduce_type",
    [
        ttcore.ir.ReduceType.Sum,
        ttcore.ir.ReduceType.Mean,
        ttcore.ir.ReduceType.Max,
        ttcore.ir.ReduceType.Min,
        ttcore.ir.ReduceType.Std,
        ttcore.ir.ReduceType.Var,
        ttcore.ir.ReduceType.Prod,
        ttcore.ir.ReduceType.Invalid,
    ],
)
def test_reduce_type_attr(reduce_type):
    ctx = ttmlir.Context()
    reduce_attr = ttcore.ir.ReduceTypeAttr.get(ctx, reduce_type)

    assert isinstance(reduce_attr, ttcore.ir.ReduceTypeAttr)
    assert reduce_attr.value == reduce_type


@pytest.mark.parametrize(
    "data_type",
    [
        ttcore.ir.DataType.Float32,
        ttcore.ir.DataType.Float16,
        ttcore.ir.DataType.BFloat16,
        ttcore.ir.DataType.BFP_Float8,
        ttcore.ir.DataType.BFP_BFloat8,
        ttcore.ir.DataType.BFP_Float4,
        ttcore.ir.DataType.BFP_BFloat4,
        ttcore.ir.DataType.BFP_Float2,
        ttcore.ir.DataType.BFP_BFloat2,
        ttcore.ir.DataType.UInt32,
        ttcore.ir.DataType.UInt16,
        ttcore.ir.DataType.UInt8,
        ttcore.ir.DataType.Int32,
        ttcore.ir.DataType.Bool,
    ],
)
def test_data_type_attr(data_type):
    ctx = ttmlir.Context()
    data_type_attr = ttcore.ir.DataTypeAttr.get(ctx, data_type)

    assert isinstance(data_type_attr, ttcore.ir.DataTypeAttr)
    assert data_type_attr.value == data_type


def test_tile_type():
    ctx = ttmlir.Context()
    height = 4
    width = 8
    data_type = ttcore.ir.DataTypeAttr.get(ctx, ttcore.ir.DataType.Float32)
    tile_type = ttcore.ir.TileType.get(ctx, height, width, data_type)

    assert isinstance(tile_type, ttcore.ir.TileType)
    assert tile_type.height == height
    assert tile_type.width == width

    downcasted_datatype = ttcore.ir.DataTypeAttr.from_attribute(tile_type.datatype)
    assert downcasted_datatype.value == ttcore.ir.DataType.Float32
