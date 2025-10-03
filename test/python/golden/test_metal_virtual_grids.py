# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import sys
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only

from ttmlir.dialects import ttir, ttcore
from builder.base.builder import Operand, Shape, TypeInfo
from builder.base.builder_golden import BuilderGoldenTensor
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import (
    compile_ttir_to_flatbuffer,
    compile_and_execute_ttir,
)
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttir")


def create_tileid_debug_tensor(shape: Shape, dtype: torch.dtype):
    """Create a debug tensor where each value in a tile is the row-major tile ID"""
    # where value = (pos[1] / 32) * shape[0] + (pos[0] / 32)
    TILE_DIM_SIZE = int(32)
    ROW_STRIDE = shape[1] // TILE_DIM_SIZE
    y_coords, x_coords = torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij"
    )
    tile_y = y_coords // TILE_DIM_SIZE
    tile_x = x_coords // TILE_DIM_SIZE
    input_tensor = (tile_y * ROW_STRIDE + tile_x).float()
    return input_tensor.to(dtype)


@pytest.mark.parametrize(
    "shape",
    [(64, 4096), (8192, 32), (128, 4096), (256, 8192)],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_virtual_grid_eltwise(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def eltwise_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        input_tensor = create_tileid_debug_tensor(shape, dtype)

        # abs is an identity function for positive integers, so use it for debugging ease
        result = builder.abs(in0, unit_attrs=unit_attrs)

        golden_output_tensor = torch.abs(input_tensor).to(dtype)
        builder.set_goldens({in0: input_tensor}, {result: golden_output_tensor})

        return result

    compile_and_execute_ttir(
        eltwise_wrapper,
        [shape],
        [dtype],
        device=device,
        test_base=request.node.name,
        print_ir="test_logical_not_ir",
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
