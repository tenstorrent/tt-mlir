# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
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
    """Create a debug tensor where each value in a 32x32 tile equals its row-major tile ID.

    The tile grid is computed over the last two dimensions; any leading dimensions
    are broadcast so that all slices share the same 2D tile ID pattern.
    """
    TILE_DIM_SIZE = int(32)
    assert len(shape) >= 2, "Shape must have at least 2 dimensions."
    height, width = int(shape[-2]), int(shape[-1])
    # Number of tiles per row in the last dimension.
    row_stride_tiles = width // TILE_DIM_SIZE

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    tile_y = y_coords // TILE_DIM_SIZE
    tile_x = x_coords // TILE_DIM_SIZE
    base_2d = (tile_y * row_stride_tiles + tile_x).to(dtype)

    # Broadcast across any leading dimensions.
    if len(shape) == 2:
        return base_2d
    expand_shape = tuple(int(d) for d in shape[:-2]) + (height, width)
    base_nd = base_2d
    for _ in range(len(shape) - 2):
        base_nd = base_nd.unsqueeze(0)
    return base_nd.expand(expand_shape)


@pytest.mark.skip_config(["p150"], ["p300"], reason="See issue #6248")
@pytest.mark.parametrize(
    "shape",
    [
        (32, 4096),
        (4096, 32),
        (2048, 32),
        (1, 1, 1, 1, 128, 128),
        (1, 1, 1, 1, 2, 32, 512),
        (1, 1, 1, 1, 32, 32),
        (1, 1, 1, 4, 128, 256),
        (1, 1, 2, 1, 1, 512, 64),
        (1, 1, 32, 128),
        (1, 1, 32, 32),
        (1, 1, 64, 32),
        (1, 2, 1, 1, 1, 128, 32),
        (1, 2, 1, 1, 2, 1024, 32),
        (1, 2, 1, 4, 128, 32),
        (1, 2, 1, 4, 32, 64),
        (1, 2, 256, 128),
        (1, 4, 2, 1, 128, 32),
        (1, 512, 32),
        (2, 1, 1, 1, 1, 256, 256),
        (2, 1, 2, 1, 512, 32),
        (2, 1, 2, 256, 32),
        (2, 1, 256, 256),
        (2, 1, 4, 64, 64),
        (2, 2, 1, 2, 128, 64),
        (2, 32, 64),
        (2, 4, 4, 1, 2, 1, 32, 32),
        (32, 32, 128),
        (32, 32, 32),
        (4, 1, 2, 2, 1, 64, 128),
        (4, 2, 1, 2, 1, 64, 32),
        (4, 2, 32, 512),
        (4, 256, 128),
        (4, 32, 32),
        (4, 64, 64),
        (8, 1, 512, 32),
    ],  # (64, 4096), (8192, 32), (128, 4096), (256, 8192)],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_virtual_grid_eltwise(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def eltwise_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            input_tensor = create_tileid_debug_tensor(shape, dtype)

            # abs is an identity function for positive integers, so use it for debugging ease
            result = builder.abs(in0, unit_attrs=unit_attrs)

            golden_output_tensor = torch.abs(input_tensor).to(dtype)
            builder.set_goldens({in0: input_tensor}, {result: golden_output_tensor})

            return result

    # device shape override is needed so that shapes are equivalently divisible
    # on both WH and BH.
    options = [f"collapse-tensors-2d=0", "override-device-shape=8,8"]

    compile_and_execute_ttir(
        module,
        device=device,
        test_base=request.node.name,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}} ",
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
