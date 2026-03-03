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
from conftest import x86_only, get_request_kwargs

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
    """Create a debug tensor where each value in a 32x32 tile equals its multi-dimensional tile ID.

    The tile grid is computed over the last two dimensions (32x32 tiles); outer dimensions get unique
    tile coordinates but tile extent is only 32x32 in the inner dimensions. The scalar tile ID uses
    row-major ordering across the full multi-dimensional tile grid.
    """
    TILE_DIM_SIZE = int(32)
    assert len(shape) >= 2, "Shape must have at least 2 dimensions."
    shape = tuple(int(d) for d in shape)
    ndim = len(shape)

    # Compute number of tiles along each dimension (ceil division for partial tiles)
    num_tiles_per_dim = [(dim + TILE_DIM_SIZE - 1) // TILE_DIM_SIZE for dim in shape]

    # Generate meshgrids for tile coordinates across all dimensions
    tile_coords_list = []
    for i in range(ndim):
        dim_indices = torch.arange(shape[i])
        # For outer dims (i < ndim-2), use full dimension indexing as "tile coords"
        # For inner two dims (i >= ndim-2), use 32-sized tile indexing
        if i >= ndim - 2:
            tile_coord = dim_indices // TILE_DIM_SIZE
        else:
            tile_coord = (
                dim_indices  # Each element in outer dim is its own "tile coord"
            )
        tile_coords_list.append(tile_coord)

    # Compute the meshgrid of tile coordinates across all dimensions
    tile_coords = torch.meshgrid(*tile_coords_list, indexing="ij")

    # Compute the scalar tile ID using row-major ordering across the full grid
    tile_id = torch.zeros_like(tile_coords[0])
    stride = 1
    for i in reversed(range(ndim)):
        tile_id += tile_coords[i] * stride
        stride *= num_tiles_per_dim[i]

    return tile_id.to(dtype)


@pytest.mark.skip_config(["p150"], ["p300"], reason="See issue #6248")
@pytest.mark.parametrize(
    "shape",
    [
        (32, 4096),
        (4096, 32),
        (2048, 32),
        (32, 1280),  # uses 1x40 grid
        (1536, 64),  # uses 48x1 grid
        (1120, 32),  # uses 35x1 grid
        (32, 768),  # uses 1x24 grid
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
    ],
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
            # input_tensor = create_tileid_debug_tensor(shape, dtype)
            input_tensor = torch.randint(0, 1001, shape, dtype=dtype)

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
        **get_request_kwargs(request),
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}} ",
        target=target,
    )
