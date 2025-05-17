# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape
from ttmlir.dialects import tt
from ttmlir.ir import *


@pytest.mark.parametrize("grid_y", [1, 2, 3])
@pytest.mark.parametrize("grid_x", [1, 2, 3])
@pytest.mark.parametrize("shard_mul_y", [3])
@pytest.mark.parametrize("shard_mul_x", [2])
@pytest.mark.parametrize("dst_register_size_tiles", [1])
def test_eltwise_blocking(
    grid_y: int,
    grid_x: int,
    shard_mul_y: int,
    shard_mul_x: int,
    dst_register_size_tiles: int,
    request,
):
    tile_size = 32
    shape = (
        grid_y * shard_mul_y * tile_size,
        grid_x * shard_mul_x * tile_size,
    )

    def eltwise_blocking(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.add(in0, in1, unit_attrs=unit_attrs)

    options = [
        f"max-dst-register-size-tiles={dst_register_size_tiles}",
        f"override-device-shape={grid_y},{grid_x}",
    ]
    compile_to_flatbuffer(
        eltwise_blocking,
        [shape, shape],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        print_ir="blocking_mlir",
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
