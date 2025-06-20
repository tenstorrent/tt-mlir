# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape
from ttmlir.ir import *


@pytest.mark.parametrize("grid_y", [1])
@pytest.mark.parametrize("grid_x", [1])
@pytest.mark.parametrize("shard_mul_y", [3])
@pytest.mark.parametrize("shard_mul_x", [2])
@pytest.mark.parametrize("dst_register_size_tiles", [8])
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
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("grid_m", [1])
@pytest.mark.parametrize("grid_k", [1])
@pytest.mark.parametrize("grid_n", [1])
@pytest.mark.parametrize("m", [3])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("n", [3])
@pytest.mark.parametrize("dst_register_size_tiles", [8])
def test_matmul_blocking(
    grid_m: int,
    grid_k: int,
    grid_n: int,
    m: int,
    k: int,
    n: int,
    dst_register_size_tiles: int,
    request,
):
    tile_size = 32
    lhs = (
        grid_m * m * tile_size,
        grid_k * k * tile_size,
    )
    rhs = (
        grid_k * k * tile_size,
        grid_n * n * tile_size,
    )

    def matmul_blocking(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    options = [
        f"max-dst-register-size-tiles={dst_register_size_tiles}",
        f"override-device-shape={grid_m},{grid_n}",
        "matmul-interchange=2,0,1",
    ]
    compile_to_flatbuffer(
        matmul_blocking,
        [lhs, rhs],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (512, 512, 512),
        (512, 1024, 1024),
        (512, 1024, 2048),
        (1024, 1024, 1024),
        (1024, 1024, 2048),
        (1024, 2048, 2048),
        (2048, 2048, 2048),
        (2048, 2048, 3072),
        (2048, 3072, 3072),
        (3072, 3072, 3072),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("dst_register_size_tiles", [8])
def test_matmul_ttnn(
    shape: tuple[int],
    dtype: torch.dtype,
    dst_register_size_tiles: int,
    request,
):
    lhs = (
        shape[0],
        shape[1],
    )
    rhs = (
        shape[1],
        shape[2],
    )

    def matmul_blocking(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    options = [
        f"max-dst-register-size-tiles={dst_register_size_tiles}",
        "matmul-interchange=2,0,1",
    ]
    compile_to_flatbuffer(
        matmul_blocking,
        [lhs, rhs],
        inputs_types=[dtype, dtype],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
