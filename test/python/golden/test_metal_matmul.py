# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer


@pytest.mark.fails_golden
@pytest.mark.parametrize("m", [2])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("n", [4])
# Single core matmuls, 8 output tiles per core max
def test_matmul_single_core_8otpc(
    m: int,
    k: int,
    n: int,
    request,
):
    tile_size = 32
    lhs = (
        m * tile_size,
        k * tile_size,
    )
    rhs = (
        k * tile_size,
        n * tile_size,
    )

    def matmul(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    options = [f"override-device-shape=1,1"]
    compile_ttir_to_flatbuffer(
        matmul,
        [lhs, rhs],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("m", [3, 6, 9])
@pytest.mark.parametrize("k", [4])
@pytest.mark.parametrize("n", [3, 6])
# Multi core matmuls, 8 output tiles per core max
def test_matmul_multi_core_8otpc(
    m: int,
    k: int,
    n: int,
    request,
):
    tile_size = 32
    lhs = (
        m * tile_size,
        k * tile_size,
    )
    rhs = (
        k * tile_size,
        n * tile_size,
    )

    def matmul(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        matmul,
        [lhs, rhs],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
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
    ],
)
@pytest.mark.parametrize("dst_register_size_tiles", [8])
# Large matmuls, based on ttnn's matmul benchmarks
def test_matmul_ttnn_shapes(
    shape: tuple[int],
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
        f"matmul-interchange=2,0,1",
    ]
    compile_ttir_to_flatbuffer(
        matmul_blocking,
        [lhs, rhs],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
