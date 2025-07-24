# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import List

from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder
from ttmlir.ir import *


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
    compile_to_flatbuffer(
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

    compile_to_flatbuffer(
        matmul,
        [lhs, rhs],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
