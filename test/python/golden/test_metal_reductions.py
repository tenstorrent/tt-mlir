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

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.fails_golden
@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sum(
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    request,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    def reduce_sum(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.sum(
            in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=unit_attrs
        )

    compile_ttir_to_flatbuffer(
        reduce_sum,
        [shape],
        target=target,
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [0, 1])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_max(
    m: int,
    n: int,
    dim_arg: int,
    keep_dim: bool,
    target: str,
    request,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    def reduce_max(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return builder.max(
            in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=unit_attrs
        )

    compile_ttir_to_flatbuffer(
        reduce_max,
        [shape],
        target=target,
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
