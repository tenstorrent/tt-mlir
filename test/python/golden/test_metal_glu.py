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
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_simple_subgraph(
    shape: tuple[int, ...],
    target: str,
    request,
):
    def simple_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        
        out1 = builder.sigmoid(x, unit_attrs=unit_attrs)
        return builder.exp(out1, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        simple_subgraph,
        [shape],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

@pytest.mark.fails_golden
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_swish1_subgraph(
    shape: tuple[int, ...],
    target: str,
    request,
):
    def swish1_subgraph(
        x: Operand,
        ones: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        custom_init = lambda s: torch.full(shape, 1).to(torch.bfloat16)
        builder.set_goldens({ones: custom_init})
        
        out1 = builder.neg(x, unit_attrs=unit_attrs)
        out2 = builder.exp(out1, unit_attrs=unit_attrs)
        out3 = builder.add(out2, ones, unit_attrs=unit_attrs)
        sig = builder.reciprocal(out3, unit_attrs=unit_attrs)
        # builder.multiply(x, sig, unit_attrs=unit_attrs)
        
        return sig

    options = []
    compile_ttir_to_flatbuffer(
        swish1_subgraph,
        [shape, shape],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

@pytest.mark.fails_golden
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_swish2_subgraph(
    shape: tuple[int, ...],
    target: str,
    request,
):
    def swish2_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        sig = builder.sigmoid(x, unit_attrs=unit_attrs)
        return builder.multiply(x, sig, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        swish2_subgraph,
        [shape],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize(
    "shape",
    [
        (512, 512, 512),
    ],
)
@pytest.mark.parametrize("dst_register_size_tiles", [8])
@pytest.mark.parametrize("use_tile_matmul", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_SwiGLU_ffn(
    shape: tuple[int, ...],
    dst_register_size_tiles: int,
    use_tile_matmul: bool,
    target: str,
    request,
):
    batch_size, dim, hidden_dim = shape 
    x = (batch_size, dim)
    w1 = (dim, hidden_dim)
    w2 = (dim, hidden_dim)

    def subgraph(
        x: Operand,
        w1: Operand,
        w2: Operand,
        w3: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        out1 = builder.matmul(w1, x, unit_attrs=unit_attrs)
        # s = SILU(out1) = x * sigmoid(x)
        s = builder.multiply(out1, builder.sigmoid(out1, unit_attrs=unit_attrs), unit_attrs=unit_attrs)
        out2 = builder.matmul(w2, x, unit_attrs=unit_attrs)
        return builder.multiply(s, out2, unit_attrs=unit_attrs)

    options = [
        #f"max-dst-register-size-tiles={dst_register_size_tiles}",
        f"matmul-interchange=2,0,1",
        f"num-stream-buffers=1",
        f"use-tile-matmul={use_tile_matmul}",
    ]
    compile_ttir_to_flatbuffer(
        subgraph,
        [x, w1, w2, w3],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

# add glu, reglu, geglu