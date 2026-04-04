# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for stablehlo.dynamic_update_slice lowered to ttir.slice_write.
"""

import pytest
import torch
from typing import List

from builder.base.builder_utils import Operand, Shape
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_apis import compile_and_execute_shlo
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("shlo")


SHAPES_BF16 = [
    # 5D bf16
    ((3, 1, 24, 64, 32), (1, 1, 24, 64, 32), [2, 0, 0, 0, 0]),
    ((3, 4, 12, 64, 32), (1, 4, 12, 64, 32), [2, 0, 0, 0, 0]),
    ((3, 16, 6, 64, 32), (1, 16, 6, 64, 32), [0, 0, 0, 0, 0]),
    ((3, 64, 3, 64, 32), (1, 64, 3, 64, 32), [1, 0, 0, 0, 0]),
    ((3, 1, 24, 49, 32), (1, 1, 24, 49, 32), [2, 0, 0, 0, 0]),
    ((3, 4, 12, 49, 32), (1, 4, 12, 49, 32), [1, 0, 0, 0, 0]),
    ((3, 16, 6, 49, 32), (1, 16, 6, 49, 32), [0, 0, 0, 0, 0]),
    ((3, 64, 3, 49, 32), (1, 64, 3, 49, 32), [2, 0, 0, 0, 0]),
    ((3, 1, 32, 49, 32), (1, 1, 32, 49, 32), [1, 0, 0, 0, 0]),
    ((3, 4, 16, 49, 32), (1, 4, 16, 49, 32), [2, 0, 0, 0, 0]),
    ((3, 16, 8, 49, 32), (1, 16, 8, 49, 32), [1, 0, 0, 0, 0]),
    ((3, 64, 4, 49, 32), (1, 64, 4, 49, 32), [0, 0, 0, 0, 0]),
    ((3, 1, 32, 64, 32), (1, 1, 32, 64, 32), [2, 0, 0, 0, 0]),
    ((3, 4, 16, 64, 32), (1, 4, 16, 64, 32), [1, 0, 0, 0, 0]),
    ((3, 16, 8, 64, 32), (1, 16, 8, 64, 32), [2, 0, 0, 0, 0]),
    ((3, 64, 4, 64, 32), (1, 64, 4, 64, 32), [0, 0, 0, 0, 0]),
    # 3D bf16
    ((1, 1370, 1280), (1, 1, 1280), [0, 500, 0]),
    ((1, 5, 2), (1, 1, 2), [0, 3, 0]),
    ((1, 197, 1024), (1, 1, 1024), [0, 100, 0]),
    ((1, 197, 768), (1, 1, 768), [0, 196, 0]),
    ((1, 50, 768), (1, 1, 768), [0, 25, 0]),
    ((1, 50, 1024), (1, 1, 1024), [0, 49, 0]),
    ((1, 197, 192), (1, 1, 192), [0, 50, 0]),
    ((1, 197, 384), (1, 1, 384), [0, 150, 0]),
    ((1, 128, 768), (1, 1, 768), [0, 64, 0]),
    ((1, 9, 768), (1, 1, 768), [0, 7, 0]),
    ((1, 11, 2), (1, 1, 2), [0, 10, 0]),
    ((1, 6, 16), (1, 1, 16), [0, 5, 0]),
    ((1, 3, 1024), (1, 1, 1024), [0, 2, 0]),
    # 4D bf16
    ((3, 1370, 1, 1280), (1, 1370, 1, 1280), [2, 0, 0, 0]),
    ((3, 197, 1, 1024), (1, 197, 1, 1024), [1, 0, 0, 0]),
    ((3, 197, 1, 768), (1, 197, 1, 768), [2, 0, 0, 0]),
    ((3, 50, 1, 768), (1, 50, 1, 768), [1, 0, 0, 0]),
    ((3, 50, 1, 1024), (1, 50, 1, 1024), [0, 0, 0, 0]),
    # Multi-dim start indices — update doesn't start at 0 in multiple dims
    ((2, 16, 32), (1, 8, 16), [1, 4, 10]),
    ((8, 8, 8), (2, 2, 2), [3, 5, 1]),
    ((1, 64, 64), (1, 32, 32), [0, 16, 16]),
]

SHAPES_F32 = [
    # 4D f32
    ((1, 3072, 6, 16), (1, 3072, 1, 16), [0, 0, 4, 0]),
    ((1, 4096, 6, 16), (1, 4096, 1, 16), [0, 0, 5, 0]),
    ((1, 2048, 6, 16), (1, 2048, 1, 16), [0, 0, 2, 0]),
    # Multi-dim f32
    ((4, 16, 32), (2, 8, 16), [1, 4, 8]),
]


def _shape_id(shape_tuple):
    op_shape, _, starts = shape_tuple
    return (
        f"{'x'.join(str(d) for d in op_shape)}" f"_at{'_'.join(str(s) for s in starts)}"
    )


@pytest.mark.parametrize(
    "operand_shape,update_shape,start_indices_val",
    SHAPES_BF16,
    ids=[_shape_id(s) for s in SHAPES_BF16],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_dynamic_update_slice_bf16(
    operand_shape: Shape,
    update_shape: Shape,
    start_indices_val: List[int],
    target: str,
    request,
    device,
):
    dtype = torch.bfloat16

    def module(builder: StableHLOBuilder):
        @builder.func([operand_shape, update_shape], [dtype, dtype])
        def dynamic_update_slice(in0: Operand, in1: Operand, builder: StableHLOBuilder):
            builder.set_graph_level_check(True)
            start_indices = [
                builder.constant(torch.tensor(idx, dtype=torch.int64))
                for idx in start_indices_val
            ]
            return builder.dynamic_update_slice(in0, in1, start_indices)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pcc=0.99,
        print_ir=False,
    )


@pytest.mark.parametrize(
    "operand_shape,update_shape,start_indices_val",
    SHAPES_F32,
    ids=[_shape_id(s) for s in SHAPES_F32],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_dynamic_update_slice_f32(
    operand_shape: Shape,
    update_shape: Shape,
    start_indices_val: List[int],
    target: str,
    request,
    device,
):
    dtype = torch.float32

    def module(builder: StableHLOBuilder):
        @builder.func([operand_shape, update_shape], [dtype, dtype])
        def dynamic_update_slice(in0: Operand, in1: Operand, builder: StableHLOBuilder):
            builder.set_graph_level_check(True)
            start_indices = [
                builder.constant(torch.tensor(idx, dtype=torch.int64))
                for idx in start_indices_val
            ]
            return builder.dynamic_update_slice(in0, in1, start_indices)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pcc=0.99,
        print_ir=False,
    )
