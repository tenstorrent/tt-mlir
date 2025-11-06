# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from builder.base.builder import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_and_execute_shlo
from test_utils import shape_str
from wrappers.eltwise import *

pytestmark = pytest.mark.frontend("shlo")


unary_ops = [
    abs,
    ceil,
    cosine,
    exp,
    floor,
    log,
    logistic,
    neg,
    rsqrt,
    sine,
    sqrt,
    tan,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize("test_fn", unary_ops)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        add,
    ],
)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64), (64, 64)]], ids=["64x64"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_stablehlo_multi_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    def multi_return_model(
        in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder
    ):
        builder.set_graph_level_check(True)

        add_result = builder.add(in0, in1)
        exp_result = builder.exp(in2)
        sqrt_result = builder.sqrt(exp_result)

        return exp_result, sqrt_result

    compile_and_execute_shlo(
        multi_return_model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
