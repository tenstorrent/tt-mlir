# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only

from builder.base.builder import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_stablehlo_to_flatbuffer
from test_utils import Marks, shape_str

pytestmark = pytest.mark.frontend("shlo")


def add(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.add(in0, in1, unit_attrs=unit_attrs)


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
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    compile_stablehlo_to_flatbuffer(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )

@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64), (64, 64)]], ids=["64x64"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_stablehlo_multi_return_support(shapes: List[Shape], dtypes: List[torch.dtype], target: str, request):
    def multi_return_model(in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder):
        builder.set_graph_level_check(True)

        add_result = builder.add(in0, in1)
        exp_result = builder.exp(in2)
        mult_result = builder.multiply(add_result, exp_result)
        
        return exp_result, mult_result
    
    compile_stablehlo_to_flatbuffer(
        multi_return_model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
