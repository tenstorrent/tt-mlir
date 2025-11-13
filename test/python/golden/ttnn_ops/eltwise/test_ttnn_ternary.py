# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only
from builder.base.builder import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base import get_golden_function
from builder.base.builder_utils import (
    compile_and_execute_ttnn,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
)
from ttmlir.dialects import ttnn

pytestmark = pytest.mark.frontend("ttnn")


def where(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.where(in0, in1, in2, unit_attrs=unit_attrs)


ternary_ops = [
    where | Marks(pytest.mark.xfail(reason="Fails Golden")),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", ternary_ops)
def test_ternary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    pipeline_options = []
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape, shape, shape],
        inputs_types=[dtype, dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


# Ternary eltwise ops with implicit broadcasting
@pytest.mark.xfail(reason="Fails Golden")
@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 16, 32), (8, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (1, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (8, 16, 32), (1, 16, 32)],
        [(8, 16, 32), (1, 1, 32), (1, 1, 32)],
        [(1, 1, 32), (8, 16, 32), (1, 1, 32)],
        [(1, 1, 32), (1, 1, 32), (8, 16, 32)],
        [(1, 16, 32), (8, 1, 32), (8, 16, 1)],
        [(1, 4, 1), (1, 4, 768), (1, 1, 1)],
        [(1, 1, 1, 4), (1, 1, 1, 1), (1, 1, 1, 1)],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        pytest.param((torch.float32, torch.float32, torch.float32), id="f32-f32-f32"),
        pytest.param((torch.float32, torch.int32, torch.int32), id="f32-i32-i32"),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", [where])
def test_ternary_eltwise_ops_implicit_broadcast(
    test_fn: Callable,
    shapes: List[Shape],
    input_dtypes: Tuple[torch.dtype, torch.dtype, torch.dtype],
    target: str,
    request,
    device,
):
    dtype1, dtype2, dtype3 = input_dtypes

    compile_and_execute_ttnn(
        test_fn,
        shapes,
        [dtype1, dtype2, dtype3],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
