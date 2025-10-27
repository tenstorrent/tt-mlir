# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional

from builder.base.builder import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_utils import compile_and_execute_ttnn
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttnn")


def multiply(
    in0: Operand,
    in1: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "test_fn",
    [
        multiply,
    ],
)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, request, device
):
    compile_and_execute_ttnn(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


def mish(
    in0: Operand,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.mish(in0, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        mish,
    ],
)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if test_fn == mish and dtype == torch.float32:
        pytest.skip(
            "Mish with float 32 causes PCC: https://github.com/tenstorrent/tt-metal/issues/31112"
        )

    compile_and_execute_ttnn(
        test_fn,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
