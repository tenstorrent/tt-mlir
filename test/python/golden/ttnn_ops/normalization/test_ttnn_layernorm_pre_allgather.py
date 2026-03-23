# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize(
    "input_shape,recip_shape",
    [
        ((1, 1, 32, 1024), (1, 1, 32, 32)),
        ((1, 1, 32, 2048), (1, 1, 32, 32)),
    ],
    ids=["1x1x32x1024", "1x1x32x2048"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_layernorm_pre_allgather(
    input_shape: Shape, recip_shape: Shape, target: str, request, device
):
    def module(builder: TTNNBuilder):
        @builder.func(
            [input_shape, recip_shape], [torch.bfloat16, torch.bfloat16]
        )
        def layernorm_pre_allgather(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.layernorm_pre_allgather(
                in0, in1, unit_attrs=unit_attrs
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
