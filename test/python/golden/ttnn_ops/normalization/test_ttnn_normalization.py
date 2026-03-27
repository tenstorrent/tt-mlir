# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttnn")


# LayerNormPreAllGather tests


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),
        (1, 1, 32, 512),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
def test_layer_norm_pre_all_gather(
    shape: Shape,
    has_residual: bool,
    target: str,
    request,
    device,
):
    shapes = [shape]
    if has_residual:
        shapes.append(shape)

    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.bfloat16] * len(shapes))
        def layer_norm_pre_all_gather(*inputs, unit_attrs: Optional[List[str]] = None):
            builder = inputs[-1]
            in0 = inputs[0]
            residual = None
            if has_residual and len(inputs) > 2:
                residual = inputs[1]

            return builder.layer_norm_pre_all_gather(
                in0,
                residual_input=residual,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
