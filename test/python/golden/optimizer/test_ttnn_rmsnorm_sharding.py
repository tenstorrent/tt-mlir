# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import pytest
import torch

from builder.base.builder_utils import Shape
from builder.base.builder_apis import compile_and_execute_ttnn
from builder.ttnn.ttnn_builder import TTNNBuilder
from conftest import get_request_kwargs

# Temporarily disabled: RMSNorm with sharded input causes crash in metal.
pytestmark = [
    pytest.mark.frontend("ttnn"),
    pytest.mark.skip(
        reason="Temporarily disabled: RMSNorm sharding causes crash in metal"
    ),
]


@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 1, 32, 2048), (2048,), (2048,)],
        [(1, 5, 32, 2048), (2048,), (2048,)],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16]])
@pytest.mark.parametrize("epsilon", [1.0e-5])
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_bias", [False, True])
def test_rmsnorm_sharding(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    epsilon: float,
    has_weight: bool,
    has_bias: bool,
    request,
    device,
):
    used_dtypes = [dtypes[0]]
    used_shapes = [shapes[0]]
    if has_weight:
        used_dtypes += [dtypes[0]]
        used_shapes += [shapes[1]]
    if has_bias:
        used_dtypes += [dtypes[0]]
        used_shapes += [shapes[2]]

    def module(builder: TTNNBuilder):
        @builder.func(used_shapes, used_dtypes)
        def rmsnorm_test(*inputs):
            builder: TTNNBuilder = inputs[-1]
            input_tensor = inputs[0]
            weight = None
            bias = None

            if has_weight:
                weight = inputs[1]
                if has_bias:
                    bias = inputs[2]
            elif has_bias:
                bias = inputs[1]

            result = builder.rms_norm(
                input_tensor,
                weight=weight,
                bias=bias,
                epsilon=epsilon,
                unit_attrs=["l1_width_sharded"],
            )
            return result

    # Execute the test with sharding enabled
    output_file_mlir = compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
        target="ttnn",
    )
    print("Output MLIR file:", output_file_mlir)
