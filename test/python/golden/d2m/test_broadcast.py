# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional

from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


EXPLICIT_BROADCAST_CASES = [
    pytest.param(
        (1,),
        [128],
        torch.float32,
        id="f32-1d-1-to-128",
    ),
    pytest.param(
        (1, 64),
        [128, 1],
        torch.float32,
        id="f32-row-1x64-to-128x64",
    ),
    pytest.param(
        (64, 1),
        [1, 128],
        torch.float32,
        id="f32-col-64x1-to-64x128",
    ),
    pytest.param(
        (1, 1),
        [64, 128],
        torch.bfloat16,
        id="bf16-scalar-1x1-to-64x128",
    ),
    pytest.param(
        (1, 1, 32, 1),
        [1, 4, 1, 128],
        torch.float32,
        id="f32-nd-col-1x1x32x1-to-1x4x32x128",
    ),
    pytest.param(
        (1, 1, 1, 128),
        [1, 4, 32, 1],
        torch.bfloat16,
        id="bf16-nd-row-1x1x1x128-to-1x4x32x128",
    ),
    pytest.param(
        (1, 1, 32, 128),
        [1, 4, 1, 1],
        torch.float32,
        id="f32-outer-1x1x32x128-to-1x4x32x128",
    ),
    pytest.param(
        (1, 32, 32),
        [4, 1, 1],
        torch.float32,
        id="f32-outer-1x32x32-to-4x32x32",
    ),
    pytest.param(
        (1, 1, 32, 128),
        [2, 4, 1, 1],
        torch.bfloat16,
        id="bf16-multi-outer-1x1x32x128-to-2x4x32x128",
    ),
]


def broadcast_output_shape(shape: Shape, broadcast_dimensions: List[int]) -> Shape:
    return tuple(
        broadcast_dim if broadcast_dim != 1 else input_dim
        for input_dim, broadcast_dim in zip(shape, broadcast_dimensions)
    )


@pytest.mark.parametrize("shape,broadcast_dimensions,dtype", EXPLICIT_BROADCAST_CASES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_explicit_broadcast(
    shape: Shape,
    broadcast_dimensions: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def explicit_broadcast(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            if shape == (1,):
                builder.set_goldens(inputs={in0: torch.ones(shape, dtype=dtype)})
            return builder.broadcast(
                in0,
                broadcast_dimensions=broadcast_dimensions,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        print_ir=False,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_explicit_broadcast_consumed_by_binary(request, device, target: str):
    input_shape = (1, 1, 32, 1)
    broadcast_dimensions = [1, 4, 1, 128]
    output_shape = broadcast_output_shape(input_shape, broadcast_dimensions)

    def module(builder: TTIRBuilder):
        @builder.func(
            [input_shape, output_shape],
            [torch.float32, torch.float32],
        )
        def broadcast_add(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            broadcasted = builder.broadcast(
                in0,
                broadcast_dimensions=broadcast_dimensions,
                unit_attrs=unit_attrs,
            )
            return builder.add(broadcasted, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        print_ir=False,
    )
