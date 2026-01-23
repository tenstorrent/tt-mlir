# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


def create_reductions_constrained_inputs(input_shape, reduce_type, dim_arg, keep_dim):
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def reductions_constrained_inputs(
            in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
        ):
            in_tensor = torch.randn(input_shape, dtype=torch.float32)
            # Simulate TF32 truncation in the golden computation
            # TF32 has 10 bits mantissa vs FP32's 23 bits = ~3 decimal digits precision
            scale = 2**13  # Roughly equivalent to TF32 precision
            in_tensor = (in_tensor * scale).round() / scale
            builder.set_goldens(inputs={in0: in_tensor})
            if reduce_type == "sum":
                return builder.sum(in0, dim_arg=dim_arg, keep_dim=keep_dim)
            elif reduce_type == "max":
                return builder.max(
                    in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=unit_attrs
                )

    return module


@pytest.mark.skip_config(["p150"], ["p300"])
@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sum(
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "sum", dim_arg, keep_dim),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=shape[0] * shape[1] * 0.0005,  # 5e-4
    )


@pytest.mark.skip_config(["p150"], ["p300"])
@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_max(
    m: int, n: int, dim_arg: int, keep_dim: bool, target: str, request, device
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "max", dim_arg, keep_dim),
        target=target,
        **get_request_kwargs(request),
        device=device,
    )


def create_reshape_then_reduction_keep_dim_false(input_shape, reduce_dim_after_reshape):
    """Reshape (add size-1 dim) then reduction with keep_dim=False.
    Equivalent to reduction on original with keep_dim=True (ttir-reduction-keep-dim fuses this).
    """
    # Insert size-1 at reduce_dim_after_reshape: e.g. [M,N] -> [M,1,N], reduce dim 2
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def reshape_then_reduce(in0: Operand, builder: TTIRBuilder):
            in_tensor = torch.randn(input_shape, dtype=torch.float32)
            scale = 2**13
            in_tensor = (in_tensor * scale).round() / scale
            builder.set_goldens(inputs={in0: in_tensor})
            shape_list = list(input_shape)
            shape_list.insert(reduce_dim_after_reshape, 1)
            reshaped = builder.reshape(in0, shape_list)
            return builder.sum(
                reshaped, dim_arg=[reduce_dim_after_reshape + 1], keep_dim=False
            )

    return module


def create_reduction_then_reshape_keep_dim_false(input_shape, reduce_dim):
    """Reduction with keep_dim=False then reshape that adds back the reduced dim.
    Equivalent to reduction with keep_dim=True (ttir-reduction-keep-dim fuses this).
    """

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def reduce_then_reshape(in0: Operand, builder: TTIRBuilder):
            in_tensor = torch.randn(input_shape, dtype=torch.float32)
            scale = 2**13
            in_tensor = (in_tensor * scale).round() / scale
            builder.set_goldens(inputs={in0: in_tensor})
            reduced = builder.sum(in0, dim_arg=[reduce_dim], keep_dim=False)
            out_shape = list(builder.get_shape(reduced))
            out_shape.insert(reduce_dim, 1)
            return builder.reshape(reduced, out_shape)

    return module


@pytest.mark.skip_config(["p150"], ["p300"])
@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape_then_sum_keep_dim_false(m: int, n: int, target: str, request, device):
    tile_size = 32
    shape = (m * tile_size, n * tile_size)
    compile_and_execute_ttir(
        create_reshape_then_reduction_keep_dim_false(shape, reduce_dim_after_reshape=1),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=shape[0] * shape[1] * 0.0005,
    )


@pytest.mark.skip_config(["p150"], ["p300"])
@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sum_then_reshape_keep_dim_false(m: int, n: int, target: str, request, device):
    tile_size = 32
    shape = (m * tile_size, n * tile_size)
    compile_and_execute_ttir(
        create_reduction_then_reshape_keep_dim_false(shape, reduce_dim=1),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=shape[0] * shape[1] * 0.0005,
    )
