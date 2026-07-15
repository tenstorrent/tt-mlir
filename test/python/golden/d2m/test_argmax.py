# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def create_argmax_inputs(input_shape, dim_arg, keep_dim, dtype):
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def argmax_inputs(
            in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
        ):
            in_tensor = torch.randn(input_shape, dtype=dtype)
            builder.set_goldens(inputs={in0: in_tensor})
            return builder.argmax(in0, dim_arg=dim_arg, keep_dim=keep_dim)

    return module


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dim_arg", [[0], [1], None])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (32, 64), (64, 64), (128, 128), (256, 256), (512, 512)],
)
def test_argmax_base(
    shape: tuple[int, int],
    target: str,
    dim_arg: list[int] | None,
    keep_dim: bool,
    request,
    device,
):

    compile_and_execute_ttir(
        create_argmax_inputs(
            shape, dim_arg=dim_arg, keep_dim=keep_dim, dtype=torch.bfloat16
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "shape,target,dim_arg,keep_dim",
    [
        pytest.param((32, 32768), "ttmetal", [1], False, id="phi_1"),
        pytest.param((32, 51200), "ttmetal", [1], False, id="mistral_7b"),
        pytest.param((32, 128256), "ttmetal", [1], False, id="llama_3_2_3b"),
        pytest.param((32, 131072), "ttmetal", [1], False, id="ministral_8b"),
        pytest.param(
            (32, 151936),
            "ttmetal",
            [1],
            False,
            id="qwen_2_5_0_5b",
            marks=pytest.mark.xfail(reason="(32, 151936) has rounding issues"),
        ),
        pytest.param(
            (32, 256000),
            "ttmetal",
            [1],
            False,
            id="gemma_1_1_2b",
            marks=pytest.mark.xfail(
                reason="(32, 256000) tensor exhausts L1 at the moment"
            ),
        ),
    ],
)
def test_argmax_models(
    shape: tuple[int, int],
    target: str,
    dim_arg: list[int] | None,
    keep_dim: bool,
    request,
    device,
):

    compile_and_execute_ttir(
        create_argmax_inputs(
            shape, dim_arg=dim_arg, keep_dim=keep_dim, dtype=torch.bfloat16
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=0.0,
    )
