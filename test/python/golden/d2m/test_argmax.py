# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
torch.manual_seed(0)


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


# Single case: 32x32 tensor, row-wise reduction (argmax over dim 1), bf16 input.
# argmax returns int32 indices, so the golden comparison is exact (atol=0).
@pytest.mark.parametrize("target", ["ttmetal"])
def test_argmax_2d_rowwise_bf16(target: str, request, device):
    shape = (32, 32)

    compile_and_execute_ttir(
        create_argmax_inputs(shape, dim_arg=[0], keep_dim=False, dtype=torch.bfloat16),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=0.0,
    )
