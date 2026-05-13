# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str, SkipIf

pytestmark = pytest.mark.frontend("ttir")


# Example test for usage of ttnn-mode to verify simple builder functionality
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.int32 | SkipIf("sim"),
    ],
    ids=["i32"],
)
@pytest.mark.parametrize("target", ["ttnn-mode"])
def test_subtract(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype], ttnn_inputs=True)
        def binary_op_fn(in0: Operand, in1: Operand, builder: TTIRBuilder) -> Operand:
            sub = builder.subtract(in0, in1)
            return builder.to_layout(sub, in0.type)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
