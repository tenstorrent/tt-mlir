# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_and_execute_ttir,
)
from test_utils import shape_str


pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "shape",
    [
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (2048, 2048),
        (256, 256),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_binary_tree(shape: Shape, dtype: torch.dtype, target: str, request, device):
    """Test a binary tree of adds: add(add(arg0, arg1), add(arg2, arg3))"""

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape, shape, shape], [dtype, dtype, dtype, dtype])
        def binary_tree(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            in3: Operand,
            builder: TTIRBuilder,
        ) -> Operand:
            left = builder.add(in0, in1)
            right = builder.add(in2, in3)
            return builder.add(left, right)

    if shape == (2048, 2048) and dtype == torch.float32:
        pytest.xfail("Too big to fit in L1, see issue #7216")

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["enable-elementwise-fusion=true"],
        save_artifacts=True,
    )
