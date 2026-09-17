# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("loop_count", [1, 4], ids=["n1", "n4"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_while_bounded_add(
    shape: Shape,
    dtype: torch.dtype,
    loop_count: int,
    target: str,
    request,
    device,
):
    """Bounded `ttir.while` that adds 1 to the input `loop_count` times.

    The result must match `original + loop_count`.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def while_add_one(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            i0 = builder.constant(torch.tensor(0, dtype=torch.int32))
            limit = builder.constant(torch.tensor(loop_count, dtype=torch.int32))
            step = builder.constant(torch.tensor(1, dtype=torch.int32))
            one = builder.full(list(shape), dtype, 1.0)

            def cond(i, acc, l, s, ones):
                return builder.lt(i, l, output_type=torch.bool)

            def body(i, acc, l, s, ones):
                return builder.add(i, s), builder.add(acc, ones)

            results = builder.while_(
                [i0, in0],
                cond,
                body,
                captures=[limit, step, one],
            )
            acc_out = results[1]

            input_data = torch.randn(shape, dtype=dtype)
            builder.set_goldens(
                {in0: input_data},
                {acc_out: input_data + loop_count},
            )
            return acc_out

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
