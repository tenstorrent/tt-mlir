# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional

from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


@pytest.mark.parametrize("shape", [(1, 128)])
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("stable", [False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_sort(
    shape: Shape,
    dim: int,
    descending: bool,
    stable: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def sort_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            sort_values, _ = builder.sort(
                in0,
                dim=dim,
                descending=descending,
                stable=stable,
                unit_attrs=unit_attrs,
            )
            return sort_values

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
    )
