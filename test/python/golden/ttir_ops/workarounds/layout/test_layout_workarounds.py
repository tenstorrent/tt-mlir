# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
from typing import Callable, List, Optional, Tuple
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_and_execute_ttir,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
)
from ttmlir.dialects import ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shape", [(1, 64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="Sort with float32 input requires workaround to convert to bfloat16. Metal issue: https://github.com/tenstorrent/tt-metal/issues/37322"
)
def test_sort_without_workaround(
    shape: Shape,
    dtype: torch.dtype,
    dim: int,
    target: str,
    request,
    device,
):
    """
    Test sort operation with float32 input and workarounds disabled.
    Should fail because metal sort expects bfloat16 input.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def sort_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            sort_0_values, sort_0_indices = builder.sort(
                in0,
                dim=dim,
                descending=False,
                stable=False,
                unit_attrs=unit_attrs,
            )
            return sort_0_values

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=["disable-workarounds=true"],
    )
