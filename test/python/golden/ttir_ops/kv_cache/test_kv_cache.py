# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shapes_list_str

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param([(1, 32, 64, 512), (1, 32, 1, 512), (1,)], id="single_user"),
        # Multi-user decoder pattern: num_input_users == num_users > 1.
        pytest.param([(8, 4, 16, 32), (1, 4, 8, 32), (1,)], id="multi_user"),
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32, torch.float32, torch.int32]])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_update_cache(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def update_cache(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Provide a valid index within the cache sequence dimension
            cache_seq_len = shapes[0][2]
            update_index = torch.randint(0, cache_seq_len, shapes[2], dtype=torch.int32)
            builder.set_goldens(inputs={in2: update_index})
            return builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
