# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")

# Dimension shifting and relabeling logic is not trivial, so this test exists to catch any silently incorrect transformations.


@pytest.mark.parametrize(
    "shape,permutation,dim_arg",
    [
        # rank-3, single dim reductions
        ((4, 8, 6), [1, 2, 0], [0]),
        ((4, 8, 6), [2, 0, 1], [1]),
        ((4, 8, 6), [1, 2, 0], [2]),
        # rank-3, two dim reduction
        ((4, 8, 6), [2, 0, 1], [0, 2]),
        # rank-4, single dim reductions
        ((2, 3, 4, 5), [0, 2, 3, 1], [1]),
        ((2, 3, 4, 5), [3, 1, 0, 2], [2]),
        ((2, 3, 4, 5), [0, 3, 1, 2], [3]),
        # rank-4, two dim reductions
        ((2, 3, 4, 5), [3, 1, 0, 2], [0, 2]),
        ((2, 3, 4, 5), [0, 3, 1, 2], [2, 3]),
        # rank-4, three dim reduction
        ((2, 3, 4, 5), [0, 2, 3, 1], [0, 1, 3]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_permute_mean(
    shape: Shape,
    permutation: List[int],
    dim_arg: List[int],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.bfloat16])
        def permute_mean(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            permuted = builder.permute(in0, permutation=permutation)
            return builder.mean(permuted, dim_arg=dim_arg, keep_dim=False)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@pytest.mark.parametrize(
    "shape,dim_arg,permutation",
    [
        # rank-3 -> rank-2 reductions
        ((4, 8, 6), [0], [1, 0]),
        ((4, 8, 6), [1], [1, 0]),
        ((4, 8, 6), [2], [1, 0]),
        # rank-3 -> rank-1 reduction
        ((4, 8, 6), [0, 2], [0]),
        # rank-4 -> rank-3 reductions
        ((2, 3, 4, 5), [1], [0, 2, 1]),
        ((2, 3, 4, 5), [2], [0, 2, 1]),
        ((2, 3, 4, 5), [3], [2, 0, 1]),
        # rank-4 -> rank-2 reductions
        ((2, 3, 4, 5), [0, 2], [1, 0]),
        ((2, 3, 4, 5), [2, 3], [1, 0]),
        # rank-4 -> rank-1 reduction
        ((2, 3, 4, 5), [0, 1, 3], [0]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_mean_permute(
    shape: Shape,
    dim_arg: List[int],
    permutation: List[int],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.bfloat16])
        def mean_permute(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            reduced = builder.mean(in0, dim_arg=dim_arg, keep_dim=False)
            return builder.permute(reduced, permutation=permutation)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
