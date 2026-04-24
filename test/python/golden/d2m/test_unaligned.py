# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Tests exercising eltwise ops on shapes that don't align to tile
# boundaries on the ttmetal backend.

import pytest
import torch
from typing import List, Optional

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")


# Shapes that are not multiples of the 32x32 tile. Exercised across a range
# of ranks (2D through 5D) and sizes.
UNALIGNED_SHAPES = [
    (5, 3),
    (32, 1),
    (31, 7),
    (1, 32),
    (13, 29),
    (64, 1),
    (61, 3),
    (61, 37),
    (1, 64),
    (5, 67),
    (43, 67),
    (2, 3, 5),
    (3, 17, 37),
    (9, 43, 7),
    (5, 61, 49),
    (51, 19, 23),
    (2, 3, 5, 7),
    (3, 37, 5, 53),
    (37, 3, 5, 53),
    (41, 7, 43, 11),
    (7, 41, 43, 11),
    (1, 23, 1, 1),
    (23, 1, 1, 1),
    (3, 5, 7, 11, 13),
]


@pytest.mark.parametrize("shape", UNALIGNED_SHAPES + [(677, 1, 1)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unaligned_shapes_neg(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.neg(in0, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        print_ir=False,
    )


@pytest.mark.parametrize(
    "shape",
    UNALIGNED_SHAPES
    + [
        pytest.param(
            (677, 1, 1), marks=pytest.mark.skip_config(["n150"])
        ),  # TODO (anuragsingh): Fix nondeterministic issue with Allocator for this test.
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unaligned_shapes_add(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def add(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Magnitudes of the elements should be in [0.01, 1) to avoid FP accuracy issue.
            tensor_lhs = torch.rand(shape, dtype=dtype) * 0.99 + 0.01
            tensor_rhs = torch.rand(shape, dtype=dtype) * 0.99 + 0.01
            signs_lhs = torch.randint(0, 2, shape) * 2 - 1
            signs_rhs = torch.randint(0, 2, shape) * 2 - 1
            tensor_lhs *= signs_lhs
            tensor_rhs *= signs_rhs
            builder.set_goldens(inputs={in0: tensor_lhs, in1: tensor_rhs})
            return builder.add(in0, in1)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
