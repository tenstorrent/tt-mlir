# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Arange shapes the argmax lowering builds as its index operand.

The argmax lowering constructs an arange counting along the reduction axis and
feeds it to `max_reduce_with_indices` as the index tile. On device that tile was
observed arriving in DST as all zeros, so this pins down whether plain arange
still lowers correctly on its own -- independent of the argmax op.

Kept deliberately narrow: `arange_dimension=0` (counting down rows, which is what
a dim=0 reduction needs) at the tile heights the argmax tests exercise, plus the
i32 element type the index path requires.
"""

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def create_arange(shape, dtype, arange_dimension):
    # `end` sizes the range along `arange_dimension` only -- the golden builds a
    # 1-D range of (end - start) / step elements and broadcasts it across the
    # other dimension. (test_tms passes shape[0] * shape[1] because its shapes
    # have shape[0] == 1, where the two happen to coincide.)
    end = shape[arange_dimension]

    def arange_module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def arange_operand(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.arange(
                shape=list(shape),
                dtype=dtype,
                start=0,
                end=end,
                step=1,
                arange_dimension=arange_dimension,
                unit_attrs=unit_attrs,
            )

    return arange_module


# dim 0 counts down the rows -- the layout a dim=0 argmax reduction consumes.
# dim 1 is included as the known-good control, since test_tms covers it today.
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("arange_dimension", [0, 1])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="32x32"),
        pytest.param((64, 32), id="64x32"),
        pytest.param((96, 32), id="96x32"),
    ],
)
def test_arange_f32(
    shape: tuple[int, int],
    arange_dimension: int,
    target: str,
    request,
    device,
):
    compile_and_execute_ttir(
        create_arange(shape, torch.float32, arange_dimension),
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
        atol=1e-6,
        check_atol=True,
    )


# The argmax index operand is i32. test_tms xfails i32 arange for a missing
# tile*scalar llk (issue 7946); the argmax path reaches it via a different
# lowering, so run it here rather than assuming either outcome.
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("arange_dimension", [0, 1])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 32), id="32x32"),
        pytest.param((64, 32), id="64x32"),
    ],
)
def test_arange_i32(
    shape: tuple[int, int],
    arange_dimension: int,
    target: str,
    request,
    device,
):
    compile_and_execute_ttir(
        create_arange(shape, torch.int32, arange_dimension),
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
        atol=0.0,
    )
