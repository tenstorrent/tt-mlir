# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List
from conftest import get_request_kwargs

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m


pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int32],
    ids=["f32", "i32"],
)
def test_tilize(shape: Shape, target: str, dtype: torch.dtype, request, device):
    def module(builder: D2MBuilder):
        @builder.func([shape], [dtype])
        def tilize(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):

            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(shape, tiled=True),
                unit_attrs=unit_attrs,
            )

            # Provide an explicit identity remapping for the view op.
            id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
            view_as_rm = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(shape, tiled=False),
                remapping=id_map,
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            from_device = builder.to_layout(
                view_as_rm,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int32],
    ids=["f32", "i32"],
)
def test_untilize(shape: Shape, target: str, dtype: torch.dtype, request, device):
    def module(builder: D2MBuilder):
        @builder.func([shape], [dtype])
        def untilize(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=False, grid=(1, 1)
                ),
                unit_attrs=unit_attrs,
            )

            # Provide an explicit identity remapping for the view op.
            id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
            view_as_tiled = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=(1, 1), tiled=True
                ),
                remapping=id_map,
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            from_device = builder.untilize(
                view_as_tiled,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64)])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int32],
    ids=["f32", "i32"],
)
def test_tilize_untilize(
    shape: Shape, target: str, dtype: torch.dtype, request, device
):
    def module(builder: D2MBuilder):
        @builder.func([shape], [dtype])
        def tilize_untilize(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=True, grid=(1, 1)
                ),
                unit_attrs=unit_attrs,
            )
            from_device = builder.untilize(
                to_device,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )
            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )
