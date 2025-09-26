# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder import Operand, Shape
from builder.base import builder_golden
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_utils import compile_d2m_to_flatbuffer


pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_tilize(shape: Shape, target: str, request):
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

        # Provide an explicit identity index_map for the view output type.
        id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
        view_as_rm = builder.view_layout(
            to_device,
            output_type=builder.get_metal_tensor_layout(
                shape, tiled=False, index_map=id_map
            ),
            reinterpret_layout=True,
            unit_attrs=unit_attrs,
        )

        from_device = builder.to_layout(
            view_as_rm,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return from_device

    compile_d2m_to_flatbuffer(
        tilize,
        [shape],
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip(
    reason="Issue #3486: Unit testing untilize hits some unexpected lowering behaviour."
)
@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_untilize(shape: Shape, target: str, request):
    def untilize(
        in0: Operand,
        builder: D2MBuilder,
        unit_attrs: List[str] = None,
    ):

        input = torch.randn(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        golden_output = builder_golden.get_golden_function(
            ttir.ToLayoutOp, tilize=False
        )(input)
        builder.set_graph_input_output([input], [golden_output])

        to_device = builder.to_layout(
            in0,
            output_type=builder.get_metal_tensor_layout(
                shape, tiled=False, grid=(1, 1)
            ),
            unit_attrs=unit_attrs,
        )

        # Provide an explicit identity index_map for the view output type.
        id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
        view_as_tiled = builder.view_layout(
            to_device,
            output_type=builder.get_metal_tensor_layout(
                shape, (1, 1), True, index_map=id_map
            ),
            reinterpret_layout=True,
            unit_attrs=unit_attrs,
        )

        from_device = builder.untilize(
            view_as_tiled,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return from_device

    compile_d2m_to_flatbuffer(
        untilize,
        [shape],
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64)])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_tilize_untilize(shape: Shape, target: str, request):
    def tilize_untilize(
        in0: Operand,
        builder: D2MBuilder,
        unit_attrs: List[str] = None,
    ):
        to_device = builder.tilize(
            in0,
            output_type=builder.get_metal_tensor_layout(shape, tiled=True, grid=(1, 1)),
            unit_attrs=unit_attrs,
        )
        from_device = builder.untilize(
            to_device,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )
        return from_device

    compile_d2m_to_flatbuffer(
        tilize_untilize,
        [shape],
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
