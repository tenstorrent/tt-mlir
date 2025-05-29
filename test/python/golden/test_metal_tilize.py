# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List

from ttir_builder.utils import compile_to_flatbuffer, Marks, shape_str
from ttir_builder import Operand, TTIRBuilder, UnitAttr, Shape, TypeInfo
from ttmlir.dialects import ttir, tt
from ttmlir.ir import *


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
def test_tilize(shape: Shape, request):
    def tilize(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):

        to_device = builder.tilize(
            in0,
            output_type=builder.metal_tensor_layout(shape, (1, 1), True),
            unit_attrs=unit_attrs,
        )

        view_as_rm = builder.view_layout(
            to_device,
            output_type=builder.metal_tensor_layout(shape, (1, 1), False),
            reinterpret_layout=True,
            unit_attrs=unit_attrs,
        )

        from_device = builder.to_layout(
            view_as_rm,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return from_device

    compile_to_flatbuffer(
        tilize,
        [shape],
        target="ttmetal",
        custom_pipeline="ttir-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip(
    reason="Issue #3486: Unit testing untilize hits some unexpected lowering behaviour."
)
@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
def test_untilize(shape: Shape, request):
    def untilize(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):

        input = torch.randn(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
        golden_output = builder.untilize_golden(input)
        builder.set_graph_input_output([input], [golden_output])

        to_device = builder.to_layout(
            in0,
            output_type=builder.metal_tensor_layout(shape, (1, 1), False),
            unit_attrs=unit_attrs,
        )

        view_as_tiled = builder.view_layout(
            to_device,
            output_type=builder.metal_tensor_layout(shape, (1, 1), True),
            reinterpret_layout=True,
            unit_attrs=unit_attrs,
        )

        from_device = builder.untilize(
            view_as_tiled,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return from_device

    compile_to_flatbuffer(
        untilize,
        [shape],
        target="ttmetal",
        custom_pipeline="ttir-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64)])
def test_tilize_untilize(shape: Shape, request):
    def tilize_untilize(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        to_device = builder.tilize(
            in0,
            output_type=builder.metal_tensor_layout(shape, (1, 1), True),
            unit_attrs=unit_attrs,
        )
        from_device = builder.untilize(
            to_device,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )
        return from_device

    compile_to_flatbuffer(
        tilize_untilize,
        [shape],
        target="ttmetal",
        custom_pipeline="ttir-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
