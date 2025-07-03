# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional

from ttir_builder import Operand, TTIRBuilder, Shape
from ttir_builder.utils import compile_to_flatbuffer, shape_str


@pytest.mark.parametrize("shape_lhs", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("shape_rhs", [(32, 1)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_add(
    shape_lhs: Shape, shape_rhs: Shape, dtype: torch.dtype, target: str, request
):
    def add(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        input_A = torch.ones(shape_lhs, dtype=dtype)
        input_B = torch.ones(shape_rhs, dtype=dtype)
        golden_output_tensor = torch.add(input_A, input_B)
        builder.set_graph_input_output(
            [input_A, input_B], [golden_output_tensor], override=True
        )
        return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_to_flatbuffer(
        add,
        [shape_lhs, shape_rhs],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
