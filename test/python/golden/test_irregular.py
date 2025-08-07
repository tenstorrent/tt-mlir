# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.ttir.ttir_utils import compile_ttir_to_flatbuffer
from test_utils import shape_str


shapes = [
    # (32, 32),
    # (32, 1),
    # (32, 2),
    # (64, 32),
    (64, 1),
    # (64, 2),
]


@pytest.mark.parametrize("shape", shapes, ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_add(shape: Shape, dtype: torch.dtype, target: str, request):
    def add(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        input_A = torch.ones(shape, dtype=dtype)
        input_B = torch.ones(shape, dtype=dtype)
        golden_output_tensor = torch.add(input_A, input_B)
        builder.set_graph_input_output(
            [input_A, input_B], [golden_output_tensor], override=True
        )
        return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_ttir_to_flatbuffer(
        add,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
