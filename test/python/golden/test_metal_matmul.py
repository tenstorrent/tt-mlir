# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttir_builder.utils import compile_to_flatbuffer
from ttir_builder import Operand, TTIRBuilder, Shape


@pytest.mark.parametrize("grid_m", [1])
@pytest.mark.parametrize("grid_k", [1])
@pytest.mark.parametrize("grid_n", [1])
@pytest.mark.parametrize("m", [2])
@pytest.mark.parametrize("k", [4, 6])
@pytest.mark.parametrize("n", [3, 4])
def test_matmul(
    grid_m: int,
    grid_k: int,
    grid_n: int,
    m: int,
    k: int,
    n: int,
    request,
):
    tile_size = 32
    lhs = (
        grid_m * m * tile_size,
        grid_k * k * tile_size,
    )
    rhs = (
        grid_k * k * tile_size,
        grid_n * n * tile_size,
    )

    def matmul(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        a = torch.randint(
            0, 3, builder._get_golden_tensor(in0).shape, dtype=torch.float32
        )
        b = torch.randint(
            0, 3, builder._get_golden_tensor(in1).shape, dtype=torch.float32
        )
        golden_output = torch.matmul(a, b)
        builder.set_graph_input_output([a, b], [golden_output], override=True)

        return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    options = [
        f"override-device-shape={grid_m},{grid_n}",
    ]
    compile_to_flatbuffer(
        matmul,
        [lhs, rhs],
        target="ttmetal",
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
