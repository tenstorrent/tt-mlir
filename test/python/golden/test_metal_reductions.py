# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


def create_reductions_constrained_inputs(input_shape, reduce_type, dim_arg, keep_dim):
    def reductions_constrained_inputs(
        in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
    ):
        in_tensor = torch.randn(input_shape, dtype=torch.float32)
        # Simulate TF32 truncation in the golden computation
        # TF32 has 10 bits mantissa vs FP32's 23 bits = ~3 decimal digits precision
        scale = 2**13  # Roughly equivalent to TF32 precision
        in_tensor = (in_tensor * scale).round() / scale
        builder.set_goldens(inputs={in0: in_tensor})
        if reduce_type == "sum":
            return builder.sum(
                in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=unit_attrs
            )
        elif reduce_type == "max":
            return builder.max(
                in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=unit_attrs
            )

    return reductions_constrained_inputs


@pytest.mark.skip_config(["p150"], ["p300"])
@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sum(
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "sum", dim_arg, keep_dim),
        [shape],
        target=target,
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        atol=shape[0] * shape[1] * 0.0005,  # 5e-4
    )


@pytest.mark.skip_config(["p150"], ["p300"])
@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [0, 1])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_max(
    m: int, n: int, dim_arg: int, keep_dim: bool, target: str, request, device
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "max", dim_arg, keep_dim),
        [shape],
        target=target,
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )
