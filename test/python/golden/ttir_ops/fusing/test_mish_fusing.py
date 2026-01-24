# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import torch.nn.functional as F
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str, dialect: str = "ttnn") -> bool:
    """Check if an op exists in the MLIR file."""
    op_pattern = f"{dialect}.{op_name}"
    with open(mlir_file, "r") as f:
        for line in f:
            if op_pattern in line:
                return True
    return False


def build_ttir(
    input: Operand,
    scalar_constant_value_for_threshold: float,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    # constant for numerically stable softplus
    shape = builder.get_shape(input)
    c = builder.constant(
        torch.full(shape, scalar_constant_value_for_threshold, dtype=torch.float32)
    )

    # Check if input is greater than constant
    gt = builder.gt(input, c, unit_attrs=unit_attrs)

    # Exponentiate the input
    exp = builder.exp(input, unit_attrs=unit_attrs)

    log1p = builder.log1p(exp, unit_attrs=unit_attrs)

    where = builder.where(gt, input, log1p, unit_attrs=unit_attrs)

    tanh = builder.tanh(where, unit_attrs=unit_attrs)

    return builder.multiply(input, tanh, unit_attrs=unit_attrs)


def build_torch_golden(input, scalar_constant_value_for_threshold) -> torch.Tensor:
    c = torch.full_like(input, scalar_constant_value_for_threshold)
    stable_softplus = torch.where(input > c, input, F.softplus(input))
    golden_output = input * torch.tanh(stable_softplus)
    return golden_output


@pytest.mark.parametrize("shape", [(1, 64, 64)])  # input
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_mish_fusion(shape: Shape, dtype: torch.dtype, target: str, request, device):
    """
    This test verifies the Mish fusion pattern.

    The Mish operation is currently represented as the following sequence of ops:
        mish(x) = x * tanh(numerically_stable_softplus(x)) where,
    numerically_stable_softplus(x) is implemented as:
        where(x > C, x, log1p(exp(x)))

    Specifically, the unfused form performs:
    - A comparison of the input tensor `x` against a constant `C`
    - If `x > C`, the value `x` is selected directly
        - Otherwise: - `exp(x)` is computed
        - Followed by `log1p(exp(x))`
    - The selected value is passed through `tanh`
    - The result is multiplied elementwise with the original input `x`

    This test checks that the above sequence is correctly recognized and fused into a single `mish` operation.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def mish_fusing(
            input: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            # Build input tensor
            constant_scalar = 20.0
            input_data = torch.randn(shape, dtype=dtype)
            golden_output = build_torch_golden(input_data, constant_scalar)

            result = build_ttir(input, constant_scalar, builder, unit_attrs=unit_attrs)

            builder.set_goldens(
                {input: input_data},
                {result: golden_output},
            )

            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )

    assert not check_op(output_path, "multiply")
    assert not check_op(output_path, "tanh")
    assert not check_op(output_path, "where")
    assert check_op(
        output_path, "mish"
    ), "Sequence of exp, log1p, where, tanh and multiply should be fused to mish op"
