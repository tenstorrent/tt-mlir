# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import os
from typing import List, Optional

from builder.base.builder_utils import Operand, get_artifact_dir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str, dialect: str = "ttnn") -> bool:
    """Check if an op exists in the MLIR file."""
    op_pattern = f"{dialect}.{op_name}"
    with open(mlir_file, "r") as f:
        for line in f:
            if op_pattern in line:
                return True
    return False


def create_permute_matmul_lhs(lhs_shape, rhs_shape):
    """Create a permute -> matmul pattern where LHS is permuted."""

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch.float32, torch.float32])
        def permute_matmul(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Use float32 to avoid dtype issues with golden computation
            in0_data = torch.rand(lhs_shape, dtype=torch.float32) * 0.999 + 0.001
            in1_data = torch.rand(rhs_shape, dtype=torch.float32) * 0.999 + 0.001

            builder.set_goldens(inputs={in0: in0_data, in1: in1_data})

            # Permute the first input (transpose)
            permuted = builder.permute(in0, [1, 0], unit_attrs=unit_attrs)

            # Matmul: permuted @ in1
            return builder.matmul(permuted, in1, unit_attrs=unit_attrs)

    return module


def create_permute_matmul_rhs(lhs_shape, rhs_shape):
    """Create a permute -> matmul pattern where RHS is permuted."""

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch.float32, torch.float32])
        def permute_matmul(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Use float32 to avoid dtype issues with golden computation
            in0_data = torch.rand(lhs_shape, dtype=torch.float32) * 0.999 + 0.001
            in1_data = torch.rand(rhs_shape, dtype=torch.float32) * 0.999 + 0.001

            builder.set_goldens(inputs={in0: in0_data, in1: in1_data})

            # Permute the second input (transpose)
            permuted = builder.permute(in1, [1, 0], unit_attrs=unit_attrs)

            # Matmul: in0 @ permuted
            return builder.matmul(in0, permuted, unit_attrs=unit_attrs)

    return module


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        # permute(64,128)->(128,64), then (128,64)@(64,128)->(128,128)
        ((64, 128), (64, 128)),
    ],
)
def test_permute_matmul_lhs_fusion_enabled(
    lhs_shape,
    rhs_shape,
    request,
    device,
):
    """Test that permute on LHS is fused into matmul when flag is enabled (default)."""
    compile_and_execute_ttir(
        create_permute_matmul_lhs(lhs_shape, rhs_shape),
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        pipeline_options=["enable-permute-matmul-fusion=true"],
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )

    # When fusion is enabled, permute should NOT appear (it's fused into matmul)
    assert not check_op(output_path, "permute"), "Permute should be fused into matmul"
    assert check_op(output_path, "matmul"), "Matmul should exist"


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        ((64, 128), (64, 128)),
    ],
)
def test_permute_matmul_lhs_fusion_disabled(
    lhs_shape,
    rhs_shape,
    request,
    device,
):
    """Test that permute on LHS is NOT fused into matmul when flag is disabled."""
    compile_and_execute_ttir(
        create_permute_matmul_lhs(lhs_shape, rhs_shape),
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        pipeline_options=["enable-permute-matmul-fusion=false"],
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )

    # When fusion is disabled, permute should still appear
    assert check_op(
        output_path, "permute"
    ), "Permute should NOT be fused when flag is disabled"
    assert check_op(output_path, "matmul"), "Matmul should exist"


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        # in0=(64,128), in1=(64,128), permute(in1)->(128,64), then (64,128)@(128,64)->(64,64)
        ((64, 128), (64, 128)),
    ],
)
def test_permute_matmul_rhs_fusion_enabled(
    lhs_shape,
    rhs_shape,
    request,
    device,
):
    """Test that permute on RHS is fused into matmul when flag is enabled."""
    compile_and_execute_ttir(
        create_permute_matmul_rhs(lhs_shape, rhs_shape),
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        pipeline_options=["enable-permute-matmul-fusion=true"],
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )

    # When fusion is enabled, permute should NOT appear (it's fused into matmul)
    assert not check_op(output_path, "permute"), "Permute should be fused into matmul"
    assert check_op(output_path, "matmul"), "Matmul should exist"
