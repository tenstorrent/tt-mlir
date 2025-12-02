# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TM (Tensor Manipulation) operations on TTMetal backend.

This test suite validates TM operations like permute, transpose, etc.
"""

import pytest
from typing import List

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "shapes, permutation",
    [
        # 4d outer permutes
        [(1, 32, 31, 32), [0, 2, 1, 3]],
        [(1, 32, 1, 32), [0, 2, 1, 3]],
        [(5, 7, 2, 32), [0, 2, 1, 3]],
        # 5d outer permutes
        [(1, 3, 3, 3, 3), [0, 2, 1, 3, 4]],
        [(1, 3, 3, 3, 3), [0, 2, 1, 3, 4]],
        [(5, 7, 2, 3, 3), [0, 2, 1, 3, 4]],
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_permute_abs(
    shapes: List[Shape], permutation: List[int], target: str, request, device
):
    """Test 4D permute operation with abs on TTMetal backend."""

    def permute_with_abs(
        in0: Operand,
        builder: TTIRBuilder,
    ):
        res = builder.permute(in0, permutation=permutation)
        res = builder.abs(res)
        return res

    options = ["collapse-tensors-2d=false"]

    compile_and_execute_ttir(
        permute_with_abs,
        [shapes],
        system_desc_path=request.config.getoption("--sys-desc"),
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
    )
