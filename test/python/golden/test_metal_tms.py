# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TM (Tensor Manipulation) operations on TTMetal backend.

This test suite validates TM operations like permute, transpose, etc.
"""

import pytest
import torch
from typing import List

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "shape, permutation",
    [
        # 3d inner permutes
        [(3, 32, 32), [0, 2, 1]],
        [(3, 32, 64), [0, 2, 1]],
        [(1, 32, 64), [0, 2, 1]],
        # 4d inner permutes
        [(5, 7, 2, 32), [0, 1, 3, 2]],
        [(5, 7, 2, 64), [0, 1, 3, 2]],
        [(5, 7, 2, 128), [0, 1, 3, 2]],
        # 3d inner permutes (llama-like)
        [(1, 50, 12), [0, 2, 1]],
        [(32, 12, 100), [0, 2, 1]],
        [(32, 11, 64), [0, 2, 1]],
        # 4d inner permutes (llama-like)
        [(1, 32, 12, 100), [0, 1, 3, 2]],
        [(1, 32, 11, 64), [0, 1, 3, 2]],
        [(1, 8, 11, 64), [0, 1, 3, 2]],
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_permute_abs(
    shape: Shape, permutation: List[int], target: str, request, device
):
    """Test permute operations with abs on TTMetal backend."""

    def permute_with_abs_module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def permute_with_abs(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            res = builder.permute(in0, permutation=permutation)
            res = builder.abs(res)
            return res

    options = ["collapse-tensors-2d=false"]

    compile_and_execute_ttir(
        permute_with_abs_module,
        target=target,
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
    )
