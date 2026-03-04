# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_sparse_matmul_b_sparse(target: str, request, device):
    """
    sparse_matmul with is_input_b_sparse=True (column-parallel MoE gate/up).
    a: [A, B, M, K] = [2, 4, 32, 64]
    b: [1, E, K, N] = [1, 4, 64, 128]
    sparsity: [A, B, 1, E] = [2, 4, 1, 4]
    output: [A, B, 1, E, M, N] = [2, 4, 1, 4, 32, 128]
    """

    def module(builder: TTIRBuilder):
        @builder.func(
            [(2, 4, 32, 64), (1, 4, 64, 128), (2, 4, 1, 4)],
            [torch.bfloat16, torch.bfloat16, torch.bfloat16],
        )
        def sparse_matmul_b_sparse(
            a: Operand,
            b: Operand,
            sparsity: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.sparse_matmul(
                a,
                b,
                sparsity,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
                nnz=0,
                output_shape=(2, 4, 1, 4, 32, 128),
                output_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_sparse_matmul_a_sparse(target: str, request, device):
    """
    sparse_matmul with is_input_a_sparse=True, is_input_b_sparse=False (row-parallel).
    a: [A, E, M, K] = [2, 4, 32, 64]
    b: [1, E, K, N] = [1, 4, 64, 128]
    sparsity: [1, 1, A, E] = [1, 1, 2, 4]
    output: [A, E, M, N] = [2, 4, 32, 128]
    """

    def module(builder: TTIRBuilder):
        @builder.func(
            [(2, 4, 32, 64), (1, 4, 64, 128), (1, 1, 2, 4)],
            [torch.bfloat16, torch.bfloat16, torch.bfloat16],
        )
        def sparse_matmul_a_sparse(
            a: Operand,
            b: Operand,
            sparsity: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.sparse_matmul(
                a,
                b,
                sparsity,
                is_input_a_sparse=True,
                is_input_b_sparse=False,
                nnz=0,
                output_shape=(2, 4, 32, 128),
                output_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        **get_request_kwargs(request),
    )
