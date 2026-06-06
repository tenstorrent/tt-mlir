# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


# Batched prefill: input_batch == batch_idx_tensor.shape[0] > 1.
# Exercises the routing enabled by the verifier relaxation in this PR.
# Requires tt-metal #45117 (batched batch_idx_tensor support in
# ttnn::experimental::paged_fill_cache); test will fail until that uplift.
#
# seq_len kept as a multiple of block_size: the device kernel and golden
# disagree on partial-block-tail semantics (pre-existing, unrelated to this
# PR), which produces a PCC ~0.98 on misaligned shapes.
@pytest.mark.parametrize(
    "shapes",
    [
        pytest.param(
            [(128, 12, 32, 256), (4, 12, 96, 256), (8, 16), (4,)], id="multi_user"
        ),
    ],
)
@pytest.mark.parametrize(
    "dtypes", [[torch.bfloat16, torch.bfloat16, torch.int32, torch.int32]]
)
def test_paged_fill_cache(
    shapes: List[Shape], dtypes: List[torch.dtype], request, device
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def paged_fill_cache(
            cache: Operand,
            input: Operand,
            page_table: Operand,
            batch_idx: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            num_blocks = shapes[0][0]
            page_table_batch, max_blocks_per_seq = shapes[2]

            # Deterministic, in-range physical block indices for the page table
            # so cache routing is well-defined.
            page_table_vals = (
                torch.arange(
                    page_table_batch * max_blocks_per_seq, dtype=torch.int32
                ).reshape(page_table_batch, max_blocks_per_seq)
                % num_blocks
            )

            # Distinct, non-positional batch indices — catches a regression
            # where the op silently uses b_idx instead of batch_idx_tensor[b_idx].
            batch_idx_vals = torch.tensor([3, 0, 7, 2], dtype=torch.int32)

            builder.set_goldens(
                inputs={page_table: page_table_vals, batch_idx: batch_idx_vals}
            )
            return builder.paged_fill_cache(
                cache, input, page_table, batch_idx, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )
