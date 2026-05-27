# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize(
    "topk_shape,expert_mapping_shape,expert_metadata_shape,reduction_size,"
    "mapping_shape,reduced_shape",
    [
        (
            (1, 1, 32, 8),  # topk_tensor: [1, BD, S, E]
            (1, 1, 8, 1),  # expert_mapping: [1, 1, E, num_devices=1]
            (1, 1, 32, 4),  # expert_metadata: [1, BD, S, K]
            32,  # reduction_size
            (1, 1, 32, 8),  # mapping output: [1, BD, S, E/num_devices]
            (1, 1, 1, 8),  # reduced output: [1, 1, ceil(BD*S/R), E/num_devices]
        ),
    ],
    ids=["1x32x8_r32"],
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_moe_expert_token_remap(
    topk_shape: Shape,
    expert_mapping_shape: Shape,
    expert_metadata_shape: Shape,
    reduction_size: int,
    mapping_shape: Shape,
    reduced_shape: Shape,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func(
            [topk_shape, expert_mapping_shape, expert_metadata_shape],
            [torch.bfloat16, torch.int64, torch.int64],
        )
        def moe_expert_token_remap(
            topk_tensor: Operand,
            expert_mapping: Operand,
            expert_metadata: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.moe_expert_token_remap(
                topk_tensor,
                expert_mapping,
                expert_metadata,
                reduction_size=reduction_size,
                mapping_shape=mapping_shape,
                mapping_type=torch.bfloat16,
                reduced_shape=reduced_shape,
                reduced_type=torch.bfloat16,
                unit_attrs=unit_attrs,
            )

    # Golden for moe_expert_token_remap is a stub (returns zeros) since the op
    # performs device-level expert routing that can't be replicated on host.
    # Skip PCC check; this test validates compilation and execution.
    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        check_pcc=False,
    )
