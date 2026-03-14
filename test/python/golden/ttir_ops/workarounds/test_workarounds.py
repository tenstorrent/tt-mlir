# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_and_execute_ttir,
)
from test_utils import (
    shape_str,
    shapes_list_str,
)

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str) -> bool:
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False


@pytest.mark.parametrize(
    "shapes", [((3, 128, 128), (3, 128, 128), (128,))], ids=shapes_list_str
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.skip(
    reason="Batched input not supported when bias exists (linear operation). "
    "The TT_FATAL on device corrupts device state, causing subsequent SDPA "
    "decode tests to produce incorrect results. "
    "https://github.com/tenstorrent/tt-metal/issues/31634"
)
def test_linear_without_workaround(
    shapes: List[Shape],
    dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype, dtype])
        def linear_wrapper(
            in0: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.linear(
                in0, weight, bias, transpose_a, transpose_b, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # Multi-row bias: bias shape [17, 32] has 17 rows.
        # The fused bias kernel (add_tiles_bcast_rows) only broadcasts row 0,
        # silently ignoring rows 1-16 and producing incorrect results.
        # Small K=32 so bias error is not masked by matmul output variance.
        pytest.param(
            ((17, 32), (32, 32), (17, 32)),
            marks=pytest.mark.xfail(
                reason="Fused bias kernel broadcasts only row 0 of bias tile, "
                "silently producing incorrect results for multi-row bias. "
                "https://github.com/tenstorrent/tt-metal/issues/39390"
            ),
        ),
        # Batched bias [2, 1, 1] with non-batched weight.
        # Bias batch dim > 1 triggers TT_FATAL(bias_batch_size == 1).
        # Skipped instead of xfail because TT_FATAL corrupts device state,
        # causing subsequent tests (e.g. SDPA decode) to produce wrong results.
        pytest.param(
            ((2, 33, 1024), (1024, 1024), (2, 1, 1)),
            marks=pytest.mark.skip(
                reason="Bias with non-unit batch dimensions triggers TT_FATAL "
                "which corrupts device state for subsequent tests. "
                "https://github.com/tenstorrent/tt-metal/issues/31634"
            ),
        ),
        # Output shape mismatch on fused kernel path.
        # When bias is effectively 1D (padded height == TILE_HEIGHT) and the
        # broadcasted output shape differs from the matmul shape, the fused
        # kernel produces output with matmul shape [256, 512] instead of the
        # expected broadcasted shape [1, 256, 512].
        # Skipped instead of xfail because TT_FATAL corrupts device state,
        # causing subsequent tests (e.g. SDPA decode) to produce wrong results.
        pytest.param(
            ((256, 1024), (1024, 512), (1, 1, 512)),
            marks=pytest.mark.skip(
                reason="Fused linear kernel triggers TT_FATAL which corrupts "
                "device state for subsequent tests. "
                "https://github.com/tenstorrent/tt-metal/issues/39392"
            ),
        ),
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_linear_bias_decomposition_without_workaround(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test linear op bias decomposition scenarios with workarounds disabled.
    Each param tests a different reason the bias cannot stay fused:
    - Multi-row bias (silent correctness bug in fused kernel)
    - Batched bias with non-batched weight
    """

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype, dtype])
        def linear_bias_decomp_wrapper(
            in0: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.linear(
                in0, weight, bias, False, False, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # Head dim not divisible by 32
        pytest.param(
            [
                (1, 8, 64, 50),  # query
                (1, 8, 64, 50),  # key
                (1, 8, 64, 50),  # value
                (1, 1, 64, 64),  # attention mask
            ],
            marks=pytest.mark.xfail(
                reason="SDPA with non-32-divisible head_dim fails without ttnn-workaround pass. Metal issue: https://github.com/tenstorrent/tt-metal/issues/33434"
            ),
        ),
        # Both seq_len and head_dim not divisible by 32
        pytest.param(
            [
                (1, 8, 63, 50),  # query
                (1, 8, 64, 50),  # key
                (1, 8, 64, 50),  # value
                (1, 1, 63, 64),  # attention mask
            ],
            marks=pytest.mark.xfail(
                reason="SDPA with non-32-divisible head_dim fails without ttnn-workaround pass. Metal issue: https://github.com/tenstorrent/tt-metal/issues/33434"
            ),
        ),
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 4])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_with_mask_no_workaround(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test Scaled Dot Product Attention with non-32-divisible head_dim,
    with ttnn-workaround pass disabled. Expected to fail without the
    head_dim padding workaround.
    """

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def sdpa_with_mask_no_workaround(
            query: Operand,
            key: Operand,
            value: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            head_dim = shapes[0][-1]
            scale = 1.0 / math.sqrt(head_dim)
            return builder.scaled_dot_product_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # Decode with mask num_heads=1 (broadcast needed)
        # Q: [1, batch, num_heads, head_dim], K/V: [batch, kv_heads, kv_seq, head_dim]
        # Mask: [batch, 1, 1, kv_seq] - heads=1 needs broadcast to num_heads
        [
            (1, 32, 32, 64),  # query (decode shape)
            (32, 32, 128, 64),  # key
            (32, 32, 128, 64),  # value
            (32,),  # cur_pos_tensor
            (32, 1, 1, 128),  # attention mask with heads=1
        ],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "dtypes",
    [[torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.int32, torch.bfloat16]],
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="SDPA decode with mask num_heads=1 fails without workaround. "
    "tt-metal requires mask[2] == num_heads and does not support implicit broadcast."
)
def test_sdpa_decode_mask_broadcast_no_workaround(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test that SDPA decode with mask num_heads=1 fails without the workaround.
    tt-metal requires mask[2] == num_heads for decode.
    """
    batch = shapes[0][1]
    kv_seq = shapes[1][2]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def sdpa_decode_mask_broadcast(
            query: Operand,
            key: Operand,
            value: Operand,
            cur_pos_tensor: Operand,
            attention_mask: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            head_dim = shapes[0][-1]
            scale = 1.0 / math.sqrt(head_dim)
            result = builder.scaled_dot_product_attention_decode(
                query,
                key,
                value,
                cur_pos_tensor=cur_pos_tensor,
                attention_mask=attention_mask,
                is_causal=False,
                scale=scale,
                unit_attrs=unit_attrs,
            )

            cur_pos_data = torch.full((batch,), kv_seq - 1, dtype=torch.int32)
            builder.set_goldens({cur_pos_tensor: cur_pos_data})
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize("shape", [(1, 64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="Sort with float32 input requires workaround to convert to bfloat16. Metal issue: https://github.com/tenstorrent/tt-metal/issues/37322"
)
def test_sort_without_workaround(
    shape: Shape,
    dtype: torch.dtype,
    dim: int,
    target: str,
    request,
    device,
):
    """
    Test sort operation with float32 input and workarounds disabled.
    Should fail because metal sort expects bfloat16 input.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def sort_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            sort_0_values, sort_0_indices = builder.sort(
                in0,
                dim=dim,
                descending=False,
                stable=False,
                unit_attrs=unit_attrs,
            )
            return sort_0_values

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=["disable-workarounds=true"],
    )
