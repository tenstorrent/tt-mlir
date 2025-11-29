# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
from typing import List, Optional
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str) -> bool:
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False


def build_torch_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build golden output using PyTorch's scaled_dot_product_attention.
    Supports standard attention and Grouped-Query Attention (GQA).
    """
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    enable_gqa = q_heads != kv_heads

    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scale, enable_gqa=enable_gqa
    )


def build_ttir(
    query: Operand,
    key: Operand,
    value: Operand,
    builder: TTIRBuilder,
    scale: Optional[float] = None,
    attention_mask: Optional[Operand] = None,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of SDPA pattern:
    - Broadcast K/V heads if GQA (query has more heads than K/V)
    - Q @ K^T matmul
    - Scale (multiply)
    - Add mask (optional)
    - Softmax
    - @ V matmul
    """
    # Handle Grouped-Query Attention: broadcast K/V heads to match Q heads if needed
    q_shape = builder.get_shape(query)
    k_shape = builder.get_shape(key)
    q_heads = q_shape[1]
    kv_heads = k_shape[1]

    if q_heads != kv_heads:
        # GQA: repeat each K/V head to match Q heads
        assert (
            q_heads % kv_heads == 0
        ), f"Q heads ({q_heads}) must be divisible by KV heads ({kv_heads})"
        num_repeats = q_heads // kv_heads
        # Repeat K and V along the head dimension (dim=1)
        key = builder.repeat_interleave(
            key, repeats=num_repeats, dim=1, unit_attrs=unit_attrs
        )
        value = builder.repeat_interleave(
            value, repeats=num_repeats, dim=1, unit_attrs=unit_attrs
        )

    # Transpose key: [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
    key_transposed = builder.transpose(key, dim0=-2, dim1=-1)

    # Q @ K^T
    qk = builder.matmul(query, key_transposed, unit_attrs=unit_attrs)

    # Scale if provided
    if scale is not None:
        qk_shape = builder.get_shape(qk)
        scale_shape = [1] * len(qk_shape)
        scale_tensor = builder.full(
            scale_shape, torch.bfloat16, scale, unit_attrs=unit_attrs
        )
        qk = builder.multiply(qk, scale_tensor, unit_attrs=unit_attrs)

    # Add attention mask if provided
    if attention_mask is not None:
        qk = builder.add(qk, attention_mask, unit_attrs=unit_attrs)

    # Softmax on last dimension
    softmax_out = builder.softmax(qk, dimension=-1, unit_attrs=unit_attrs)

    # @ V
    output = builder.matmul(softmax_out, value, unit_attrs=unit_attrs)

    return output


@pytest.mark.parametrize(
    "shapes",
    [
        # Standard attention shape: [batch, num_heads, seq_len, head_dim]
        [
            (1, 8, 128, 64),  # query
            (1, 8, 128, 64),  # key
            (1, 8, 128, 64),  # value
        ],
        # Smaller seq_len
        [
            (1, 4, 32, 64),
            (1, 4, 32, 64),
            (1, 4, 32, 64),
        ],
        # Larger batch
        [
            (2, 8, 64, 64),
            (2, 8, 64, 64),
            (2, 8, 64, 64),
        ],
        # Grouped-Query Attention (GQA): 8 query heads, 2 KV heads (4:1 ratio)
        [
            (1, 8, 128, 64),
            (1, 2, 128, 64),
            (1, 2, 128, 64),
        ],
        # Multi-Query Attention (MQA): 8 query heads, 1 KV head (8:1 ratio)
        [
            (1, 8, 128, 64),
            (1, 1, 128, 64),
            (1, 1, 128, 64),
        ],
        # SDPA decode: query_seq_len=1, standard cache size
        [
            (1, 8, 1, 64),
            (1, 8, 128, 64),
            (1, 8, 128, 64),
        ],
        # SDPA decode: larger batch
        [
            (2, 8, 1, 64),
            (2, 8, 64, 64),
            (2, 8, 64, 64),
        ],
        # SDPA decode: larger cache size
        [
            (1, 8, 1, 64),
            (1, 8, 512, 64),
            (1, 8, 512, 64),
        ],
        # SDPA decode: different head dimension
        [
            (1, 8, 1, 128),
            (1, 8, 256, 128),
            (1, 8, 256, 128),
        ],
        # GQA decode: 32 query heads, 8 KV heads (4:1 ratio), query_seq_len=1
        [
            (1, 32, 1, 128),
            (1, 8, 256, 128),
            (1, 8, 256, 128),
        ],
        # MQA decode: 8 query heads, 1 KV head, query_seq_len=1
        [
            (1, 8, 1, 64),
            (1, 1, 128, 64),
            (1, 1, 128, 64),
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 3])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_basic(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test basic Scaled Dot Product Attention pattern fusion without mask.
    This test implements the SDPA operation as a sequence of TTIR ops:
    - Transpose key
    - Q @ K^T matmul
    - Scale by 1/sqrt(head_dim)
    - Softmax
    - @ V matmul

    Expected to fuse into ttnn.scaled_dot_product_attention
    """

    def sdpa_basic(
        query: Operand,
        key: Operand,
        value: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensors
        query_data = torch.randn(shapes[0], dtype=dtypes[0])
        key_data = torch.randn(shapes[1], dtype=dtypes[1])
        value_data = torch.randn(shapes[2], dtype=dtypes[2])

        head_dim = shapes[0][-1]
        scale = 1.0 / math.sqrt(head_dim)

        golden_output = build_torch_golden(
            query_data, key_data, value_data, scale=scale
        )

        result = build_ttir(
            query, key, value, builder, scale=scale, unit_attrs=unit_attrs
        )

        builder.set_goldens(
            {query: query_data, key: key_data, value: value_data},
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        sdpa_basic,
        shapes,
        dtypes,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "scaled_dot_product_attention")


@pytest.mark.parametrize(
    "shapes",
    [
        # Query, Key, Value, and mask shapes
        [
            (1, 8, 64, 64),  # query
            (1, 8, 64, 64),  # key
            (1, 8, 64, 64),  # value
            (1, 1, 64, 64),  # attention mask (broadcasts over heads)
        ],
        [
            (1, 4, 32, 64),  # query
            (1, 4, 32, 64),  # key
            (1, 4, 32, 64),  # value
            (1, 1, 32, 32),  # attention mask
        ],
        # Query seq_len not divisible by 32
        [
            (1, 8, 63, 64),  # query
            (1, 8, 64, 64),  # key
            (1, 8, 64, 64),  # value
            (1, 1, 63, 64),  # attention mask
        ],
        # Key/Value seq_len not divisible by 32
        [
            (1, 8, 64, 64),  # query
            (1, 8, 50, 64),  # key
            (1, 8, 50, 64),  # value
            (1, 1, 64, 50),  # attention mask
        ],
        # Both query and key/value seq_len not divisible by 32
        [
            (1, 4, 100, 64),  # query
            (1, 4, 77, 64),  # key
            (1, 4, 77, 64),  # value
            (1, 1, 100, 77),  # attention mask
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 4])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_with_mask(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test Scaled Dot Product Attention pattern fusion with attention mask.
    This test implements the SDPA operation as a sequence of TTIR ops:
    - Transpose key
    - Q @ K^T matmul
    - Scale by 1/sqrt(head_dim)
    - Add attention mask
    - Softmax
    - @ V matmul

    Expected to fuse into ttnn.scaled_dot_product_attention
    """

    def sdpa_with_mask(
        query: Operand,
        key: Operand,
        value: Operand,
        attention_mask: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        query_data = torch.randn(shapes[0], dtype=dtypes[0])
        key_data = torch.randn(shapes[1], dtype=dtypes[1])
        value_data = torch.randn(shapes[2], dtype=dtypes[2])

        # Attention mask typically has large negative values for masked positions
        mask_data = torch.randn(shapes[3], dtype=dtypes[3]) * -1000.0

        head_dim = shapes[0][-1]
        scale = 1.0 / math.sqrt(head_dim)

        golden_output = build_torch_golden(
            query_data, key_data, value_data, scale=scale, attention_mask=mask_data
        )

        result = build_ttir(
            query,
            key,
            value,
            builder,
            scale=scale,
            attention_mask=attention_mask,
            unit_attrs=unit_attrs,
        )

        builder.set_goldens(
            {
                query: query_data,
                key: key_data,
                value: value_data,
                attention_mask: mask_data,
            },
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        sdpa_with_mask,
        shapes,
        dtypes,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "scaled_dot_product_attention")


@pytest.mark.parametrize(
    "shapes",
    [
        # Query seq_len not divisible by 32
        [
            (1, 8, 63, 64),  # query
            (1, 8, 64, 64),  # key
            (1, 8, 64, 64),  # value
            (1, 1, 63, 64),  # attention mask
        ],
        # Key/Value seq_len not divisible by 32
        [
            (1, 8, 64, 64),  # query
            (1, 8, 50, 64),  # key
            (1, 8, 50, 64),  # value
            (1, 1, 64, 50),  # attention mask
        ],
        # Both query and key/value seq_len not divisible by 32
        [
            (1, 4, 100, 64),  # query
            (1, 4, 77, 64),  # key
            (1, 4, 77, 64),  # value
            (1, 1, 100, 77),  # attention mask
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 4])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    # Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/32503
    reason="SDPA with non-32-divisible shapes fails without ttnn-workaround pass"
)
def test_sdpa_with_mask_no_workaround(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test Scaled Dot Product Attention pattern fusion with attention mask and
    non-32-divisible sequence lengths, with ttnn-workaround pass disabled.
    """

    def sdpa_with_mask_no_workaround(
        query: Operand,
        key: Operand,
        value: Operand,
        attention_mask: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        query_data = torch.randn(shapes[0], dtype=dtypes[0])
        key_data = torch.randn(shapes[1], dtype=dtypes[1])
        value_data = torch.randn(shapes[2], dtype=dtypes[2])

        # Attention mask typically has large negative values for masked positions
        mask_data = torch.randn(shapes[3], dtype=dtypes[3]) * -1000.0

        head_dim = shapes[0][-1]
        scale = 1.0 / math.sqrt(head_dim)

        golden_output = build_torch_golden(
            query_data, key_data, value_data, scale=scale, attention_mask=mask_data
        )

        result = build_ttir(
            query,
            key,
            value,
            builder,
            scale=scale,
            attention_mask=attention_mask,
            unit_attrs=unit_attrs,
        )

        builder.set_goldens(
            {
                query: query_data,
                key: key_data,
                value: value_data,
                attention_mask: mask_data,
            },
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        sdpa_with_mask_no_workaround,
        shapes,
        dtypes,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )

    assert check_op(output, "scaled_dot_product_attention")


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (32, 32, 1, 64),
            (32, 8, 128, 64),
            (32, 8, 128, 64),
            (1, 1, 1, 128),
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 4])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    # Metal issue reference: https://github.com/tenstorrent/tt-metal/issues/32641
    reason="SDPA decode without program config set fails (without ttnn-workaround pass)"
)
def test_sdpa_decode_no_workaround(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test Scaled Dot Product Attention Decode without program config set,
    with ttnn-workaround pass disabled.
    """

    def test_sdpa_decode_no_workaround(
        query: Operand,
        key: Operand,
        value: Operand,
        attention_mask: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        query_data = torch.randn(shapes[0], dtype=dtypes[0])
        key_data = torch.randn(shapes[1], dtype=dtypes[1])
        value_data = torch.randn(shapes[2], dtype=dtypes[2])

        # Attention mask typically has large negative values for masked positions
        mask_data = torch.randn(shapes[3], dtype=dtypes[3]) * -1000.0

        head_dim = shapes[0][-1]
        scale = 1.0 / math.sqrt(head_dim)

        golden_output = build_torch_golden(
            query_data, key_data, value_data, scale=scale, attention_mask=mask_data
        )

        result = build_ttir(
            query,
            key,
            value,
            builder,
            scale=scale,
            attention_mask=attention_mask,
            unit_attrs=unit_attrs,
        )

        builder.set_goldens(
            {
                query: query_data,
                key: key_data,
                value: value_data,
                attention_mask: mask_data,
            },
            {result: golden_output},
        )
        return result

    output = compile_and_execute_ttir(
        test_sdpa_decode_no_workaround,
        shapes,
        dtypes,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )

    assert check_op(output, "scaled_dot_product_attention")
