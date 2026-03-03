# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_and_execute_ttir,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
)
from ttmlir.dialects import ttir

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
@pytest.mark.xfail(
    reason="Batched input not supported when bias exists (linear operation). https://github.com/tenstorrent/tt-metal/issues/31634"
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
        # Query seq_len not divisible by 32
        pytest.param(
            [
                (1, 8, 63, 64),  # query
                (1, 8, 64, 64),  # key
                (1, 8, 64, 64),  # value
                (1, 1, 63, 64),  # attention mask
            ],
            marks=pytest.mark.xfail(
                reason="SDPA with non-32-divisible seq_len fails without ttnn-workaround pass. Metal issue: https://github.com/tenstorrent/tt-metal/issues/32503"
            ),
        ),
        # Key/Value seq_len not divisible by 32
        pytest.param(
            [
                (1, 8, 64, 64),  # query
                (1, 8, 50, 64),  # key
                (1, 8, 50, 64),  # value
                (1, 1, 64, 50),  # attention mask
            ],
            marks=pytest.mark.xfail(
                reason="SDPA with non-32-divisible seq_len fails without ttnn-workaround pass. Metal issue: https://github.com/tenstorrent/tt-metal/issues/32503"
            ),
        ),
        # Both query and key/value seq_len not divisible by 32
        pytest.param(
            [
                (1, 4, 100, 64),  # query
                (1, 4, 77, 64),  # key
                (1, 4, 77, 64),  # value
                (1, 1, 100, 77),  # attention mask
            ],
            marks=pytest.mark.xfail(
                reason="SDPA with non-32-divisible seq_len fails without ttnn-workaround pass. Metal issue: https://github.com/tenstorrent/tt-metal/issues/32503"
            ),
        ),
        # Head dim not divisible by 32
        pytest.param(
            [
                (1, 8, 64, 50),  # query
                (1, 8, 64, 50),  # key
                (1, 8, 64, 50),  # value
                (1, 1, 64, 64),  # attention mask
            ],
            marks=pytest.mark.xfail(
                reason="SDPA with non-32-divisible seq_len fails without ttnn-workaround pass. Metal issue: https://github.com/tenstorrent/tt-metal/issues/33434"
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
                reason="SDPA with non-32-divisible seq_len and head_dim fails without ttnn-workaround pass. Metal issues: https://github.com/tenstorrent/tt-metal/issues/32503, https://github.com/tenstorrent/tt-metal/issues/33434"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 4])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sdpa_with_mask_no_workaround(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    """
    Test Scaled Dot Product Attention pattern fusion with attention mask and
    non-32-divisible sequence lengths, with ttnn-workaround pass disabled.
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
            query_data = torch.randn(shapes[0], dtype=dtypes[0])
            key_data = torch.randn(shapes[1], dtype=dtypes[1])
            value_data = torch.randn(shapes[2], dtype=dtypes[2])

            mask_data = torch.triu(
                torch.full(shapes[3], float("-inf"), dtype=dtypes[3]), diagonal=1
            )

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
        module,
        target=target,
        **get_request_kwargs(request),
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

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
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

            mask_data = torch.triu(
                torch.full(shapes[3], float("-inf"), dtype=dtypes[3]), diagonal=1
            )

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
        module,
        target=target,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )

    assert check_op(output, "scaled_dot_product_attention")


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
