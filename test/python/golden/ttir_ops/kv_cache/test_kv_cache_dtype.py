# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shapes_list_str

pytestmark = pytest.mark.frontend("ttir")


@x86_only
@pytest.mark.parametrize("kv_cache_dtype", ["bfp_bf8"])
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 64, 512)]], ids=shapes_list_str
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16]])
@pytest.mark.parametrize("target", ["ttnn"])
def test_fill_cache_kv_cache_dtype(
    kv_cache_dtype: str,
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def fill_cache(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            result = builder.fill_cache(in0, in1, unit_attrs=unit_attrs)
            # float32 (not bfloat16) to survive same-dtype canonicalization
            return builder.typecast(result, torch.float32)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=[f"experimental-kv-cache-dtype={kv_cache_dtype}"],
    )


@x86_only
@pytest.mark.parametrize("kv_cache_dtype", ["bfp_bf8"])
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 63, 511), (1, 32, 63, 511)]], ids=shapes_list_str
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16]])
@pytest.mark.parametrize("target", ["ttnn"])
def test_fill_cache_kv_cache_dtype_non_padded(
    kv_cache_dtype: str,
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def fill_cache(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            result = builder.fill_cache(in0, in1, unit_attrs=unit_attrs)
            return builder.typecast(result, torch.float32)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=[f"experimental-kv-cache-dtype={kv_cache_dtype}"],
    )


@x86_only
@pytest.mark.parametrize("kv_cache_dtype", ["bfp_bf8"])
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 1, 512), (1,)]], ids=shapes_list_str
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16, torch.int32]])
@pytest.mark.parametrize("target", ["ttnn"])
def test_update_cache_kv_cache_dtype(
    kv_cache_dtype: str,
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def update_cache(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            cache_seq_len = shapes[0][2]
            update_index = torch.randint(0, cache_seq_len, shapes[2], dtype=torch.int32)
            builder.set_goldens(inputs={in2: update_index})
            result = builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)
            # float32 (not bfloat16) to survive same-dtype canonicalization
            return builder.typecast(result, torch.float32)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=[f"experimental-kv-cache-dtype={kv_cache_dtype}"],
    )


@x86_only
@pytest.mark.parametrize("kv_cache_dtype", ["bfp_bf8"])
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 63, 511), (1, 32, 1, 511), (1,)]], ids=shapes_list_str
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16, torch.bfloat16, torch.int32]])
@pytest.mark.parametrize("target", ["ttnn"])
def test_update_cache_kv_cache_dtype_non_padded(
    kv_cache_dtype: str,
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def update_cache(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            cache_seq_len = shapes[0][2]
            update_index = torch.randint(0, cache_seq_len, shapes[2], dtype=torch.int32)
            builder.set_goldens(inputs={in2: update_index})
            result = builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)
            # float32 (not bfloat16) to survive same-dtype canonicalization
            return builder.typecast(result, torch.float32)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=[f"experimental-kv-cache-dtype={kv_cache_dtype}"],
    )
