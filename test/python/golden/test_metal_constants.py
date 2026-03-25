# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from conftest import get_request_kwargs
from test_utils import SkipIf, shape_str

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.int32 | SkipIf("sim"),
    ],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_constant(
    shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """End-to-end ttir.constant through the TTMetal pipeline (compile + device run)."""

    def module(builder: TTIRBuilder):
        if dtype.is_floating_point:
            tensor = torch.full(shape, 1.25, dtype=dtype)
        else:
            tensor = torch.full(shape, 10, dtype=dtype)

        @builder.func([], [])
        def constant_fn(
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.constant(tensor, unit_attrs=unit_attrs)

    kwargs = {
        "target": target,
        **get_request_kwargs(request),
        "device": device,
    }
    if dtype == torch.bfloat16:
        kwargs["atol"] = 0.02
        kwargs["check_atol"] = True
    compile_and_execute_ttir(module, **kwargs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.int32 | SkipIf("sim")],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_constant_unary(shape, dtype: torch.dtype, target: str, request, device):
    """Unary eltwise on (input + ttir.constant); exercises const + add + abs."""

    if dtype.is_floating_point:
        bias = torch.full(shape, 1.23, dtype=dtype)
    else:
        bias = torch.full(shape, 7, dtype=dtype)

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def constant_unary(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            c = builder.constant(bias, unit_attrs=unit_attrs)
            return builder.abs(c)

    kwargs = {
        "target": target,
        **get_request_kwargs(request),
        "device": device,
    }
    if dtype == torch.bfloat16:
        kwargs["atol"] = 0.02
        kwargs["check_atol"] = True
    compile_and_execute_ttir(module, **kwargs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.int32 | SkipIf("sim")],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal" | SkipIf("sim")])
def test_constant_binary(
    shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Binary maximum between a live input and a ttir.constant tensor."""

    if dtype.is_floating_point:
        floor = torch.full(shape, -1.0, dtype=dtype)
    else:
        floor = torch.full(shape, -100, dtype=dtype)

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def constant_binary(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            c = builder.constant(floor, unit_attrs=unit_attrs)
            return builder.maximum(in0, c, unit_attrs=unit_attrs)

    kwargs = {
        "target": target,
        **get_request_kwargs(request),
        "device": device,
    }
    if dtype == torch.bfloat16:
        kwargs["atol"] = 0.02
        kwargs["check_atol"] = True
    compile_and_execute_ttir(module, **kwargs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.int32 | SkipIf("sim")],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_constant_ternary(
    shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Ternary where(cond, x, ttir.constant) with the false branch from a constant."""

    if dtype.is_floating_point:
        value = torch.full(shape, -2.0, dtype=dtype)
    else:
        value = torch.full(shape, -2, dtype=dtype)

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def constant_ternary(
            cond: Operand,
            x: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            cond_tensor = torch.randint(0, 2, shape, dtype=torch.int32).to(dtype)
            if dtype.is_floating_point:
                x_tensor = torch.randn(shape, dtype=dtype) * 0.5
            else:
                x_tensor = torch.randint(-30, 30, shape, dtype=dtype)
            builder.set_goldens(inputs={cond: cond_tensor, x: x_tensor})
            c = builder.constant(value, unit_attrs=unit_attrs)
            return builder.where(cond, x, c, unit_attrs=unit_attrs)

    kwargs = {
        "target": target,
        **get_request_kwargs(request),
        "device": device,
    }
    if dtype == torch.bfloat16:
        kwargs["atol"] = 0.02
        kwargs["check_atol"] = True
    compile_and_execute_ttir(module, **kwargs)
