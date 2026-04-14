# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# End-to-end golden tests for ttir.constant, ttir.ones, and ttir.zeros through
# the TTMetal pipeline. Each test compiles a TTIR module and runs it on device,
# verifying the output matches a torch golden.

import pytest
import torch
from typing import List

from conftest import get_request_kwargs
from test_utils import SkipIf, shape_str

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")

_CONSTANT_ONES_ZEROS_SHAPES = [(128, 128), (128,), (1, 128, 128)]


@pytest.mark.parametrize("shape", _CONSTANT_ONES_ZEROS_SHAPES, ids=shape_str)
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


@pytest.mark.parametrize("shape", _CONSTANT_ONES_ZEROS_SHAPES, ids=shape_str)
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
def test_ones(
    shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([], [])
        def ones_fn(
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.ones(shape, dtype, unit_attrs=unit_attrs)

    kwargs = {
        "target": target,
        **get_request_kwargs(request),
        "device": device,
        "atol": 0,
        "check_atol": True,
    }
    compile_and_execute_ttir(module, **kwargs)


@pytest.mark.parametrize("shape", _CONSTANT_ONES_ZEROS_SHAPES, ids=shape_str)
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
def test_zeros(
    shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([], [])
        def zeros_fn(
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.zeros(shape, dtype, unit_attrs=unit_attrs)

    kwargs = {
        "target": target,
        **get_request_kwargs(request),
        "device": device,
        "atol": 0,
        "check_atol": True,
    }
    compile_and_execute_ttir(module, **kwargs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.int32 | SkipIf("sim")],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_constant_unary(shape, dtype: torch.dtype, target: str, request, device):
    if dtype.is_floating_point:
        bias = torch.full(shape, 1.25, dtype=dtype)
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
@pytest.mark.parametrize("target", ["ttmetal"])
def test_constant_binary(
    shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    if dtype.is_floating_point:
        low, high = -4.0, 4.0
        midpoint = (low + high) / 2
        input_tensor = torch.rand(shape, dtype=dtype) * (high - low) + low
        const_tensor = torch.full(shape, midpoint, dtype=dtype)
    else:
        low, high = -100, 100
        midpoint = (low + high) // 2
        input_tensor = torch.randint(low, high + 1, shape, dtype=dtype)
        const_tensor = torch.full(shape, midpoint, dtype=dtype)

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def constant_binary(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: List[str] = None,
        ):
            builder.set_goldens(inputs={in0: input_tensor})
            c = builder.constant(const_tensor, unit_attrs=unit_attrs)
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
