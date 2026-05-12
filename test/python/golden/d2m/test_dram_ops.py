# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Optional

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")

DRAM_PIPELINE_OPTIONS = [
    "default-input-memspace=dram",
    "default-output-memspace=dram",
]

DRAM_SHAPES = [
    (1, 16),
    (16, 16),
    (32, 32),
    (32, 64),
    (64, 128),
    (128, 128),
    (256, 512),
    (512, 1024),
    (2048, 2048),
]


def compile_dram_test(module, request, device, target: str):
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=list(DRAM_PIPELINE_OPTIONS),
    )


@pytest.mark.parametrize("shape", DRAM_SHAPES, ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_unary_relu(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def relu(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[list[str]] = None
        ):
            return builder.relu(in0, unit_attrs=unit_attrs)

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape", DRAM_SHAPES, ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_binary_add(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def add(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[list[str]] = None,
        ):
            return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape", DRAM_SHAPES, ids=shape_str)
@pytest.mark.parametrize(
    "dtype,scalar_value",
    [
        (torch.float32, 2.5),
        (torch.bfloat16, 5.0),
        pytest.param(torch.int32, 5, marks=pytest.mark.skip_config(["sim"])),
    ],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_binary_add_scalar(
    shape: Shape, dtype: torch.dtype, scalar_value, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def add_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[list[str]] = None
        ):
            scalar = builder.constant(torch.full(shape, scalar_value, dtype=dtype))
            return builder.add(in0, scalar, unit_attrs=unit_attrs)

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape", DRAM_SHAPES, ids=shape_str)
@pytest.mark.parametrize(
    "dtype,min_value,max_value",
    [
        (torch.float32, -0.5, 0.8),
        (torch.bfloat16, -0.5, 0.8),
        pytest.param(torch.int32, 0, 3, marks=pytest.mark.skip_config(["sim"])),
    ],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_ternary_clamp_tensor(
    shape: Shape, dtype: torch.dtype, min_value, max_value, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def clamp_tensor(
            in0: Operand,
            min_tensor: Operand,
            max_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[list[str]] = None,
        ):
            if dtype == torch.int32:
                input_golden = torch.randint(-5, 10, shape, dtype=dtype)
            else:
                input_golden = torch.rand(shape, dtype=dtype) * 2 - 1
            min_golden = torch.full(shape, min_value, dtype=dtype)
            max_golden = torch.full(shape, max_value, dtype=dtype)
            builder.set_goldens(
                inputs={
                    in0: input_golden,
                    min_tensor: min_golden,
                    max_tensor: max_golden,
                }
            )
            return builder.clamp_tensor(
                in0, min_tensor, max_tensor, unit_attrs=unit_attrs
            )

    compile_dram_test(module, request, device, target)


@pytest.mark.parametrize("shape", DRAM_SHAPES, ids=shape_str)
@pytest.mark.parametrize(
    "dtype,min_value,max_value",
    [
        (torch.float32, -0.5, 0.8),
        (torch.bfloat16, -0.5, 0.8),
        pytest.param(torch.int32, 0, 3, marks=pytest.mark.skip_config(["sim"])),
    ],
    ids=["f32", "bf16", "i32"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_dram_ternary_clamp_scalar(
    shape: Shape, dtype: torch.dtype, min_value, max_value, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def clamp_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[list[str]] = None
        ):
            if dtype == torch.int32:
                input_golden = torch.randint(-5, 10, shape, dtype=dtype)
            else:
                input_golden = torch.rand(shape, dtype=dtype) * 2 - 1
            builder.set_goldens(inputs={in0: input_golden})
            return builder.clamp_scalar(
                in0, min_arg=min_value, max_arg=max_value, unit_attrs=unit_attrs
            )

    compile_dram_test(module, request, device, target)
