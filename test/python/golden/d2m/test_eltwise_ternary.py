# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# ttmetal-only mirrors of multi-backend tests in
# `ttir_ops/eltwise/test_ttir_ternary.py`.

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from d2m.shape_cases import ELEMENTWISE_2D_SHAPES, rotated_params
from test_utils import SkipIf

pytestmark = pytest.mark.frontend("ttir")


# NOTE: test_ternary_ops below covers int32/int64 dtypes (with the int64
# golden-truncation logic baked into module_clamp_tensor), so the int64-only
# variants previously living here have been folded into the full mirror.


# Ternary ops
def module_where(shape: Shape, dtype: torch.dtype):
    def _module_where(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype] * 3)
        def where(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            condition_tensor = torch.randint(0, 2, shape).to(dtype)
            builder.set_goldens(inputs={in0: condition_tensor})
            return builder.where(in0, in1, in2, unit_attrs=unit_attrs)

    return _module_where


def module_clamp_tensor(shape: Shape, dtype: torch.dtype):
    def _module_clamp_tensor(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype] * 3)
        def clamp_tensor(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # int64 on ttmetal is normalized to int32; use int32-range values
            # so truncation is a no-op and golden matches device output.
            if dtype == torch.int64:
                in0_golden = torch.randint(
                    -(2**31), 2**31, shape, dtype=torch.int64
                )
                in1_golden = torch.randint(
                    -(2**31), 2**31, shape, dtype=torch.int64
                )
                in2_golden = torch.randint(
                    -(2**31), 2**31, shape, dtype=torch.int64
                )
                builder.set_goldens({in0: in0_golden, in1: in1_golden, in2: in2_golden})
            return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)

    return _module_clamp_tensor


ternary_ops = [
    module_where,
    module_clamp_tensor,
]

ternary_ops_dtypes = [
    torch.float32,
    torch.bfloat16,
    torch.int32 | SkipIf("sim"),
    torch.int64 | SkipIf("sim"),
    torch.bool | SkipIf("sim"),
]


_TERNARY_OP_PARAMS = rotated_params(
    ELEMENTWISE_2D_SHAPES, ternary_ops, ternary_ops_dtypes, value_order=[0, 2, 1]
) + [
    pytest.param((128,), torch.float32, module_where, id="128-f32-where_1d"),
]


@pytest.mark.parametrize(
    "shape,dtype,test_fn",
    _TERNARY_OP_PARAMS,
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_ternary_ops(
    shape: Shape, dtype: torch.dtype, test_fn: Callable, target: str, request, device
):
    compile_and_execute_ttir(
        test_fn(shape, dtype),
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=[],
    )


# Ternary eltwise ops with implicit broadcasting
ternary_broadcast_shapes = [
    # 2D shapes
    [(128, 128), (1, 128), (128, 128)],
    [(32, 64), (32, 64), (1, 64)],
    [(1, 32), (64, 32), (64, 1)],
    # 3D shapes
    [(1, 16, 32), (8, 16, 32), (8, 16, 32)],
    [(8, 16, 32), (1, 16, 32), (8, 16, 32)],
    [(8, 16, 32), (8, 16, 32), (1, 16, 32)],
    [(8, 16, 32), (1, 1, 32), (1, 1, 32)],
    [(1, 1, 32), (8, 16, 32), (1, 1, 32)],
    [(1, 1, 32), (1, 1, 32), (8, 16, 32)],
    [(1, 16, 32), (8, 1, 32), (8, 16, 1)],
    [(1, 4, 1), (1, 4, 768), (1, 1, 1)],
    # 4D shapes
    [(1, 1, 1, 4), (1, 1, 1, 1), (1, 1, 1, 1)],
]

ternary_broadcast_dtypes = [
    pytest.param((torch.float32, torch.float32, torch.float32), id="f32-f32-f32"),
    pytest.param((torch.float32, torch.int32, torch.int32), id="f32-i32-i32"),
    pytest.param((torch.bfloat16, torch.bfloat16, torch.bfloat16), id="bf16-bf16-bf16"),
]


@pytest.mark.parametrize(
    "shapes,input_dtypes",
    rotated_params(ternary_broadcast_shapes, ternary_broadcast_dtypes),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_ternary_eltwise_ops_implicit_broadcast(
    shapes: List[Shape],
    input_dtypes: Tuple[torch.dtype, torch.dtype, torch.dtype],
    target: str,
    request,
    device,
):
    is_non_2d = any(len(shape) != 2 for shape in shapes)
    if is_non_2d:
        pytest.xfail(
            "non-2D shapes bcast not yet supported on ttmetal backend, issue here: https://github.com/tenstorrent/tt-mlir/issues/3023"
        )
    if input_dtypes[0] != input_dtypes[1]:
        pytest.xfail(
            "different input dtypes not yet supported on ttmetal backend, issue here: https://github.com/tenstorrent/tt-mlir/issues/6289"
        )

    dtype1, dtype2, dtype3 = input_dtypes

    def module_implicit_broadcast(builder: TTIRBuilder):
        @builder.func(shapes, [dtype1, dtype2, dtype3])
        def where(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            condition_tensor = torch.randint(0, 2, shapes[0], dtype=dtype1)
            builder.set_goldens(inputs={in0: condition_tensor})
            return builder.where(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module_implicit_broadcast,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


clamp_scalar_cases = [
    pytest.param(0.8, -0.5, torch.float32, id="f32"),
    pytest.param(0.8, -0.5, torch.bfloat16, id="bf16"),
    pytest.param(3, 0, torch.int32, marks=pytest.mark.skip_config(["sim"]), id="i32"),
    pytest.param(3, 0, torch.int64, marks=pytest.mark.skip_config(["sim"]), id="i64"),
]


@pytest.mark.parametrize(
    "shape,dtype,max_arg,min_arg",
    rotated_params(ELEMENTWISE_2D_SHAPES, clamp_scalar_cases, value_order=[0, 3, 1, 2]),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_clamp_scalar(
    shape: Shape, dtype: torch.dtype, max_arg, min_arg, target: str, request, device
):
    def module_clamp_scalar(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def clamp_scalar(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            if dtype == torch.int32 or dtype == torch.int64:
                input_tensor = torch.randint(-5, 10, shape, dtype=torch.int32)
            else:
                input_tensor = torch.rand(shape, dtype=dtype) * 2 - 1
            builder.set_goldens(inputs={in0: input_tensor})
            return builder.clamp_scalar(
                in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module_clamp_scalar,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
