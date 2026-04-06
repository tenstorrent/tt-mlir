# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import (
    Marks,
    shapes_list_str,
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")


reduction_op_names = [
    "argmax",
    "max",
    "mean",
    "min",
    "prod",
    "reduce_and" | Marks(pytest.mark.skip(reason="Builder test not supported #5792")),
    "reduce_or" | Marks(pytest.mark.skip(reason="Builder test not supported #5792")),
    "sum",
]


keep_dim_options = [
    True,
    False,
]


dim_arg_options = [
    [0],
    [2],
    [1, 2],
    None,
]


@pytest.mark.parametrize("shapes", [[(32, 128, 128)], [(1, 1, 1)]], ids=shapes_list_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("keep_dim", keep_dim_options)
@pytest.mark.parametrize("dim_arg", dim_arg_options)
@pytest.mark.parametrize("reduction_op_name", reduction_op_names)
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_reduction_ops(
    shapes,
    dtype: torch.dtype,
    keep_dim: bool,
    dim_arg: Optional[List[int]],
    reduction_op_name: str,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype])
        def reduction_op_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            reduction_op_builder_map = {
                "argmax": builder.argmax,
                "max": builder.max,
                "mean": builder.mean,
                "min": builder.min,
                "prod": builder.prod,
                "reduce_and": builder.reduce_and,
                "reduce_or": builder.reduce_or,
                "sum": builder.sum,
            }

            reduction_func = reduction_op_builder_map.get(reduction_op_name)

            return reduction_func(in0, dim_arg=dim_arg, keep_dim=keep_dim)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


reduction_op_cpu_hoisted_names = [
    "argmax",
    "max",
    "mean",
    "min",
    "prod",
    "reduce_and" | Marks(pytest.mark.skip(reason="Builder test not supported #5792")),
    "reduce_or" | Marks(pytest.mark.skip(reason="Builder test not supported #5792")),
    "sum",
]


@x86_only
@pytest.mark.parametrize("shape", [(32, 128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("keep_dim", keep_dim_options)
@pytest.mark.parametrize("dim_arg", dim_arg_options)
@pytest.mark.parametrize("reduction_op_name", reduction_op_cpu_hoisted_names)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
def test_reduction_cpu_hoisted_ops(
    shape,
    dtype: torch.dtype,
    keep_dim: bool,
    dim_arg: Optional[List[int]],
    reduction_op_name: str,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def reduction_op_cpu_hoisted_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            reduction_op_builder_map = {
                "argmax": builder.argmax,
                "max": builder.max,
                "mean": builder.mean,
                "min": builder.min,
                "prod": builder.prod,
                "reduce_and": builder.reduce_and,
                "reduce_or": builder.reduce_or,
                "sum": builder.sum,
            }

            reduction_func = reduction_op_builder_map.get(reduction_op_name)

            return reduction_func(
                in0,
                dim_arg=dim_arg,
                keep_dim=keep_dim,
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@pytest.mark.parametrize(
    "shapes, dim",
    [
        pytest.param([(4, 4, 128, 128)], 1, id="rank4_dim1"),
        pytest.param([(4, 4, 128, 128)], 2, id="rank4_dim2"),
        pytest.param([(128,)], 0, id="rank1_dim0"),
        pytest.param([(4, 128)], 0, id="rank2_dim0"),
        pytest.param([(4, 4, 128)], 1, id="rank3_dim1"),
        pytest.param([(4, 4, 128)], 2, id="rank3_dim2"),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_cumsum(shapes: List[Shape], dim: int, request, target, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def cumsum(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.cumsum(in0, dim=dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(4, 4, 32, 32)], 1),
        ([(2, 8, 16, 16)], 0),
        ([(4, 4, 32, 32)], -1),
    ],
    ids=["dim1", "dim0", "dim_negative"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
def test_hoisted_cumsum(
    shapes: List[Shape],
    dim: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def hoisted_cumsum(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.cumsum(in0, dim=dim, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
    )
