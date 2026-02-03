# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import Callable, List, Optional
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import (
    Marks,
    shapes_list_str,
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
    [2],
    [1, 2],
    None,
]


@pytest.mark.parametrize("shapes", [[(32, 128, 128)]], ids=shapes_list_str)
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

    if (
        reduction_op_name == "max"
        or reduction_op_name == "min"
        or reduction_op_name == "mean"
        or reduction_op_name == "sum"
    ) and dim_arg is None:
        request.node.add_marker(
            pytest.xfail(
                reason="Fails Golden, see issue https://github.com/tenstorrent/tt-metal/issues/32274"
            )
        )

    # FP32 prod reduction fails due to tt-metal untilize NaN handling.
    # See: https://github.com/tenstorrent/tt-metal/pull/33904
    if reduction_op_name == "prod" and dtype == torch.float32 and target == "ttnn":
        pytest.xfail(
            "FP32 prod reduction fails due to tt-metal untilize NaN handling. "
            "See: https://github.com/tenstorrent/tt-metal/pull/33904"
        )

    if reduction_op_name == "prod" and dim_arg is None and keep_dim is True:
        request.node.add_marker(
            pytest.xfail(
                reason="Fails in ttnn runtime, see issue https://github.com/tenstorrent/tt-metal/issues/32279"
            )
        )

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
    "argmax"
    | Marks(
        pytest.mark.xfail(reason="Not supported in CPU hoisted mode, see issue #5809")
    ),
    "max",
    "mean",
    "min",
    "prod",
    "reduce_and" | Marks(pytest.mark.xfail(reason="Builder test not supported #5792")),
    "reduce_or" | Marks(pytest.mark.xfail(reason="Builder test not supported #5792")),
    "sum",
]


@x86_only
@pytest.mark.parametrize("shapes", [[(32, 128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("keep_dim", keep_dim_options)
@pytest.mark.parametrize("dim_arg", dim_arg_options)
@pytest.mark.parametrize("reduction_op_name", reduction_op_cpu_hoisted_names)
@pytest.mark.parametrize("target", ["ttnn"])
def test_reduction_cpu_hoisted_ops(
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
