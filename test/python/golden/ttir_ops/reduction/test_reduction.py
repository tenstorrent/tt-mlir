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
    "sum",
]


keep_dim_options = [
    True,
]


dim_arg_options = [
    [2],
]


@pytest.mark.parametrize("shapes", [[(32, 128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("keep_dim", keep_dim_options)
@pytest.mark.parametrize("dim_arg", dim_arg_options)
@pytest.mark.parametrize("reduction_op_name", reduction_op_names)
@pytest.mark.parametrize("iterations", list(range(1, 1000)))
@pytest.mark.parametrize("target", ["emitc"])
def test_reduction_ops(
    shapes,
    dtype: torch.dtype,
    keep_dim: bool,
    dim_arg: Optional[List[int]],
    reduction_op_name: str,
    iterations: str,
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
