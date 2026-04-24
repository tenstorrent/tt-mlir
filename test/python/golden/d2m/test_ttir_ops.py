# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# ttmetal-only mirrors of multi-backend tests in `test_ttir_ops.py`.

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only, get_request_kwargs

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir, build_module
from builder.base.builder_enums import *
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    SkipIf,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttir")


import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir, build_module
from builder.base.builder_enums import *
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    SkipIf,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_callable_initialization_basic(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    """Basic test demonstrating callable initialization with torch.zeros and torch.ones"""

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def test_with_basic_callables(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_goldens({in0: torch.zeros, in1: torch.ones})
            result = builder.add(in0, in1)
            return result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_callable_initialization_zeros(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def test_with_zeros_init(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            builder.set_goldens({in0: torch.zeros})
            zeros_result = builder.neg(in0, unit_attrs=unit_attrs)
            return zeros_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_callable_initialization_ones(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def test_with_ones_init(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            builder.set_goldens({in0: torch.ones})
            ones_result = builder.neg(in0, unit_attrs=unit_attrs)
            return ones_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_callable_initialization_eye(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def test_with_eye_init(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            def eye_init(s):
                if len(s) == 2 and s[0] == s[1]:
                    return torch.eye(s[0])
                elif len(s) == 2:
                    return torch.eye(s[0], s[1])
                else:
                    raise ValueError(f"torch.eye only supports 2D shapes, got {s}")

            builder.set_goldens({in0: eye_init})
            eye_result = builder.abs(in0, unit_attrs=unit_attrs)
            return eye_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_callable_initialization_mixed(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def test_with_mixed_init(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_goldens({in0: torch.zeros, in1: torch.ones})
            add_result = builder.add(in0, in1)
            return add_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_callable_initialization_custom_lambda(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def test_with_custom_lambda(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            custom_init = lambda s: torch.full(s, 2.0)
            builder.set_goldens({in0: custom_init})
            result = builder.multiply(in0, in0)
            return result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "inputs_shapes,inputs_dtypes",
    [
        pytest.param(
            [(33, 32), (512, 128)],
            [torch.float32] * 2,
        ),
    ],
)
@pytest.mark.parametrize(
    "target",
    ["ttmetal" | Marks(pytest.mark.xfail(reason="Unimplemented ttir.embedding"))],
)
def test_unique_ops(
    inputs_shapes: List[Shape],
    inputs_dtypes: List[torch.dtype],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(inputs_shapes, inputs_dtypes)
        def embedding(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.embedding(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "shape,dtype,start,end,step,dim",
    [
        ((5,), torch.float32, 0, 5, 1, 0),
        ((10,), torch.int32, 0, 10, 1, 0),
        ((8,), torch.float32, 2, 10, 1, 0),
    ],
    ids=["f32_simple", "i32_simple", "f32_offset_start"],
)
@pytest.mark.parametrize("target", ["ttmetal" | SkipIf("sim")])
def test_hoisted_arange(
    shape: Shape,
    dtype: torch.dtype,
    start: int,
    end: int,
    step: int,
    dim: int,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_arange(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.arange(
                shape, dtype, start, end, step, dim, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
