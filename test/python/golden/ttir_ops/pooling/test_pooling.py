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
from test_utils import (
    shapes_list_str,
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")


# Max pool 2d tests


@pytest.mark.parametrize("shape", [(1, 32, 32, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode",
    [
        ([3, 3], [1, 1], [1, 1], [0, 0, 0, 0], False),
        ([3, 3], [2, 2], [1, 1], [0, 0, 0, 0], False),
        ([3, 3], [2, 2], [2, 2], [0, 0, 0, 0], False),
        ([3, 3], [2, 2], [2, 2], [1, 1, 1, 1], False),
        ([3, 3], [2, 2], [2, 2], [1, 1, 1, 1], True),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_max_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    target: str,
    request,
    device,
):
    if target == "emitc":
        pytest.skip(
            "EmitC tests are hanging in CI after switching targets (emitPy->emitC). Disabling them to unblock the uplift. See issue: https://github.com/tenstorrent/tt-mlir/issues/7282"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def max_pool2d(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.max_pool2d(
                in0,
                kernel=kernel,
                stride=stride,
                dilation=dilation,
                padding=padding,
                ceil_mode=ceil_mode,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(1, 32, 32, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode",
    [
        ([3, 3], [1, 1], [1, 1], [0, 0, 0, 0], False),
        ([3, 3], [2, 2], [1, 1], [0, 0, 0, 0], False),
        pytest.param(
            [3, 3],
            [2, 2],
            [2, 2],
            [0, 0, 0, 0],
            False,
            marks=pytest.mark.skip(
                reason="Dilation > 1 crashes in TTIRToLinalg with assertion"
            ),
        ),
        pytest.param(
            [3, 3],
            [2, 2],
            [2, 2],
            [1, 1, 1, 1],
            False,
            marks=pytest.mark.skip(
                reason="Dilation > 1 crashes in TTIRToLinalg with assertion"
            ),
        ),
        ([3, 3], [2, 2], [1, 1], [1, 1, 1, 1], True),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_max_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    target: str,
    request,
    device,
):
    """Test hoisted max_pool2d operation"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def hoisted_max_pool2d(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.max_pool2d(
                in0,
                kernel=kernel,
                stride=stride,
                dilation=dilation,
                padding=padding,
                ceil_mode=ceil_mode,
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Avg pool 2d tests


@pytest.mark.parametrize("shape", [(1, 32, 32, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode,count_include_pad",
    [
        ([3, 3], [1, 1], [1, 1], [0, 0, 0, 0], False, True),
        ([3, 3], [2, 2], [1, 1], [0, 0, 0, 0], False, True),
        ([4, 4], [2, 2], [1, 1], [2, 2, 2, 2], False, True),
        ([3, 3], [2, 2], [1, 1], [1, 1, 1, 1], True, True),
        ([4, 4], [2, 2], [1, 1], [2, 2, 2, 2], False, False),
        ([8, 8], [1, 1], [1, 1], [7, 7, 7, 7], False, True),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_avg_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    target: str,
    request,
    device,
):
    if target == "emitc":
        pytest.skip(
            "EmitC tests are hanging in CI after switching targets (emitPy->emitC). Disabling them to unblock the uplift. See issue: https://github.com/tenstorrent/tt-mlir/issues/7282"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def avg_pool2d(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.avg_pool2d(
                in0,
                kernel=kernel,
                stride=stride,
                dilation=dilation,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(1, 32, 32, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode,count_include_pad",
    [
        ([3, 3], [1, 1], [1, 1], [0, 0, 0, 0], False, True),
        ([3, 3], [2, 2], [1, 1], [0, 0, 0, 0], False, True),
        ([4, 4], [2, 2], [1, 1], [2, 2, 2, 2], False, True),
        ([3, 3], [2, 2], [1, 1], [1, 1, 1, 1], True, True),
        ([4, 4], [2, 2], [1, 1], [2, 2, 2, 2], False, False),
        ([8, 8], [1, 1], [1, 1], [7, 7, 7, 7], False, True),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_avg_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    target: str,
    request,
    device,
):
    """Test hoisted avg_pool2d operation"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def hoisted_avg_pool2d(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.avg_pool2d(
                in0,
                kernel=kernel,
                stride=stride,
                dilation=dilation,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Max pool 2d with indices tests


@pytest.mark.parametrize("shape", [(1, 32, 32, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode",
    [
        ([3, 3], [1, 1], [1, 1], [0, 0, 0, 0], False),
        ([3, 3], [2, 2], [1, 1], [0, 0, 0, 0], False),
        ([3, 3], [2, 2], [2, 2], [0, 0, 0, 0], False),
        ([3, 3], [2, 2], [2, 2], [1, 1, 1, 1], False),
        ([3, 3], [2, 2], [2, 2], [1, 1, 1, 1], True),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_max_pool2d_with_indices(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    target: str,
    request,
    device,
):
    if target == "emitc":
        pytest.skip(
            "EmitC tests are hanging in CI after switching targets (emitPy->emitC). Disabling them to unblock the uplift. See issue: https://github.com/tenstorrent/tt-mlir/issues/7282"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype, torch.int64])
        def max_pool2d_with_indices(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.max_pool2d_with_indices(
                in0,
                kernel=kernel,
                stride=stride,
                dilation=dilation,
                padding=padding,
                ceil_mode=ceil_mode,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Global avg pool 2d tests


@pytest.mark.parametrize("shape", [(1, 32, 32, 64), (1, 7, 7, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_global_avg_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    if target == "emitc":
        pytest.skip(
            "EmitC tests are hanging in CI after switching targets (emitPy->emitC). Disabling them to unblock the uplift. See issue: https://github.com/tenstorrent/tt-mlir/issues/7282"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def global_avg_pool2d(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.global_avg_pool2d(
                in0,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(1, 32, 32, 64), (1, 7, 7, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_global_avg_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Test hoisted global_avg_pool2d operation"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def hoisted_global_avg_pool2d(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.global_avg_pool2d(
                in0,
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
