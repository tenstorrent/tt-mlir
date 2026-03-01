# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
from typing import Callable, List, Optional, Tuple
from collections import OrderedDict

from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn, build_module
from test_utils import shape_str, shapes_list_str

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("max_arg,min_arg", [(3.0, 2.0)])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([shape], [torch.float32])
        def clamp_scalar(
            in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
        ):
            print(f"Clamping with min: {min_arg}, max: {max_arg}")
            return builder.clamp_scalar(
                in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes", [[(32, 64), (32, 64), (32, 64)]], ids=shapes_list_str
)
def test_clamp_tensor(shapes: List[Shape], request, device):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def clamp_tensor(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 32, 32), (2, 16, 16), (1, 1, 64)], ids=shape_str)
@pytest.mark.parametrize("repeat_dims", [[32, 1, 1], [1, 2, 2], [2, 3, 4], [1, 1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
def test_repeat(shape: Shape, repeat_dims: List[int], dtype, request, device):
    def module(builder: TTNNBuilder):
        @builder.func([shape], [dtype])
        def repeat(
            in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.repeat(in0, repeat_dims, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 8, 1, 12, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [1])
def test_repeat_interleave(
    shapes: List[Shape], repeats: int, dim: int, request, device
):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.float32])
        def repeat_interleave(
            in0: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.repeat_interleave(
                in0, repeats=repeats, dim=dim, unit_attrs=unit_attrs
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (32, 128),
            (16, 128),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
def test_concat(shapes: List[Shape], dim: int, request, device):
    # Create a wrapper function that captures dim
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def concat_wrapper(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.concat([in0, in1, in2], dim=dim, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 128), (128, 256)],
        [(32, 64), (64, 128)],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_matmul(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [dtype, dtype])
        def matmul(
            in0: Operand,
            in1: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 128), (256, 128)],
        [(64, 128), (256, 128), (256,)],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_linear(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTNNBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def linear(
            in0: Operand,
            in1: Operand,
            bias_or_builder,
            builder_or_none=None,
            unit_attrs: Optional[List[str]] = None,
        ):
            if builder_or_none is not None:
                bias = bias_or_builder
                builder = builder_or_none
            else:
                bias = None
                builder = bias_or_builder

            return builder.linear(
                in0, in1, bias=bias, transpose_b=True, unit_attrs=unit_attrs
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 32, 32),
        (1, 32, 32, 32),
        (32, 32, 1, 1),
        (1, 1, 64, 128),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
@pytest.mark.parametrize("all_gather_dim", range(4))
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_gather(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    all_gather_dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if all_gather_dim >= len(test_shape):
        pytest.skip("all_gather_dim is out of range")
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("all_gather across 1 device is meaningless")

    mesh_dict = OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])])

    expected_output_shape = list(test_shape)
    expected_output_shape[all_gather_dim] *= mesh_shape[cluster_axis]

    def module(builder: TTNNBuilder):
        @builder.func([test_shape], [dtype])
        def all_gather(
            in0: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.all_gather(
                in0,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
                unit_attrs=unit_attrs,
            )

    mlir_module, builder = build_module(
        module,
        "ttnn",
        mesh_name="mesh",
        mesh_dict=mesh_dict,
    )

    module_str = str(mlir_module)
    assert "ttnn.all_gather" in module_str
    assert f"all_gather_dim = {all_gather_dim} : si32" in module_str
    assert f"cluster_axis = {cluster_axis} : ui32" in module_str

    input_shape_str = "x".join(str(d) for d in test_shape)
    output_shape_str = "x".join(str(d) for d in expected_output_shape)
    assert f"tensor<{input_shape_str}" in module_str, (
        f"Expected input shape {input_shape_str} in IR"
    )
    assert f"tensor<{output_shape_str}" in module_str, (
        f"Expected output shape {output_shape_str} in IR"
    )
