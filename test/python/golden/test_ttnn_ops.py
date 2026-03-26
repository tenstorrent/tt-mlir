# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
from typing import List, Optional, Tuple
from collections import OrderedDict

from ttmlir.dialects import ttnn

from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn
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


@pytest.mark.parametrize("shape", [(1, 32, 32)], ids=shape_str)
@pytest.mark.parametrize("repeat_dims", [[32, 1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
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
        (1, 32, 32, 32),
        (1, 32, 32, 1),
        (32, 32, 1, 1),
        (1, 32, 32),
        (32, 32),
        (32, 40),
        (40, 32),
        (1, 1, 32, 32, 32),
        (1, 1, 1, 1, 1, 1, 32, 32, 32),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
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

    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        pytest.skip(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions."
        )

    shard_dims_candidate = list(range(rank_in - rank_mesh, rank_in))
    for d, factor in zip(shard_dims_candidate, mesh_shape):
        if test_shape[d] < factor:
            pytest.skip(
                f"Tensor dim {d} (size {test_shape[d]}) too small to shard "
                f"by factor {factor}"
            )

    shard_dims = list(range(rank_in - rank_mesh, rank_in))

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTNNBuilder):
        @builder.func([full_input_shape], [dtype], host_inputs=True)
        def all_gather(in0: Operand, builder: TTNNBuilder):
            device = builder.get_device()

            distributed = builder.distribute_tensor(
                in0,
                device=device,
                shard_dims=shard_dims,
            )
            tilized = builder.to_layout(distributed, layout=ttnn.Layout.Tile)
            on_device = builder.to_device(tilized, device=device)

            gathered = builder.all_gather(
                on_device,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
            )

            from_dev = builder.from_device(gathered)
            untilized = builder.to_layout(from_dev, layout=ttnn.Layout.RowMajor)
            return builder.aggregate_tensor(
                untilized,
                device=device,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttnn(
        module,
        custom_pipeline=(
            "ttcore-mark-functions-as-forward,"
            "ttcore-wrap-device-module,"
            "ttcore.device_module(builtin.module("
            "ttnn-configure-ccl-ops,ttnn-deallocate))"
        ),
        **get_request_kwargs(request),
        target="ttnn",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )
