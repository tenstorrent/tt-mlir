# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
from typing import Callable, List, Optional, Tuple
from collections import OrderedDict

from builder.base.builder_utils import (
    Operand,
    Shape,
    run_ttir_pipeline,
    create_custom_ttir_pipeline_fn,
)
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn, build_module
from builder.base.builder_enums import MeshShardDirection, MeshShardType
from test_utils import shape_str, shapes_list_str, make_shard_shape

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
        (1, 32, 32, 32),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("all_gather_dim", range(1))
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_all_gather(
    test_shape: Shape,
    mesh_shape,
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
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module2(builder: TTNNBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_gather(in0: Operand, builder: TTNNBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            all_gather0 = builder.all_gather(
                in_shard,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
            )

            return builder.mesh_shard(
                all_gather0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    def module3(builder: TTNNBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_gather(in0: Operand, builder: TTNNBuilder):
            # in1 = builder.abs(in0)
            a = builder.to_layout(in0)  # , layout="shard_x")
            in_shard = builder.distribute_tensor(
                a,
                shard_dims=shard_dims,
                shard_shape=shard_shape,
            )

            all_gather0 = builder.all_gather(
                in_shard,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
            )

            return builder.aggregate_tensor(
                all_gather0,
                shard_dims=shard_dims,
                shard_shape=shard_shape,
            )

    def module(builder: TTNNBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_gather(in0: Operand, builder: TTNNBuilder):
            builder.preshard_arg(in0, shard_dims=shard_dims)

            return builder.all_gather(
                in0,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
            )

    # TODO: compile_and_execute_ttnn does not yet support CCL ops because the
    # TTNN builder's compilation pipeline (ttir-to-ttnn-backend-pipeline) is
    # designed for TTIR→TTNN conversion and lacks the data-movement ops
    # (to_device, to_layout, from_device) required for multi-device execution.
    # Use build_module for IR-level verification until a dedicated TTNN
    # compilation pipeline is implemented. See follow-up issue.
    """
    module_m, builder = build_module(
        module,
        "ttnn",
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )

    custom_pipeline = create_custom_ttir_pipeline_fn(f"ttir-to-ttnn-backend-pipeline", print_ir=True)

    module_m = run_ttir_pipeline(
        module_m,
        custom_pipeline,
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )
    """
    compile_and_execute_ttnn(
        module3,
        custom_pipeline="ttcore-mark-functions-as-forward,ttcore-wrap-device-module,ttnn-configure-ccl-ops",
        **get_request_kwargs(request),
        target="ttnn",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )
    print(gr)

    # custom_pipeline=create_custom_ttir_pipeline_fn(f"ttcore-wrap-device-module,ttcore-mark-functions-as-forward", print_ir=True),
