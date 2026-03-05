# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional, Tuple
from collections import OrderedDict
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_enums import MeshShardDirection, MeshShardType
from test_utils import (
    shapes_list_str,
    shape_str,
    make_shard_shape,
)

pytestmark = pytest.mark.frontend("ttir")


# Batch norm tests


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 64, 32, 32),  # input tensor: (N, C, H, W)
            (64,),  # scale (gamma)
            (64,),  # offset (beta)
            (64,),  # mean
            (64,),  # variance
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 5])
@pytest.mark.parametrize("dimension", [1])  # channel dimension
@pytest.mark.parametrize("epsilon", [1e-5])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_batch_norm(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    dimension: int,
    epsilon: float,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def batch_norm(
            in0: Operand,
            scale: Operand,
            offset: Operand,
            mean: Operand,
            variance: Operand,
            builder,
            unit_attrs: Optional[List[str]] = None,
        ):

            return builder.batch_norm_inference(
                in0,
                scale,
                offset,
                mean,
                variance,
                epsilon=epsilon,
                dimension=dimension,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# RMS norm tests


@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((32, 128), [128]),
        ((2, 4, 64), [64]),
        ((1, 136, 2048), [2048]),
    ],
)
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_rms_norm(
    shape: Shape,
    normalized_shape: List[int],
    has_weight: bool,
    has_bias: bool,
    target: str,
    request,
    device,
):
    # Determine input shapes
    shapes = [shape]
    if has_weight:
        shapes.append(tuple(normalized_shape))
    if has_bias:
        shapes.append(tuple(normalized_shape))

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def rms_norm(*inputs, unit_attrs: Optional[List[str]] = None):

            builder = inputs[-1]
            # Extract inputs based on test configuration
            in0 = inputs[0]
            weight = None
            bias = None

            if has_weight and len(inputs) > 1:
                weight = inputs[1]
            if has_bias:
                if has_weight and len(inputs) > 2:
                    bias = inputs[2]
                elif not has_weight and len(inputs) > 1:
                    bias = inputs[1]

            return builder.rms_norm(
                in0,
                normalized_shape=normalized_shape,
                weight=weight,
                bias=bias,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Softmax tests


@pytest.mark.parametrize("shape", [(32, 512, 1024)], ids=shape_str)
@pytest.mark.parametrize("dimension", [0, 1, 2])
@pytest.mark.parametrize("numeric_stable", [False, True])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_softmax(
    shape: Shape, dimension: int, numeric_stable: bool, target: str, request, device
):
    # Create a wrapper function that captures dimension
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def softmax(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.softmax(
                in0,
                dimension=dimension,
                numeric_stable=numeric_stable,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_softmax(
    shape: Shape,
    request,
    target: str,
    device,
    dtype: torch.dtype = torch.float32,
):
    pytest.skip(
        reason="Softmax does not lower to loops properly https://github.com/tenstorrent/tt-mlir/issues/3232"
    )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def softmax(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.softmax(
                in0,
                dimension=-1,
                numeric_stable=False,
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Layer norm tests


@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((32, 128), [128]),
        ((2, 4, 64), [64]),
    ],
)
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_layer_norm(
    shape: Shape,
    normalized_shape: List[int],
    has_weight: bool,
    has_bias: bool,
    target: str,
    request,
    device,
):
    # Determine input shapes
    shapes = [shape]
    if has_weight:
        shapes.append(tuple(normalized_shape))
    if has_bias:
        shapes.append(tuple(normalized_shape))

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def layer_norm(*inputs, unit_attrs: Optional[List[str]] = None):

            builder = inputs[-1]
            # Extract inputs based on test configuration
            in0 = inputs[0]
            weight = None
            bias = None

            if has_weight and len(inputs) > 1:
                weight = inputs[1]
            if has_bias:
                if has_weight and len(inputs) > 2:
                    bias = inputs[2]
                elif not has_weight and len(inputs) > 1:
                    bias = inputs[1]

            return builder.layer_norm(
                in0,
                normalized_shape=normalized_shape,
                weight=weight,
                bias=bias,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((32, 128), [128]),
        ((2, 4, 64), [64]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_layer_norm(
    shape: Shape,
    normalized_shape: List[int],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def layer_norm(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.layer_norm(
                in0,
                normalized_shape=normalized_shape,
                weight=None,
                bias=None,
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Distributed RMS norm tests


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),
        (1, 1, 32, 512),
        (1, 1, 32, 4096),
        (1, 1, 32, 8192),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("has_weight", [True])
@pytest.mark.parametrize("has_residual", [True, False])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_distributed_rms_norm(
    shape: Shape,
    has_weight: bool,
    has_residual: bool,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    target: str,
    request,
    device,
):
    """
    Test distributed RMS normalization with all-gather across mesh devices.

    This test verifies the fused operation that combines:
    1. Optional residual addition
    2. RMS normalization
    3. All-gather collective communication
    """
    # Determine input shapes
    shapes = [shape]
    weight_shape = (shape[-1],)  # Weight matches last dimension
    if has_weight:
        shapes.append(weight_shape)
    if has_residual:
        shapes.append(shape)

    # Shard dimensions for width sharding (dim 3)
    shard_dims = [-1, 3]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.bfloat16] * len(shapes))
        def distributed_rms_norm_test(*inputs, unit_attrs: Optional[List[str]] = None):
            builder = inputs[-1]

            # Extract inputs
            in0 = inputs[0]
            weight = None
            residual = None
            input_idx = 1

            if has_weight:
                weight = inputs[input_idx]
                input_idx += 1
            if has_residual:
                residual = inputs[input_idx]

            # Shard input across devices
            shard_shape_input = make_shard_shape(len(shape), shard_dims, mesh_shape)
            sharded_input = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape_input,
                shard_dims=shard_dims,
            )

            # Shard weight by last dim if present
            sharded_weight = None
            if weight is not None:
                weight_shard_dims = [-1, 0]
                weight_shard_shape = make_shard_shape(
                    len(weight_shape), weight_shard_dims, mesh_shape
                )
                sharded_weight = builder.mesh_shard(
                    weight,
                    shard_direction=MeshShardDirection.FullToShard.value,
                    shard_type=MeshShardType.Devices.value,
                    shard_shape=weight_shard_shape,
                    shard_dims=weight_shard_dims,
                )

            # Shard residual if present
            sharded_residual = None
            if residual is not None:
                sharded_residual = builder.mesh_shard(
                    residual,
                    shard_direction=MeshShardDirection.FullToShard.value,
                    shard_type=MeshShardType.Devices.value,
                    shard_shape=shard_shape_input,
                    shard_dims=shard_dims,
                )

            result = builder.distributed_rms_norm(
                sharded_input,
                cluster_axis=cluster_axis,
                weight=sharded_weight,
                residual=sharded_residual,
                epsilon=1e-5,
            )

            gathered = builder.mesh_shard(
                result,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape_input,
                shard_dims=shard_dims,
            )

            return gathered

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
