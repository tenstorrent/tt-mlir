# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import compile_and_execute_ttnn
from test_utils import shape_str

from golden.mapping import GoldenMapTensor

pytestmark = pytest.mark.frontend("ttnn")

TILE_WIDTH = 32


def build_stats_golden(input_golden: GoldenMapTensor) -> GoldenMapTensor:
    """Build a valid stats tensor from the input, matching tt-metal's expected format.

    For single-device layernorm the stats tensor is 2 tiles wide (64 elements):
      tile 0 [0:32]:  sum(x^2) at position 0, zeros elsewhere
      tile 1 [32:64]: sum(x) at position 32, zeros elsewhere
    """

    def compute_stats(shard):
        shard_float = shard.float()
        sum_x = shard_float.sum(dim=-1, keepdim=True)
        sum_x2 = shard_float.pow(2).sum(dim=-1, keepdim=True)
        output_shape = list(shard_float.shape)
        output_shape[-1] = 2 * TILE_WIDTH
        stats = torch.zeros(output_shape, dtype=shard.dtype)
        stats[..., :1] = sum_x2
        stats[..., TILE_WIDTH : TILE_WIDTH + 1] = sum_x
        return stats

    return GoldenMapTensor.apply_shardwise(input_golden, compute_stats)


# LayerNormPreAllGather tests


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),
        (1, 1, 32, 512),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_layer_norm_pre_all_gather(
    shape: Shape,
    has_residual: bool,
    target: str,
    request,
    device,
):
    shapes = [shape]
    if has_residual:
        shapes.append(shape)

    def module(builder: TTNNBuilder):
        @builder.func(shapes, [torch.bfloat16] * len(shapes))
        def layer_norm_pre_all_gather(*inputs, unit_attrs: Optional[List[str]] = None):
            builder = inputs[-1]
            in0 = inputs[0]
            residual = None
            if has_residual and len(inputs) > 2:
                residual = inputs[1]

            return builder.layer_norm_pre_all_gather(
                in0,
                residual_input=residual,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# LayerNormPostAllGather tests


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),
        (1, 1, 32, 512),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("has_weight_bias", [False, True])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_layer_norm_post_all_gather(
    shape: Shape,
    has_weight_bias: bool,
    target: str,
    request,
    device,
):
    # Stats tensor shape: same leading dims as input, last dim = 2 * TILE_WIDTH (64)
    stats_shape = shape[:-1] + (64,)
    shapes = [shape, stats_shape]
    dtypes = [torch.bfloat16, torch.bfloat16]

    if has_weight_bias:
        weight_shape = (shape[-1],)
        shapes.append(weight_shape)
        shapes.append(weight_shape)
        dtypes.extend([torch.bfloat16, torch.bfloat16])

    def module(builder: TTNNBuilder):
        @builder.func(shapes, dtypes)
        def layer_norm_post_all_gather(*inputs, unit_attrs: Optional[List[str]] = None):
            builder = inputs[-1]
            in0 = inputs[0]
            stats = inputs[1]
            weight = None
            bias = None
            if has_weight_bias and len(inputs) > 3:
                weight = inputs[2]
                bias = inputs[3]

            # Override the random stats golden with valid statistics
            # derived from the input tensor, matching tt-metal's format.
            input_golden = builder._get_golden_tensor(in0)
            stats_golden = build_stats_golden(input_golden)
            builder._set_golden_tensor(stats, stats_golden)

            return builder.layer_norm_post_all_gather(
                in0,
                stats,
                weight=weight,
                bias=bias,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


def build_rms_stats_golden(input_golden: GoldenMapTensor) -> GoldenMapTensor:
    """Build a valid stats tensor from the input, matching tt-metal's expected format.

    For single-device rmsnorm the stats tensor is 1 tile wide (32 elements):
      tile 0 [0:32]:  sum(x^2) at position 0, zeros elsewhere
    """

    def compute_stats(shard):
        shard_float = shard.float()
        sum_x2 = shard_float.pow(2).sum(dim=-1, keepdim=True)
        output_shape = list(shard_float.shape)
        output_shape[-1] = TILE_WIDTH
        stats = torch.zeros(output_shape, dtype=shard.dtype)
        stats[..., :1] = sum_x2
        return stats

    return GoldenMapTensor.apply_shardwise(input_golden, compute_stats)


# RMSNormPostAllGather tests


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),
        (1, 1, 32, 512),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("has_weight_bias", [False, True])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_rms_norm_post_all_gather(
    shape: Shape,
    has_weight: bool,
    target: str,
    request,
    device,
):
    # Stats tensor shape: same leading dims as input, last dim = TILE_WIDTH (32)
    stats_shape = shape[:-1] + (32,)
    shapes = [shape, stats_shape]
    dtypes = [torch.bfloat16, torch.bfloat16]

    if has_weight:  # Test without bias as torch.rms_norm does not support bias
        weight_shape = (shape[-1],)
        shapes.append(weight_shape)
        dtypes.extend([torch.bfloat16, torch.bfloat16])

    def module(builder: TTNNBuilder):
        @builder.func(shapes, dtypes)
        def rms_norm_post_all_gather(*inputs, unit_attrs: Optional[List[str]] = None):
            builder = inputs[-1]
            in0 = inputs[0]
            stats = inputs[1]
            weight = None
            bias = None
            if has_weight and len(inputs) > 2:
                weight = inputs[2]

            # Override the random stats golden with valid statistics
            # derived from the input tensor, matching tt-metal's format.
            input_golden = builder._get_golden_tensor(in0)
            stats_golden = build_rms_stats_golden(input_golden)
            builder._set_golden_tensor(stats, stats_golden)

            return builder.rms_norm_post_all_gather(
                in0,
                stats,
                weight=weight,
                bias=bias,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttnn(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
