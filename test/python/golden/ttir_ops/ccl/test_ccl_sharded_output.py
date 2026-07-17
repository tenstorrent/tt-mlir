# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# PCC coverage for CCL ops (AllGather / ReduceScatter / AllReduce) whose OUTPUT
# is placed in an L1 *sharded* memory layout (width / height / block sharded),
# in addition to the usual interleaved output.
#
# Motivation: PR #8413 lets the optimizer query op-model constraints for these
# CCL ops, which means it can now choose a *sharded* output layout for them. The
# op-model constraint query only checks resource feasibility (see the
# CCLShardedOutput gtest in TestOpModelLibMockDevice) — it does NOT verify
# numerical correctness. e2e models compiled with these constraints showed
# decode PCC ~0.0, which points at a sharded CCL output being computed
# incorrectly on device. These tests pin correctness down per (op, output
# layout) by forcing the CCL op's output layout via `override-output-layout`
# and checking PCC against the builder golden.
#
# Requires a multi-device system (n300 / T3K / Galaxy); skipped on single chip.

import pytest
import torch
from collections import OrderedDict
from typing import Tuple

from builder.base.builder_utils import Operand, Shape
from builder.base.builder_enums import (
    MeshShardDirection,
    MeshShardType,
    ReduceType,
)
from builder.base.builder_apis import compile_and_execute_ttir
from builder.ttir.ttir_builder import TTIRBuilder
from conftest import get_request_kwargs
from test_utils import shape_str, make_shard_shape

pytestmark = pytest.mark.frontend("ttir")

# L1 sharded (and interleaved baseline) output memory configs for the CCL op,
# expressed as override-output-layout values. The CCL op is named "ccl_op" in
# each module below so the override targets exactly its output.
OUTPUT_LAYOUTS = [
    "l1:interleaved",
    "l1:width_sharded",
    "l1:height_sharded",
    "l1:block_sharded",
]


def _pipeline_options(out_layout: str):
    # optimization-level>=1 activates the layout override machinery; the
    # override pins the named CCL op's output to the requested L1 layout.
    return [
        "optimization-level=2",
        f"override-output-layout=ccl_op={out_layout}",
    ]


@pytest.mark.parametrize("out_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (1, 8)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_all_gather_sharded_output(
    out_layout: str,
    mesh_shape: Tuple[int, int],
    dtype: torch.dtype,
    request,
    device,
):
    cluster_axis = 1
    all_gather_dim = 3
    # Per-device shard is [1, 1, 256, 256] (8x8 tiles) — comfortably shardable.
    test_shape = (1, 1, 256, 256)
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)
    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_gather(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            gathered = builder.all_gather(
                in_shard,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
                loc="ccl_op",
            )
            return builder.mesh_shard(
                gathered,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=_pipeline_options(out_layout),
        check_pcc=True,
        pcc=0.99,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("out_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (1, 8)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_reduce_scatter_sharded_output(
    out_layout: str,
    mesh_shape: Tuple[int, int],
    dtype: torch.dtype,
    request,
    device,
):
    cluster_axis = 1
    scatter_dim = 3
    # After scatter on the last dim the per-device output is [1,1,256,256].
    test_shape = (1, 1, 256, 256 * mesh_shape[cluster_axis])
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)
    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def reduce_scatter(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            scattered = builder.reduce_scatter(
                in_shard,
                reduce_type=ReduceType.Sum.value,
                scatter_dim=scatter_dim,
                cluster_axis=cluster_axis,
                loc="ccl_op",
            )
            return builder.mesh_shard(
                scattered,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=_pipeline_options(out_layout),
        check_pcc=True,
        pcc=0.99,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("out_layout", OUTPUT_LAYOUTS)
@pytest.mark.parametrize("mesh_shape", [(1, 2), (1, 8)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
def test_all_reduce_sharded_output(
    out_layout: str,
    mesh_shape: Tuple[int, int],
    dtype: torch.dtype,
    request,
    device,
):
    cluster_axis = 1
    # all_reduce preserves shape; per-device tensor is [1,1,256,256] (8x8 tiles).
    test_shape = (1, 1, 256, 256)
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)
    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_reduce(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            reduced = builder.all_reduce(
                in_shard,
                cluster_axis=cluster_axis,
                reduce_type=ReduceType.Sum.value,
                loc="ccl_op",
            )
            return builder.mesh_shard(
                reduced,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=_pipeline_options(out_layout),
        check_pcc=True,
        pcc=0.99,
        **get_request_kwargs(request),
    )
