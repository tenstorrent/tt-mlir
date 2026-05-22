# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""E2E test for the O-projection + residual-add pattern.

Pattern B of the activation-dtype-lowering pass: a tensor-parallel matmul
whose output flows through reduce_scatter and is added to a sharded residual,
then gathered back to full shape. The pass should rewrite the matmul output
dtype to bfp_bf8 (and propagate that through reduce_scatter) and set the
residual `add.dtype = bf16` so the block output is bf16. PCC is compared
against a bf16 reference.

Structurally this mirrors `test_parallelized_matmul_with_binary_chaining` in
`test_ttir_parallels.py`, with the pass enabled via
`pipeline_options=["enable-activation-dtype-lowering=true"]`.
"""
import torch
import pytest

from collections import OrderedDict
from typing import List, Tuple

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_enums import (
    MeshShardDirection,
    MeshShardType,
    ReduceType,
)
from test_utils import shape_str, make_shard_shape
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


# K-split parallel matmul -> reduce_scatter, returning a sharded result.
# Inputs are sharded along the contraction (K) dimension on `parallelize_axis`;
# output is sharded along N on the same axis.
def _build_o_proj_kshard(
    act: Operand,
    weight: Operand,
    builder: TTIRBuilder,
    mesh_shape: Tuple[int, int],
    parallelize_axis: int,
):
    if parallelize_axis == 0:
        shard_dims_act = [1, 0]
        shard_dims_wt = [0, -1]
        shard_dims_out = [1, 0]
    else:
        shard_dims_act = [0, 1]
        shard_dims_wt = [-1, 0]
        shard_dims_out = [0, 1]

    shard_shape_act = make_shard_shape(2, shard_dims_act, mesh_shape)
    shard_shape_wt = make_shard_shape(2, shard_dims_wt, mesh_shape)

    act_shard = builder.mesh_shard(
        act,
        shard_direction=MeshShardDirection.FullToShard.value,
        shard_type=MeshShardType.Devices.value,
        shard_shape=shard_shape_act,
        shard_dims=shard_dims_act,
    )
    weight_shard = builder.mesh_shard(
        weight,
        shard_direction=MeshShardDirection.FullToShard.value,
        shard_type=MeshShardType.Devices.value,
        shard_shape=shard_shape_wt,
        shard_dims=shard_dims_wt,
    )

    partial = builder.matmul(act_shard, weight_shard)
    reduced = builder.reduce_scatter(
        partial,
        reduce_type=ReduceType.Sum.value,
        scatter_dim=1,
        cluster_axis=parallelize_axis,
    )
    return reduced, shard_dims_out


@pytest.mark.parametrize(
    "shapes",
    [
        [(32, 1024), (1024, 2048), (32, 2048)],
    ],
    ids=lambda s: "_".join(shape_str(x) for x in s),
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
def test_o_proj_residual_dtype_lowering(
    shapes: List[Shape],
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    """matmul (K-split) -> reduce_scatter -> add(residual) -> ShardToFull.

    With `enable-activation-dtype-lowering=true`, the pass should:
      * Set the matmul `dtype = bfp_bf8` and rewrite its result type encoding.
      * Propagate bfp_bf8 through reduce_scatter.
      * Set the residual `add.dtype = bf16` so the block output is bf16.
      * Insert no explicit ttnn.typecast.
    """
    parallelize_axis = 1  # split the K-dim across the y axis of the mesh

    if mesh_shape[parallelize_axis] == 1:
        pytest.skip("parallelism across 1 device is meaningless")

    def module(builder: TTIRBuilder):
        @builder.func(
            shapes, [torch.bfloat16, torch.bfloat16, torch.bfloat16]
        )
        def o_proj_residual(
            act: Operand,
            weight: Operand,
            residual: Operand,
            builder: TTIRBuilder,
        ):
            reduced, shard_dims_out = _build_o_proj_kshard(
                act, weight, builder, mesh_shape, parallelize_axis
            )

            shard_shape_out = make_shard_shape(2, shard_dims_out, mesh_shape)
            residual_shard = builder.mesh_shard(
                residual,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape_out,
                shard_dims=shard_dims_out,
            )

            added = builder.add(residual_shard, reduced)

            output = builder.mesh_shard(
                added,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape_out,
                shard_dims=shard_dims_out,
            )

            input_act = builder._get_golden_tensor(act)
            input_wt = builder._get_golden_tensor(weight)
            input_res = builder._get_golden_tensor(residual)
            golden = torch.add(torch.matmul(input_act, input_wt), input_res)
            builder.set_goldens_from_builder_tensor(
                {act: input_act, weight: input_wt, residual: input_res},
                {output: golden},
            )
            return output

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
        pipeline_options=["enable-activation-dtype-lowering=true"],
        pcc=0.98,
        **get_request_kwargs(request),
    )
