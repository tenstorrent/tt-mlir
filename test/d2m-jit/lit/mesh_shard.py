# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR coverage for d2m-jit mesh shard and gather boundaries."""

import torch

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise

d2m.mesh((1, 2), topology=("linear", "ring"))
layout = d2m.Layout(
    shape=(64, 64),
    dtype=d2m.float32,
    block_shape=[1, 1],
    grid_shape=[2, 2],
)
full = torch.zeros((64, 128), dtype=torch.float32)
shard = d2m.mesh_shard(
    full,
    layout,
    shard_dims=[0, 1],
    shard_shape=[1, 2],
)
gathered = d2m.mesh_gather(shard)

builder = _Builder.get()
assert builder._mesh_shape == [1, 2]
assert builder._mesh_topology == ["linear", "ring"]
assert shard.mesh.full_shape == [64, 128]
_emit_returns_and_finalise(builder, [gathered])
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK: module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>}
# CHECK: func.func @main(%{{.*}}: tensor<64x128xf32>) -> tensor<64x128xf32>
# CHECK: shard_direction = #ttcore.shard_direction<full_to_shard>
# CHECK-SAME: tensor<64x128xf32> -> tensor<64x64xf32, #ttcore.tensor_mesh<"mesh">>
# CHECK: tensor<64x64xf32, #ttcore.tensor_mesh<"mesh">>
# CHECK: shard_direction = #ttcore.shard_direction<shard_to_full>
# CHECK-SAME: tensor<64x64xf32, #ttcore.tensor_mesh<"mesh">> -> tensor<64x128xf32>
