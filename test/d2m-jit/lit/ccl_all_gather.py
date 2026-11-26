# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""Compile a minimal 1x2 fabric all-gather through the full backend."""

import torch

import d2m_jit as d2m
from d2m_jit._src.builder import (
    _Builder,
    _emit_returns_and_finalise,
    _pipeline_passes,
)
from ttmlir.passmanager import PassManager


@d2m.kernel
def all_gather(input_, output, start_sem, end_sem):
    dy = mesh_position(0)
    dx = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        start_sem,
        start_device=[dy, 0],
        mcast_shape=[1, 2],
        num_receivers=1,
        core_indices=[cy, cx],
    )
    scratch = empty([2, 2])
    scratch = remote_load(scratch, input_, [0, 0])
    remote_store(
        output,
        [dx, 0],
        scratch,
        start_device=[dy, 0],
        device_mcast_shape=[1, 2],
        semaphore=end_sem,
        semaphore_indices=[cy, cx],
    )
    semaphore_wait(end_sem, 1)


d2m.mesh((1, 2), topology=("linear", "linear"))
input_layout = d2m.Layout(
    shape=(64, 64),
    dtype=d2m.float32,
    block_shape=[2, 2],
    grid_shape=[1, 1],
)
output_layout = d2m.Layout(
    shape=(128, 64),
    dtype=d2m.float32,
    block_shape=[2, 2],
    grid_shape=[2, 1],
)
full = torch.randn(64, 128, dtype=torch.float32)
input_ = d2m.mesh_shard(
    full,
    input_layout,
    shard_dims=[0, 1],
    shard_shape=[1, 2],
)
output = d2m.empty(output_layout)
start_sem = d2m.global_semaphore(grid_shape=(8, 8))
end_sem = d2m.global_semaphore(grid_shape=(8, 8))
all_gather(
    input_,
    output,
    start_sem,
    end_sem,
    grid=(1, 1),
    fabric=d2m.fabric_config(cluster_axis=1, topology="linear"),
)
gathered = d2m.mesh_gather(
    output,
    shard_dims=[0, 1],
    shard_shape=[1, 2],
)

builder = _Builder.get()
_emit_returns_and_finalise(builder, [gathered])
pipeline = (
    "builtin.module("
    "ttcore-register-device{mock-system-desc-arch=wormhole_b0 "
    "mesh-shape=1,2 mesh-topology=linear,linear}," + ",".join(_pipeline_passes()) + ")"
)
PassManager.parse(pipeline, context=builder.ctx).run(builder.module.operation)
builder.module.operation.verify()
lowered = str(builder.module)
assert "setup_fabric_connections" in lowered
print("fabric lowering: ok")
_Builder.reset()


# CHECK: fabric lowering: ok
