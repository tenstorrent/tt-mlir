# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2 1 2>&1 | FileCheck %s
# RUN: %python %s 1 2 2>&1 | FileCheck %s
# RUN: %python %s 2 2 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""Lower a chunked matmul + all-gather through the complete backend."""

import sys

import torch

import d2m_jit as d2m
from d2m_jit._src.builder import (
    _Builder,
    _emit_returns_and_finalise,
    _pipeline_passes,
)
from ttmlir.passmanager import PassManager


@d2m.kernel
def chunked_matmul_all_gather(lhs, rhs, output, start_sem, end_sem, num_chunks):
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
    for chunk in range(num_chunks):
        a = remote_load(lhs, [cy * num_chunks + chunk, cx])
        b = remote_load(rhs, [cy * num_chunks + chunk, cx])
        c = a @ b
        remote_store(
            output,
            [cy * 2 * num_chunks + dx * num_chunks + chunk, cx],
            c,
            start_device=[dy, 0],
            device_mcast_shape=[1, 2],
            semaphore=end_sem,
            semaphore_indices=[cy, cx],
        )
    semaphore_wait(end_sem, 2 * num_chunks)


num_chunks = int(sys.argv[1])
assert num_chunks in (1, 2)
worker_grid_size = int(sys.argv[2])
assert worker_grid_size in (1, 2)
grid_y = worker_grid_size
grid_x = worker_grid_size
d2m.mesh((1, 2), topology=("linear", "linear"))
block_tiles = 4
block_elements = block_tiles * 32
input_layout = d2m.Layout(
    shape=(grid_y * num_chunks * block_elements, grid_x * block_elements),
    dtype=d2m.float32,
    block_shape=[block_tiles, block_tiles],
    grid_shape=[grid_y, grid_x],
)
output_layout = d2m.Layout(
    shape=(grid_y * 2 * num_chunks * block_elements, grid_x * block_elements),
    dtype=d2m.float32,
    block_shape=[block_tiles, block_tiles],
    grid_shape=[grid_y, grid_x],
)
full_lhs = torch.randn(
    grid_y * num_chunks * block_elements,
    2 * grid_x * block_elements,
)
full_rhs = torch.randn(
    grid_y * num_chunks * block_elements,
    2 * grid_x * block_elements,
)
lhs = d2m.mesh_shard(
    full_lhs,
    input_layout,
    shard_dims=[0, 1],
    shard_shape=[1, 2],
)
rhs = d2m.mesh_shard(
    full_rhs,
    input_layout,
    shard_dims=[0, 1],
    shard_shape=[1, 2],
)
output = d2m.empty(output_layout)
chunked_matmul_all_gather(
    lhs,
    rhs,
    output,
    d2m.global_semaphore(grid_shape=(8, 8)),
    d2m.global_semaphore(grid_shape=(8, 8)),
    num_chunks,
    grid=(grid_y, grid_x),
    fabric=d2m.fabric_config(
        cluster_axis=1,
        topology="linear",
        num_links=grid_y * grid_x,
    ),
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
num_setups = lowered.count("experimental::setup_fabric_connections")
num_closes = lowered.count("experimental::close_fabric_connections")
num_writes = lowered.count("experimental::fabric_mcast_fast_write_any_len")
num_fabric_sem_incs = lowered.count("experimental::fabric_mcast_sem_inc")
assert num_setups == 1, num_setups
assert num_closes == 1, num_closes
assert num_writes == 1, num_writes
assert num_fabric_sem_incs == 2, num_fabric_sem_incs

fabric_enqueue = next(
    line
    for line in lowered.splitlines()
    if '"ttmetal.enqueue_program"' in line and "fabricConnectionConfig" in line
)
assert fabric_enqueue.count("#ttmetal.noc_config<") == 2, fabric_enqueue
assert fabric_enqueue.count("#ttmetal.compute_config<") == 1, fabric_enqueue
expected_core_range = f"#ttmetal.core_range<0x0, {grid_y}x{grid_x}>"
assert fabric_enqueue.count(expected_core_range) == 3, fabric_enqueue
assert "dm_core = 1, noc0" in fabric_enqueue, fabric_enqueue
assert "dm_core = 0, noc1" in fabric_enqueue, fabric_enqueue

# The chunk loop indexes distinct generic outputs. It must not be mistaken for
# a matmul reduction loop, which would accumulate chunk t onto chunk t-1.
assert "llk_pack_reconfig_l1_acc" not in lowered, lowered
print("chunked matmul all-gather lowering: ok")
_Builder.reset()


# CHECK: chunked matmul all-gather lowering: ok
