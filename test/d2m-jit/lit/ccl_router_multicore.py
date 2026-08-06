# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2 1 2>&1 | FileCheck %s
# RUN: %python %s 2 2 2>&1 | FileCheck %s
# RUN: %python %s 1 1 2>&1 | FileCheck %s
# RUN: %python %s 1 1 2 2>&1 | FileCheck %s
# RUN: %python %s 4 4 2 2>&1 | FileCheck %s
# RUN: %python %s 4 4 2 bf16 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""Lower multicore matmul + router-core all-gather end to end."""

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
def router_matmul_all_gather(
    lhs,
    rhs,
    output,
    start_sem,
    ready,
    consumed,
    end_sem,
    num_chunks,
    grid_y,
    grid_x,
    num_workers,
    end_count,
):
    cy = core_index(0)
    cx = core_index(1)

    for chunk in range(num_chunks):
        a = remote_load(lhs, [cy * num_chunks + chunk, cx])
        b = remote_load(rhs, [cy * num_chunks + chunk, cx])
        c = a @ b
        semaphore_inc(ready, 1, core=[0, 0], compute=True)

        if is_router_core():
            dy = mesh_position(0)
            device_synchronize(
                start_sem,
                start_device=[dy, 0],
                mcast_shape=[1, 2],
                num_receivers=1,
                core_indices=[cy, cx],
            )
            semaphore_wait(ready, (chunk + 1) * num_workers)
            dx = mesh_position(1)
            for ty in range(grid_y):
                for tx in range(grid_x):
                    gathered = empty_like(c)
                    gathered = core_read(gathered, c, core=[ty, tx])
                    semaphore_inc(consumed, 1, core=[ty, tx])
                    remote_store(
                        output,
                        [
                            dx * grid_y * num_chunks + ty * num_chunks + chunk,
                            tx,
                        ],
                        gathered,
                        start_device=[dy, 0],
                        device_mcast_shape=[1, 2],
                        semaphore=end_sem,
                        semaphore_indices=[cy, 0],
                    )
            semaphore_wait(end_sem, (chunk + 1) * end_count)
        semaphore_wait(consumed, chunk + 1, compute=True)


grid_y = int(sys.argv[1])
grid_x = int(sys.argv[2])
num_chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 1
dtype_name = sys.argv[4] if len(sys.argv) > 4 else "fp32"
assert (grid_y, grid_x) in ((1, 1), (2, 1), (2, 2), (4, 4))
assert num_chunks in (1, 2)
assert dtype_name in ("fp32", "bf16")
layout_dtype = d2m.float32 if dtype_name == "fp32" else d2m.bfloat16
torch_dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
num_workers = grid_y * grid_x
end_count = 2 * num_workers
block_tiles = 4
block_elements = block_tiles * 32
d2m.mesh((1, 2), topology=("linear", "linear"))
input_layout = d2m.Layout(
    shape=(grid_y * num_chunks * block_elements, grid_x * block_elements),
    dtype=layout_dtype,
    block_shape=[block_tiles, block_tiles],
    grid_shape=[grid_y * num_chunks, grid_x],
)
output_layout = d2m.Layout(
    shape=(2 * grid_y * num_chunks * block_elements, grid_x * block_elements),
    dtype=layout_dtype,
    block_shape=[block_tiles, block_tiles],
    grid_shape=[2 * grid_y * num_chunks, grid_x],
)
full_lhs = torch.randn(
    grid_y * num_chunks * block_elements,
    2 * grid_x * block_elements,
    dtype=torch_dtype,
)
full_rhs = torch.randn(
    grid_y * num_chunks * block_elements,
    2 * grid_x * block_elements,
    dtype=torch_dtype,
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
router_matmul_all_gather(
    lhs,
    rhs,
    output,
    d2m.global_semaphore(grid_shape=(8, 8)),
    d2m.global_semaphore(grid_shape=(8, 8), init=0),
    d2m.global_semaphore(grid_shape=(8, 8), init=0),
    d2m.global_semaphore(grid_shape=(8, 8)),
    num_chunks,
    grid_y,
    grid_x,
    num_workers,
    end_count,
    grid=(grid_y, grid_x),
    fabric=d2m.fabric_config(
        cluster_axis=1,
        topology="linear",
        router_cores=[(0, 0)],
    ),
)

builder = _Builder.get()
_emit_returns_and_finalise(builder, [output])
pipeline = (
    "builtin.module("
    "ttcore-register-device{mock-system-desc-arch=wormhole_b0 "
    "mesh-shape=1,2 mesh-topology=linear,linear}," + ",".join(_pipeline_passes()) + ")"
)
pm = PassManager.parse(pipeline, context=builder.ctx)
pm.run(builder.module.operation)
builder.module.operation.verify()
lowered = str(builder.module)
num_setups = lowered.count("experimental::setup_fabric_connections")
num_closes = lowered.count("experimental::close_fabric_connections")
assert num_setups == 1, num_setups
assert num_closes == 1, num_closes
assert "async_read(unicast_ep" in lowered
assert "experimental::fabric_mcast_fast_write_any_len" in lowered
assert "experimental::fabric_mcast_sem_inc" in lowered
compute_start = lowered.index("func.func private @compute_kernel5")
compute_end = lowered.index("func.func private @", compute_start + 1)
assert "experimental::semaphore_wait" not in lowered[compute_start:compute_end]
print("multicore router matmul all-gather lowering: ok")
_Builder.reset()


# CHECK: multicore router matmul all-gather lowering: ok
