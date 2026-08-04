# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR coverage for d2m-jit fabric and CCL kernel operations."""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise
from ttmlir.passmanager import PassManager


@d2m.kernel
def ccl_kernel(input_, output, start_sem, end_sem):
    dy = mesh_position(0)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        start_sem,
        start_device=[dy, 0],
        mcast_shape=[1, 2],
        num_receivers=1,
        core_indices=[cy, cx],
    )
    scratch = empty([1, 1])
    scratch = remote_load(scratch, input_, [cy, cx])
    remote_store(
        output,
        [cy, cx],
        scratch,
        start_device=[dy, 0],
        device_mcast_shape=[1, 2],
        semaphore=end_sem,
        semaphore_indices=[cy, cx],
    )
    semaphore_wait(end_sem, 2)


d2m.mesh((1, 2), topology=("linear", "ring"))
layout = d2m.Layout(
    shape=(32, 32),
    dtype=d2m.float32,
    block_shape=[1, 1],
    grid_shape=[1, 1],
)
input_ = d2m.empty(layout)
output = d2m.empty(layout)
start_sem = d2m.global_semaphore(grid_shape=(8, 8))
end_sem = d2m.global_semaphore(grid_shape=(8, 8))
ccl_kernel(
    input_,
    output,
    start_sem,
    end_sem,
    grid=(1, 1),
    fabric=d2m.fabric_config(cluster_axis=1),
)

builder = _Builder.get()
_emit_returns_and_finalise(builder, [output])
PassManager.parse(
    "builtin.module(ttcore-register-device{mock-system-desc-arch=wormhole_b0 "
    "mesh-shape=1,2 mesh-topology=linear,ring})",
    context=builder.ctx,
).run(builder.module.operation)
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK: d2m.generic
# CHECK-SAME: fabricConnectionConfig = #ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1>
# CHECK: %[[DY:.*]] = "d2m.mesh_position"
# CHECK: d2m.device_synchronize
# CHECK-SAME: %[[DY]]
# CHECK: tensor.empty()
# CHECK: d2m.remote_load
# CHECK: d2m.remote_store
# CHECK-SAME: devices startDevice
# CHECK-SAME: deviceMcastShape
# CHECK-SAME: semaphore increment
# CHECK: d2m.semaphore_wait
