# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape lit test for a full all_gather kernel.

A `@d2m.kernel` (the only form -- unified) matching tools/d2m-jit/ccl.d2m that
uses the full CCL op set: `device_synchronize`, cross-device `remote_store`
with a semaphore, `semaphore_wait`, and `semaphore_set`. The explicit
`semaphore_set` / `semaphore_inc` ops are currently *permitted* in unified form
(checkForIllegalSemaphoreOps is temporarily permissive); they are not yet split
safely across threads -- see the TODO in DMAUtils.cpp.

Registers a mock Wormhole device so the verifiers run, then checks the IR.
"""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder
from ttmlir.dialects import func
from ttmlir.passmanager import PassManager


L = d2m.Layout(shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2])


@d2m.kernel
def all_gather(in0, out0, ccl_start_semaphore, ccl_end_semaphore):
    dy = mesh_position(0)
    dx = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    device_synchronize(
        ccl_start_semaphore,
        start_device=[dy, 0],
        mcast_shape=[1, 8],
        num_receivers=7,
        core_indices=[cy, cx],
    )
    buf = empty([1, 1])
    remote_load(buf, in0, [cy, 0])
    remote_store(
        out0,
        [dx * 2 + cy, 0],
        buf,
        start_device=[dy, 0],
        device_mcast_shape=[1, 8],
        semaphore=ccl_end_semaphore,
        semaphore_indices=[cy, 0],
    )
    semaphore_wait(ccl_end_semaphore, 7)
    semaphore_set(ccl_start_semaphore, 0)


in0 = d2m.empty(L)
out0 = d2m.empty(L)
start_sem = d2m.global_semaphore(grid_shape=(8, 8), init=0)
end_sem = d2m.global_semaphore(grid_shape=(8, 8), init=0)
all_gather(in0, out0, start_sem, end_sem, grid=(2, 2))

b = _Builder.get()
with b.ctx, b.loc, b.insert_point:
    func.ReturnOp([])
PassManager.parse(
    "builtin.module(ttcore-register-device{mock-system-desc-arch=wormhole_b0})",
    context=b.ctx,
).run(b.module.operation)
b.module.operation.verify()
print(b.module)
_Builder.reset()


# Datamovement thread (not unified) -- the form that makes set/sync legal.
# CHECK: d2m.generic
# CHECK-SAME: threads = [#d2m.thread<unified>]
# Two global semaphores threaded in as additionalArgs.
# CHECK: additionalArgs(%{{.*}}, %{{.*}} : !d2m.global_semaphore, !d2m.global_semaphore)
# CHECK: d2m.mesh_position
# CHECK: d2m.device_synchronize{{.*}}numReceivers = 7 : i32
# CHECK: d2m.remote_store {{.*}} devices startDevice[{{.*}}] deviceMcastShape[{{.*}}] semaphore increment %{{.*}}[
# CHECK: d2m.semaphore_wait %{{.*}}, %{{.*}} : !d2m.global_semaphore
# CHECK: d2m.semaphore_set %{{.*}}, %{{.*}} : !d2m.global_semaphore
