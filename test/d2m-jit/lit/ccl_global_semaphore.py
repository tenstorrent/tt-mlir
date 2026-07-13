# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape lit test for global-semaphore kernel arguments (no device).

Builds a CCL-shaped kernel that takes a `d2m.global_semaphore` argument and
checks the full host + generic IR after registering a *mock* Wormhole device
(so the create_global_semaphore / generic verifiers actually run -- the
backing-buffer grid must match the 8x8 worker grid).

This is the single-device "fake CCL" anchor modelled on
test/ttmlir/Dialect/D2M/generic/generic_global_semaphores.mlir. It uses only
`semaphore_wait` (no reset) inside the unified region. (`semaphore_set` /
`semaphore_inc` / `semaphore_wait`-with-reset are now *accepted* in unified
form but are not yet split safely across threads -- see the TODO in
DMAUtils.cpp's checkForIllegalSemaphoreOps.)
"""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder
from ttmlir.dialects import func
from ttmlir.passmanager import PassManager


L = d2m.Layout(shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2])


@d2m.kernel
def fake_ccl(in0, out0, end_sem):
    cy = core_index(0)
    buf = empty([1, 1])
    remote_load(buf, in0, [cy, 0])
    remote_store(out0, [cy, 0], buf, semaphore=end_sem, semaphore_indices=[cy, 0])
    semaphore_wait(end_sem, 1)


in0 = d2m.empty(L)
out0 = d2m.empty(L)
end_sem = d2m.global_semaphore(grid_shape=(8, 8), init=0)
fake_ccl(in0, out0, end_sem, grid=(2, 2))

# Finalise the open host func with a terminator and register a mock device so
# the verifiers (which need a registered device) run, then dump the IR.
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


# CHECK: ttcore.device
# Backing buffer over the 8x8 worker grid, 1x1 shard, ui32 element type.
# CHECK: %[[BACK:.*]] = d2m.empty() : tensor<8x8x1x1xui32
# CHECK: %[[SEM:.*]] = d2m.create_global_semaphore(%[[BACK]]) {value = 0 : ui32} : tensor<8x8x1x1xui32{{.*}} -> !d2m.global_semaphore
# Semaphore is an additionalArg of the generic.
# CHECK: d2m.generic
# CHECK: additionalArgs(%[[SEM]] : !d2m.global_semaphore)
# Cross-device store posts to the semaphore; wait inside the unified region.
# CHECK: d2m.remote_store {{.*}} semaphore increment %[[SEM]][
# CHECK: d2m.semaphore_wait %[[SEM]], %{{.*}} : !d2m.global_semaphore
# Reset emitted at host scope after the generic.
# CHECK: d2m.reset_global_semaphore(%[[SEM]]) {value = 0 : ui32}
