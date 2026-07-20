# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR coverage for d2m-jit global-semaphore kernel arguments."""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise
from ttmlir.passmanager import PassManager


@d2m.kernel
def fake_semaphore_kernel(in0, out0, wait_value, end_sem):
    cy = core_index(0)
    value = remote_load(in0, [cy, 0])
    remote_store(out0, [cy, 0], value)
    end_sem.wait(wait_value)


layout = d2m.Layout(
    shape=(64, 64),
    dtype=d2m.float32,
    block_shape=[1, 1],
    grid_shape=[2, 2],
)
input_ = d2m.empty(layout)
output = d2m.empty(layout)
end_sem = d2m.global_semaphore(grid_shape=(8, 8), init=7)
fake_semaphore_kernel(input_, output, 1, end_sem, grid=(2, 2))

builder = _Builder.get()
_emit_returns_and_finalise(builder, [output])
PassManager.parse(
    "builtin.module(ttcore-register-device{mock-system-desc-arch=wormhole_b0})",
    context=builder.ctx,
).run(builder.module.operation)
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK: %[[BACKING:.*]] = d2m.empty() : tensor<8x8x1x1xui32
# CHECK: %[[SEM:.*]] = d2m.create_global_semaphore(%[[BACKING]]) {value = 7 : ui32}
# CHECK: d2m.generic
# CHECK: additionalArgs(%{{.*}}, %[[SEM]] : index, !d2m.global_semaphore)
# CHECK: d2m.semaphore_wait %[[SEM]]
# CHECK: d2m.reset_global_semaphore(%[[SEM]]) {value = 7 : ui32}
