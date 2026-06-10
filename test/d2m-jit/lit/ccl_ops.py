# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape lit test for the CCL kernel-body ops (no device).

Builds an all_gather-shaped kernel and dumps the pre-pipeline d2m IR,
checking the ops the CCL milestone-1 work added:
  * d2m.mesh_position {dim}
  * in-kernel `empty([...])` -> d2m.empty : tensor<...x!tile>
  * explicit-buffer remote_load (loads into the `empty` buffer)
  * cross-device remote_store with startDevice / deviceMcastShape

Semaphore arguments (device_synchronize, semaphore_wait/set on a global
semaphore kernel param) land in a follow-up; this locks the IR contract for
the non-semaphore ops first.
"""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder


L = d2m.Layout(shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2])


@d2m.kernel
def all_gather(in0, out0):
    dy = mesh_position(0)
    dx = mesh_position(1)
    cy = core_index(0)
    cx = core_index(1)
    buf = empty([1, 1])
    remote_load(buf, in0, [cy, 0])
    remote_store(
        out0,
        [dx * 2 + cy, 0],
        buf,
        start_device=[dy, 0],
        device_mcast_shape=[1, 8],
    )


in0 = d2m.empty(L)
out0 = d2m.empty(L)
all_gather(in0, out0, grid=(2, 2))
print(_Builder.get().module)
_Builder.reset()


# CHECK: "d2m.generic"
# CHECK: "d2m.mesh_position"() <{dim = 0 : i64}> : () -> index
# CHECK: "d2m.mesh_position"() <{dim = 1 : i64}> : () -> index
# CHECK: "d2m.core_index"() <{dim = 0 : i64}> : () -> index
# Explicit-buffer load: tensor.empty scratch, then a remote_load whose first
# operand is that buffer (localBuffer segment = 1).
# CHECK: %[[BUF:.*]] = "tensor.empty"() : () -> tensor<1x1x!ttcore.tile<32x32, f32>>
# CHECK: "d2m.remote_load"(%[[BUF]], {{.*}}) <{operandSegmentSizes = array<i32: 1, 1, 2, 0, 0, 0, 0>}>
# Cross-device store: startDevice + deviceMcastShape present, no semaphore.
# CHECK: "d2m.remote_store"({{.*}}) <{operandSegmentSizes = array<i32: 1, 2, 1, 0, 2, 2, 0, 0>}>
