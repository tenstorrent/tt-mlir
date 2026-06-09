# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for forcing d2m-jit kernel boundary tensors into DRAM."""

import d2m_jit as d2m
from ttmlir.dialects import ttcore

from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise


@d2m.kernel
def k_add(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    remote_store(out, [0, 0], a + b)


L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1])
lhs = d2m.empty(L)
rhs = d2m.empty(L)
out = d2m.empty(L)
k_add(lhs, rhs, out, grid=(1, 1), kernel_io_in_dram=True)
assert out.layout.mem_space == ttcore.MemorySpace.DeviceDRAM

builder = _Builder.get()
_emit_returns_and_finalise(builder, [out])
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK-DAG:   #[[DRAM_LAYOUT:layout[0-9]*]] = #ttcore.metal_layout{{.*}}dram
# CHECK-LABEL: func.func @main
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[DRAM_LAYOUT]]
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[DRAM_LAYOUT]]
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[DRAM_LAYOUT]]
# CHECK:       d2m.generic
# CHECK:       ins({{.*}}#[[DRAM_LAYOUT]]
# CHECK:       outs({{.*}}#[[DRAM_LAYOUT]]
# CHECK:       d2m.remote_load
# CHECK:       d2m.remote_store
