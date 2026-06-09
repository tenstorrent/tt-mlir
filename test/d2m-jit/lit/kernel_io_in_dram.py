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


def emit_kernel_io_in_dram(label, *, kernel_io_in_dram=None, config_default=False):
    old_config = d2m.config.kernel_io_in_dram
    try:
        d2m.config.kernel_io_in_dram = config_default
        L = d2m.Layout(shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1])
        lhs = d2m.empty(L)
        rhs = d2m.empty(L)
        out = d2m.empty(L)

        if kernel_io_in_dram is None:
            k_add(lhs, rhs, out, grid=(1, 1))
        else:
            k_add(lhs, rhs, out, grid=(1, 1), kernel_io_in_dram=kernel_io_in_dram)
        assert out.layout.mem_space == ttcore.MemorySpace.DeviceDRAM

        builder = _Builder.get()
        _emit_returns_and_finalise(builder, [out])
        builder.module.operation.verify()
        print(label)
        print(builder.module)
    finally:
        d2m.config.kernel_io_in_dram = old_config
        _Builder.reset()


emit_kernel_io_in_dram("EXPLICIT_OVERRIDE_IR", kernel_io_in_dram=True)
emit_kernel_io_in_dram("CONFIG_DEFAULT_IR", config_default=True)


# CHECK-LABEL: EXPLICIT_OVERRIDE_IR
# CHECK-DAG:   #[[EXPLICIT_DRAM_LAYOUT:layout[0-9]*]] = #ttcore.metal_layout{{.*}}dram
# CHECK-LABEL: func.func @main
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[EXPLICIT_DRAM_LAYOUT]]
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[EXPLICIT_DRAM_LAYOUT]]
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[EXPLICIT_DRAM_LAYOUT]]
# CHECK:       d2m.generic
# CHECK:       ins({{.*}}#[[EXPLICIT_DRAM_LAYOUT]]
# CHECK:       outs({{.*}}#[[EXPLICIT_DRAM_LAYOUT]]
# CHECK:       d2m.remote_load
# CHECK:       d2m.remote_store

# CHECK-LABEL: CONFIG_DEFAULT_IR
# CHECK-DAG:   #[[CONFIG_DRAM_LAYOUT:layout[0-9]*]] = #ttcore.metal_layout{{.*}}dram
# CHECK-LABEL: func.func @main
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[CONFIG_DRAM_LAYOUT]]
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[CONFIG_DRAM_LAYOUT]]
# CHECK:       d2m.to_layout
# CHECK-SAME:  #[[CONFIG_DRAM_LAYOUT]]
# CHECK:       d2m.generic
# CHECK:       ins({{.*}}#[[CONFIG_DRAM_LAYOUT]]
# CHECK:       outs({{.*}}#[[CONFIG_DRAM_LAYOUT]]
# CHECK:       d2m.remote_load
# CHECK:       d2m.remote_store
