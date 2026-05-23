# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit float reductions.

This stays below `to_host()` so lit can validate the builder contract without
requiring device execution.
"""

import torch
import d2m_jit as d2m

from d2m_jit._src.builder import _Builder

_L = d2m.Layout(
    shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
)


@d2m.kernel
def k(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    s = reduce_sum(x, 1)
    m = x.reduce_max(0)
    a = reduce_mean(x, -1)
    y = s.add(m).add(a)
    remote_store(out_t, [0, 0], y)


t = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
k(d2m.to_layout(t, _L), d2m.empty(_L), grid=(1, 1))
print(_Builder.get().module)
_Builder.reset()


# CHECK: "func.func"
# CHECK: d2m.to_layout
# CHECK: "d2m.generic"
# CHECK: d2m.remote_load
# CHECK: d2m.tile_reduce_sum
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
# CHECK: d2m.tile_reduce_max
# CHECK-SAME: reduce_dim = #d2m<reduce_dim C>
# CHECK: d2m.tile_reduce_mean
# CHECK-SAME: reduce_dim = #d2m<reduce_dim R>
