# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for affine-arithmetic d2m-jit view_layout lambdas."""

import d2m_jit as d2m
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise


L = d2m.Layout(shape=(64, 64), dtype=d2m.float32, block_shape=[2, 2], grid_shape=[1, 1])
inp = d2m.empty(L)


def remap(d0, d1, d2, d3):
    feature = d1 * 2 + d3 + 1
    shifted = d1 * 2 - d3 + 3
    return (d0, (feature % 2) // 2, d2, shifted % 2)


view = d2m.view_layout(inp, remap)
out = d2m.to_layout(view, view.layout)

builder = _Builder.get()
_emit_returns_and_finalise(builder, [out])
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK:       affine_map<
# CHECK-SAME:  mod
# CHECK-SAME:  floordiv
# CHECK-SAME:  - d3
# CHECK-LABEL: func.func @main
# CHECK:       d2m.view_layout
