# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for the d2m-jit RoPE prefill helper."""

import torch

import d2m_jit as d2m
from kernels.prefill.rope import apply_rope, build_rope_tables
from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise


L = d2m.Layout(
    shape=(64, 64),
    dtype=d2m.float32,
    block_shape=[2, 2],
    grid_shape=[1, 1],
)

x = torch.randn(64, 64, dtype=torch.float32)
cos, sin_signed = build_rope_tables(64, 64, dtype=torch.float32)
out = apply_rope(
    d2m.to_layout(x, L),
    d2m.to_layout(cos, L),
    d2m.to_layout(sin_signed, L),
    L,
    grid=(1, 1),
    m_blocks=1,
    n_blocks=1,
)

builder = _Builder.get()
_emit_returns_and_finalise(builder, [out])
builder.module.operation.verify()
print(builder.module)
_Builder.reset()


# CHECK-DAG:   affine_map<{{.*}}mod{{.*}}>
# CHECK-LABEL: func.func @main
# CHECK:       d2m.view_layout
# CHECK:       d2m.generic
# CHECK:       d2m.remote_load
# CHECK:       "d2m.tile_mul"
# CHECK:       "d2m.tile_add"
