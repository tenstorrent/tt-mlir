# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lower `ttir.exp` to a fused d2m-jit kernel — e2e-device-ready variant.

Identical in spirit to `eltwise_exp_to_kernel.py`, but the kernel takes **no
runtime scalar args**: the single-tile sweep is baked as Python constants. That
is the one constraint that lets the *rewritten* module serialize to a `.ttm`
flatbuffer and run through `ttrt run` end-to-end (rewrite -> compile -> device),
which the scalar-arg form cannot (see AUTHORING.md / testing.py).

The `PatternTest` here sets `e2e=True`, so the harness additionally compiles it
to an in-memory flatbuffer, runs it on device in-process (no ttrt subprocess, no
files), and PCC-checks the device output against the torch golden.
"""

import torch

import d2m_jit as d2m
from d2m_jit.testing import PatternTest
from ttmlir.dialects import ttir


# Single 32x32 tile, fully baked — no m_blocks/n_blocks runtime args.
@d2m.kernel
def exp_e2e(in_t, out_t):
    shard = remote_load(in_t, [0, 0])
    remote_store(out_t, [0, 0], shard.exp())


@d2m.pattern(root=ttir.ExpOp, benefit=10)
def lower_exp_e2e(op, rewriter):
    src = op.operands[0]
    L = d2m.infer_layout(op.result, tiled=True, block_shape=[1, 1], grid_shape=[1, 1])
    x = d2m.to_layout(d2m.from_value(src), L)
    out = d2m.empty(L)
    exp_e2e(x, out, grid=(1, 1))
    return d2m.from_device(out)


def _golden(x):
    return torch.exp(x)


PATTERN_TESTS = [
    PatternTest(
        name="exp_e2e",
        ttir="""
        module {
          func.func @forward(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
            %r = "ttir.exp"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
            return %r : tensor<32x32xf32>
          }
        }
        """,
        golden=_golden,
        e2e=True,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT:   ttir.exp
        CHECK:       d2m.generic
        CHECK:       return %{{.*}} : tensor<32x32xf32>
        """,
    ),
]
