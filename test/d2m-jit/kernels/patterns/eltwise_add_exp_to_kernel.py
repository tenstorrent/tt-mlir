# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fuse a `ttir.add → ttir.exp` chain into a single d2m-jit kernel.

DAG match example. The pattern roots on `ttir.exp` and fires only when
its input is the result of a `ttir.add`. The replacement kernel computes
`(a + b).exp()` in a single tile-level pass — no intermediate tensor
materialised between the add and the exp.

The matched `ttir.exp` is replaced; the upstream `ttir.add` becomes
dead and is reclaimed by canonicalisation (`pdl.replace` doesn't erase
the producer for us). If the add has other users besides the exp, those
keep working — the add stays alive in that case.
"""

import torch

import d2m_jit as d2m
from runner import InputSpec, KernelBench, PatternTest, TensorSpec, eltwise_block_run
from ttmlir import ir
from ttmlir.dialects import ttir


# Fused block-level kernel: (a + b).exp(). One thread per (Y, X) grid
# core; each core sweeps m_blocks × n_blocks tile blocks.
@d2m.kernel
def add_exp_fused(a, b, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            xa = remote_load(a, [m_off + m, n_off + n])
            xb = remote_load(b, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], (xa + xb).exp())


def _exp_input_is_add(op):
    """Predicate (PDL constraint). Must be pure — no IR mutation."""
    producer = op.operands[0].owner
    return isinstance(producer, ir.Operation) and producer.name == "ttir.add"


@d2m.pattern(root=ttir.ExpOp, benefit=20, match=_exp_input_is_add)
def fuse_add_exp(op, rewriter):
    """Replace `ttir.exp(ttir.add(a, b))` with one fused d2m subgraph.

    Benefit 20 outranks the single-eltwise `lower_exp` pattern (benefit
    10) so the fused form wins when both apply.
    """
    add_op = op.operands[0].owner
    a_v = add_op.operands[0]
    b_v = add_op.operands[1]

    L = d2m.infer_layout(
        op.result,
        tiled=True,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )

    a = d2m.to_layout(d2m.from_value(a_v), L)
    b = d2m.to_layout(d2m.from_value(b_v), L)
    out = d2m.empty(L)
    add_exp_fused(a, b, out, 1, 1, grid=(1, 1))
    return d2m.from_device(out)


# ----------------------------------------------------------------------
# Co-located tests. See testing.py for the runner contract.
# ----------------------------------------------------------------------


def _golden(a, b):
    return torch.exp(a + b)


# Rewrite correctness (replaces test/d2m-jit/lit/eltwise_add_exp_pattern.py).
PATTERN_TESTS = [
    # add -> exp chain. The DAG match fires; the d2m.generic body holds both
    # tile_add and tile_exp (fully fused, no intermediate tensor).
    PatternTest(
        name="add_exp_positive",
        ttir="""
        module {
          func.func @f(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
              -> tensor<32x32xf32> {
            %sum = "ttir.add"(%a, %b) :
                (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
            %r = "ttir.exp"(%sum) : (tensor<32x32xf32>) -> tensor<32x32xf32>
            return %r : tensor<32x32xf32>
          }
        }
        """,
        golden=_golden,
        inputs=InputSpec("uniform(-1,1)"),
        # `m_blocks, n_blocks` runtime scalars are baked into the kernel body as
        # constants in the rewrite scope (see AUTHORING.md), so this compiles to
        # a flatbuffer and runs e2e on device directly.
        e2e=True,
        check="""
        CHECK-LABEL: func.func @f
        CHECK-NOT:   ttir.exp
        CHECK:       d2m.generic
        CHECK:       d2m.tile_add
        CHECK:       d2m.tile_exp
        CHECK:       return %{{.*}} : tensor<32x32xf32>
        """,
    ),
    # exp with no upstream add: the match= predicate fails, nothing rewrites.
    # Only this file's pattern is loaded, so there is no single-exp fallthrough.
    PatternTest(
        name="add_exp_negative",
        ttir="""
        module {
          func.func @g(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
            %r = "ttir.exp"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
            return %r : tensor<32x32xf32>
          }
        }
        """,
        expect_match=False,
        check="""
        CHECK-LABEL: func.func @g
        CHECK:       ttir.exp
        CHECK-NOT:   d2m.generic
        """,
    ),
]

# On-device numerics (replaces test_pattern_add_exp_kernel_on_device).
KERNEL_BENCHES = {
    "add_exp": KernelBench(
        kernel=add_exp_fused,
        golden=_golden,
        run=eltwise_block_run,
        tensors=[
            TensorSpec(
                shape=(32, 32),
                block_shape=[1, 1],
                dtype=torch.float32,
                dist="uniform(-1,1)",
            ),
            TensorSpec(
                shape=(32, 32),
                block_shape=[1, 1],
                dtype=torch.float32,
                dist="uniform(-1,1)",
            ),
        ],
        grid_shape=(1, 1),
    )
}
