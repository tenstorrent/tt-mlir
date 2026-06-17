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

import d2m_jit as d2m
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


# ============================================================================
# Test Metadata - used by test/d2m-jit/pattern_tests/ test framework
# ============================================================================

import torch

PATTERN_TEST_METADATA = {
    "pattern_name": "eltwise_add_exp",
    "description": "Fused eltwise DAG pattern: ttir.add -> ttir.exp",
    # LIT test configuration
    "lit_tests": [
        {
            "name": "add_exp_pattern_positive",
            "module_text": """
module {
  func.func @positive(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
      -> tensor<32x32xf32> {
    %sum = "ttir.add"(%a, %b) :
        (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %r = "ttir.exp"(%sum) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
}
""",
            "file_checks": [
                "CHECK-LABEL:  func.func @positive",
                "CHECK-NOT:    ttir.exp",
                "CHECK:        d2m.generic",
                "CHECK:        d2m.tile_add",
                "CHECK:        d2m.tile_exp",
                "CHECK:        return %{{.*}} : tensor<32x32xf32>",
            ],
        },
        {
            "name": "add_exp_pattern_negative",
            "module_text": """
module {
  func.func @negative(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %r = "ttir.exp"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
}
""",
            "file_checks": [
                "CHECK-LABEL:  func.func @negative",
                "CHECK:        ttir.exp",
                "CHECK-NOT:    d2m.generic",
            ],
            "description": "Negative case: exp with no upstream add. Pattern should not fire.",
        },
    ],
    # On-device E2E test configuration
    "e2e_tests": [
        {
            "name": "test_pattern_add_exp_kernel_on_device",
            "description": "The add_exp_fused kernel matches torch.exp(a + b) on device",
            "kernel_fn": add_exp_fused,
            "input_generator": lambda: {
                "a": (torch.rand(32, 32, dtype=torch.float32) - 0.5) * 2.0,
                "b": (torch.rand(32, 32, dtype=torch.float32) - 0.5) * 2.0,
            },
            "reference_fn": lambda a, b: torch.exp(a + b),
            "layout_config": {
                "shape": (32, 32),
                "dtype": d2m.float32,
                "block_shape": [1, 1],
                "grid_shape": [1, 1],
                "tiled": True,
            },
            "kernel_args": {
                "m_blocks": 1,
                "n_blocks": 1,
                "grid": (1, 1),
            },
        }
    ],
}
