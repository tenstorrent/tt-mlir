# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""Lit test for the fused-eltwise DAG pattern (ttir.add -> ttir.exp).

Two sub-cases:
  * @positive: an exp whose input is the result of an add.
    Pattern fires; the resulting d2m.generic contains both
    `d2m.tile_add` and `d2m.tile_exp` in one body.
  * @negative: an exp whose input is a func arg (no producer).
    Pattern's `match=` predicate fails; no d2m.generic appears.

The single-exp lowering from `eltwise_exp_to_kernel.py` is intentionally
NOT loaded here — the negative case is precisely about "no pattern
matched", not "fell through to the single-op lowering".
"""

from ttmlir import ir
import d2m_jit as d2m
from d2m_jit._src.rewrite import _registry

_registry.clear()
import d2m_jit.patterns.eltwise_add_exp_to_kernel  # noqa: F401


ctx = ir.Context()
ctx.load_all_available_dialects()

# ----------------------------------------------------------------------
# @positive: add -> exp chain. Pattern should fire.
# ----------------------------------------------------------------------
pos = ir.Module.parse(
    """
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
    ctx,
)
d2m.apply_patterns(pos)
pos.operation.verify()
print("=== positive ===")
print(pos)

# CHECK-LABEL: === positive ===
# CHECK-LABEL:  func.func @positive
# CHECK-NOT:    ttir.exp
# CHECK:        d2m.generic
# CHECK:        d2m.tile_add
# CHECK:        d2m.tile_exp
# CHECK:        return %{{.*}} : tensor<32x32xf32>

# ----------------------------------------------------------------------
# @negative: an exp with no upstream add. The same registered pattern
# is applied again on a fresh module; its `match=` predicate fails.
# ----------------------------------------------------------------------
neg = ir.Module.parse(
    """
module {
  func.func @negative(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %r = "ttir.exp"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
}
""",
    ctx,
)
d2m.apply_patterns(neg)
neg.operation.verify()
print("=== negative ===")
print(neg)

# CHECK-LABEL: === negative ===
# CHECK-LABEL:  func.func @negative
# CHECK:        ttir.exp
# CHECK-NOT:    d2m.generic
