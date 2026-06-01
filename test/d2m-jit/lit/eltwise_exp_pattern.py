# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""Lit test for the single-eltwise pattern (ttir.exp -> d2m subgraph).

Parses a tiny TTIR module containing one `ttir.exp`, invokes the
`d2m_jit` rewrite framework in-process with the bundled
`patterns.eltwise_exp_to_kernel` module loaded, and prints the
rewritten module to stdout. No device needed.

Checks:
  * Original `ttir.exp` is gone (replaced).
  * A `d2m.generic` op was emitted (the @d2m.kernel materialised).
  * The replacement returns a plain `tensor<32x32xf32>` matching the
    matched op's result type — verifies cleanly.
"""

from ttmlir import ir
import d2m_jit as d2m
from d2m_jit._src.rewrite import _registry

# Load the example pattern (registers via @d2m.pattern on import).
_registry.clear()
import d2m_jit.patterns.eltwise_exp_to_kernel  # noqa: F401


ctx = ir.Context()
ctx.load_all_available_dialects()

mod = ir.Module.parse(
    """
module {
  func.func @forward(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %r = "ttir.exp"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
}
""",
    ctx,
)

d2m.apply_patterns(mod)
mod.operation.verify()
print(mod)


# CHECK-LABEL: func.func @forward
# CHECK-NOT:    ttir.exp
# CHECK:        d2m.generic
# CHECK:        return %{{.*}} : tensor<32x32xf32>
