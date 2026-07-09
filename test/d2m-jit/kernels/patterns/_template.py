# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TEMPLATE: a self-contained d2m-jit pattern with co-located tests.

Copy this file to a descriptive, NON-underscore name to activate it, e.g.

    cp _template.py eltwise_mul_relu_to_kernel.py

Discovery skips any `_`-prefixed file, so this template is never collected
as a test.

A pattern file is the complete unit — kernel + rewrite + tests — and declares
two module-level lists the generic runner picks up automatically:

    PATTERN_TESTS  : rewrite correctness, checked with FileCheck (no device).
    KERNEL_BENCHES : on-device numerics, PCC-compared against a torch golden.

Run just this pattern after copying:

    pytest test/d2m-jit/test_patterns.py -k <kernel_or_test_name>

See `eltwise_exp_to_kernel.py` / `eltwise_add_exp_to_kernel.py` for worked
examples. The body below is a runnable `ttir.exp` example; replace the TODO-
marked parts with your op.
"""

import torch

import d2m_jit as d2m
from runner import InputSpec, KernelBench, PatternTest, TensorSpec, eltwise_block_run
from ttmlir.dialects import ttir


# ----------------------------------------------------------------------
# 1. Kernel — the fused tile-level computation the pattern emits.
#    Signature convention for eltwise_block_run: (in..., out, m_blocks,
#    n_blocks). One thread per (Y, X) grid core; each core sweeps
#    m_blocks x n_blocks tile blocks.
# ----------------------------------------------------------------------
@d2m.kernel
def template_kernel(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            shard = remote_load(in_t, [m_off + m, n_off + n])
            # TODO: replace `.exp()` with your fused tile computation.
            remote_store(out_t, [m_off + m, n_off + n], shard.exp())


# ----------------------------------------------------------------------
# 2. Pattern — match a TTIR op (DAG) and replace it with the kernel.
#    `benefit` breaks ties when multiple patterns match the same root;
#    higher wins. For DAG matches, add `match=<pure predicate>` and root
#    on the tail op (see eltwise_add_exp_to_kernel.py).
# ----------------------------------------------------------------------
@d2m.pattern(root=ttir.ExpOp, benefit=10)  # TODO: your root op
def lower_template(op, rewriter):
    src = op.operands[0]
    L = d2m.infer_layout(op.result, tiled=True, block_shape=[1, 1], grid_shape=[1, 1])
    x = d2m.to_layout(d2m.from_value(src), L)
    out = d2m.empty(L)
    template_kernel(x, out, 1, 1, grid=(1, 1))
    return d2m.from_device(out)


# ----------------------------------------------------------------------
# 3. Tests — co-located with the pattern they exercise.
# ----------------------------------------------------------------------


def _golden(x):
    # TODO: torch reference for your op. Args match the func args in `ttir`
    # below (and input_shapes in KERNEL_BENCH), in order.
    return torch.exp(x)


# Rewrite correctness: input TTIR -> apply this file's pattern(s) -> FileCheck.
PATTERN_TESTS = [
    PatternTest(
        name="template_positive",
        # The func signature is the single source of truth for input shapes
        # and dtypes; the e2e device runner reads them from here.
        ttir="""
        module {
          func.func @forward(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
            %r = "ttir.exp"(%x) : (tensor<32x32xf32>) -> tensor<32x32xf32>
            return %r : tensor<32x32xf32>
          }
        }
        """,
        golden=_golden,
        inputs=InputSpec("uniform(-1,1)"),
        # FileCheck directives. Assert the matched op is gone and the fused
        # d2m subgraph appears. `CHECK-NOT` between two `CHECK`s scopes the
        # negative to that span.
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT:   ttir.exp
        CHECK:       d2m.generic
        CHECK:       return %{{.*}} : tensor<32x32xf32>
        """,
        tags=("template",),
    ),
    # Optional negative case: an input the pattern must NOT match. Set
    # expect_match=False and assert the root op survives with no d2m.generic.
    # PatternTest(
    #     name="template_negative",
    #     ttir="...",
    #     expect_match=False,
    #     check="CHECK: ttir.exp\nCHECK-NOT: d2m.generic",
    # ),
]

# On-device numerics: drive the kernel directly and PCC-compare vs torch.
# Reuse eltwise_block_run for the common elementwise-block shape; write a
# custom run(kernel, inputs, tensors, grid_shape) -> host tensor for
# anything else.
KERNEL_BENCHES = {
    "template": KernelBench(
        kernel=template_kernel,
        golden=_golden,
        run=eltwise_block_run,
        tensors=[
            TensorSpec(
                shape=(32, 32),
                block_shape=[1, 1],
                dtype=torch.float32,
                dist="uniform(-1,1)",
            )
        ],
        grid_shape=(1, 1),
    )
}
