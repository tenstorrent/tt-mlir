# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""On-device end-to-end test for the bundled eltwise pattern kernels.

Drives the kernels from `tools/d2m-jit/patterns/` directly through the
lazy d2m-jit path (no pattern rewrite step at run time) and PCC-compares
against torch.

This validates the kernel templates produce correct silicon output. The
lit tests under `test/d2m-jit/lit/` validate that the *rewrite path*
emits these exact kernels, so the two together cover the framework
end-to-end:

  lit:    TTIR input -> apply_patterns -> rewritten IR (FileCheck'd)
  pytest: same kernel -> d2m pipeline -> flatbuffer -> ttrt -> assert PCC
"""

import torch
from utils import assert_pcc

import d2m_jit as d2m
from d2m_jit.patterns.eltwise_exp_to_kernel import exp_fused
from d2m_jit.patterns.eltwise_add_exp_to_kernel import add_exp_fused


def test_pattern_exp_kernel_on_device():
    """The exp_fused kernel from `eltwise_exp_to_kernel.py` matches torch.exp.

    The pattern uses tiled L1 with block_shape=[1,1] and grid=(1,1); we
    mirror that here. Input ~Uniform(-1, 1) so exp stays in a numerically
    well-behaved range for PCC scoring.
    """
    torch.manual_seed(0)
    x = (torch.rand(32, 32, dtype=torch.float32) - 0.5) * 2.0  # in (-1, 1)

    L = d2m.Layout(
        shape=x.shape,
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
        tiled=True,
    )

    x_d = d2m.to_layout(x, L)
    out_d = d2m.empty(L)
    exp_fused(x_d, out_d, 1, 1, grid=(1, 1))
    out = out_d.to_host()

    expected = torch.exp(x)
    assert_pcc(expected, out)


def test_pattern_add_exp_kernel_on_device():
    """The add_exp_fused kernel from `eltwise_add_exp_to_kernel.py` matches
    torch.exp(a + b) on silicon. Same single-tile layout as the pattern.
    """
    torch.manual_seed(0)
    a = (torch.rand(32, 32, dtype=torch.float32) - 0.5) * 2.0
    b = (torch.rand(32, 32, dtype=torch.float32) - 0.5) * 2.0

    L = d2m.Layout(
        shape=a.shape,
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
        tiled=True,
    )

    a_d = d2m.to_layout(a, L)
    b_d = d2m.to_layout(b, L)
    out_d = d2m.empty(L)
    add_exp_fused(a_d, b_d, out_d, 1, 1, grid=(1, 1))
    out = out_d.to_host()

    expected = torch.exp(a + b)
    assert_pcc(expected, out)
