# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: d2m-jit

"""__matmul__ via linalg.generic + d2m.tile_matmul compile + run smoke.

The linalg.generic body accumulates into the output operand; until we
zero-initialise that output (linalg.fill with a d2m.fill_tile zero) the
numerical result is undefined. This test only confirms:
  - the IR compiles through the d2m -> ttmetal pipeline;
  - the binary executes on device;
  - the output tensor has the expected shape and dtype.
"""

import torch
import d2m_jit as d2m


@d2m.kernel
def matmul_kernel(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            c = a @ b
            remote_store(out, [m_off + m, n_off + n], c)


def test_matmul_compiles_and_runs():
    lhs = torch.eye(64, dtype=torch.float32)
    rhs = torch.eye(64, dtype=torch.float32)
    L = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )
    out_d = d2m.empty(L)
    matmul_kernel(
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, 1, 1, grid=(2, 2)
    )
    result = out_d.to_host()
    assert tuple(result.shape) == (64, 64)
    assert result.dtype == torch.float32


test_matmul_compiles_and_runs()
print("PASS matmul")
