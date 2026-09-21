# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the DTensor tests.
"""

import math

import pytest
import torch

import tt_crank.torch  # noqa: F401  — registers the "tt" backend (and torch.tt)


@pytest.fixture
def mesh_2d_shape() -> tuple[int, int]:
    """Balanced 2-D `(rows, cols)` mesh shape for the host's chip count,
    `rows <= cols`:

        2 -> (1, 2)    4 -> (2, 2)    6 -> (2, 3)    8 -> (2, 4)

    The largest divisor `<= sqrt(n)` becomes `rows`, so the mesh is as square as
    the chip count allows. A prime count yields a degenerate `rows == 1`; that
    still exercises the 2-D code path (two mesh dims, per-axis collectives) with
    one trivial axis.
    """
    n = torch.tt.num_chips()
    rows = 1
    for d in range(1, math.isqrt(n) + 1):
        if n % d == 0:
            rows = d
    return rows, n // rows
