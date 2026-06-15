"""Fixtures for the DTensor tests.
"""

import math
import os

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

import tt_kurbla.torch  # noqa: F401  — registers the "tt" backend (and torch.tt)


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


@pytest.fixture(scope="package", autouse=True)
def _reset_runtime_mesh() -> None:
    """Reset the process-global runtime MeshDevice to single-device after the
    dtensor suite. These tests open a multi-chip mesh via
    `torch.tt.init_device_mesh`, and the runtime mesh persists across tests —
    without this it would leak into later single-chip tests (models/op_tests
    open no mesh of their own and would inherit it).
    """
    yield
    torch.tt.set_mesh_shape(1, 1)


@pytest.fixture(scope="session")
def tt_pg() -> None:
    """Init the tt-backed PG with world_size = num_chips()."""
    n = torch.tt.num_chips()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", str(n))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="tt", rank=0, world_size=n, store=FakeStore()
        )
    yield
