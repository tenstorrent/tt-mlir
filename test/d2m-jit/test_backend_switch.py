# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`config.backend` switch on the canonical `import d2m_jit` surface.

The device half of the switch needs silicon to run `to_host`, so here we only
exercise the device-free parts: that the sim backend dispatches and computes
correctly, that the device backend still builds the lazy graph, and that an
invalid backend is rejected. The autouse fixture restores the process-global
`config.backend` so these tests cannot leak into the device tests.
"""

import pytest
import torch

import d2m_jit as d2m
from d2m_jit._src.sim import SimTensor
from d2m_jit._src.builder import LazyTensor


@pytest.fixture(autouse=True)
def _restore_backend():
    saved = d2m.config.backend
    yield
    d2m.config.backend = saved


@d2m.kernel
def add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a + b)


def _layout():
    return d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )


def test_sim_backend_dispatch_and_run():
    d2m.config.backend = "sim"
    layout = _layout()
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)
    a = d2m.to_layout(lhs, layout)
    assert isinstance(a, SimTensor)
    out = d2m.empty(layout)
    add(a, d2m.to_layout(rhs, layout), out, 1, 1, grid=(2, 2))
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)


def test_device_backend_builds_lazy_graph():
    d2m.config.backend = "device"
    lt = d2m.to_layout(torch.randn(64, 64), _layout())
    assert isinstance(lt, LazyTensor)  # no to_host (needs a device)


def test_invalid_backend_raises():
    d2m.config.backend = "nonsense"
    with pytest.raises(ValueError, match="backend"):
        d2m.empty(_layout())


def test_kernel_decorator_picks_backend_per_call():
    # The same @d2m.kernel object dispatches by backend at call time.
    layout = _layout()
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)

    d2m.config.backend = "sim"
    out = d2m.empty(layout)
    add(d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, 1, 1, grid=(2, 2))
    assert isinstance(out, SimTensor)
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)
