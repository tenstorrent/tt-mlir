# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator-specific tests. No device, no SYSTEM_DESC_PATH.

Scope: this file deliberately does **not** re-cover the op surface. CI re-runs
the whole device suite with `D2M_JIT_BACKEND=sim` (see
.github/test_scripts/d2m_jit.sh), so eltwise, reductions, matmul, comparisons,
views, broadcasts and the host ops are all checked against the kernels people
actually write -- in the device suite one directory up (test_eltwise.py,
test_ops.py, test_reductions.py, test_matmul.py, test_compare.py, test_views.py,
test_bespoke.py, test_broadcasts.py, test_zeros_full_where.py). Duplicating them
here would only test kernels somebody remembered to copy over.

What lives here is what that re-run cannot reach:

  * the shadow surface `import d2m_jit.sim` -- the re-run goes through
    `import d2m_jit` plus `config.backend`, so nothing else exercises
    tools/d2m-jit/sim.py at all;
  * intended divergences from device (SIMULATOR_SPEC.md §9) -- assertions that
    hold by construction in sim and are undefined on device, so a shared test
    could not make them;
  * simulator-only rejections and internals: `async def` bodies, the declarative
    generic form, the in-kernel `zeros` block;
  * the runtime-free import property (§2).

The op-registry audit (every device-registered in-kernel name has a sim backing)
lives in test_backend_switch.py, since reading the device registry needs the
MLIR bindings that this file deliberately does without.

See tools/d2m-jit/SIMULATOR_SPEC.md.
"""

import subprocess
import sys

import pytest
import torch

import d2m_jit.sim as d2m
from utils import assert_pcc


def _layout(shape, block_shape=(1, 1), grid=(1, 1), dtype=d2m.float32):
    return d2m.Layout(
        shape=shape, dtype=dtype, block_shape=list(block_shape), grid_shape=list(grid)
    )


def _blocks_per_core(shape, block_shape, grid):
    mt, nt = shape[0] // 32, shape[1] // 32
    return mt // block_shape[0] // grid[0], nt // block_shape[1] // grid[1]


# --- kernels -----------------------------------------------------------------


@d2m.kernel
def add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a + b)


@d2m.kernel
def matmul_kernel(lhs, rhs, out):
    remote_store(out, [0, 0], remote_load(lhs, [0, 0]) @ remote_load(rhs, [0, 0]))


# --- shadow surface ----------------------------------------------------------


def test_shadow_module_runs_a_kernel_end_to_end():
    """`import d2m_jit.sim` on its own: host ops, a multi-core SPMD kernel, and
    `to_host`, with no `config.backend` involved. Nothing else in the suite
    imports the shadow module, so without this tools/d2m-jit/sim.py is untested."""
    lhs, rhs = torch.randn(512, 512), torch.randn(512, 512)
    lin = _layout((512, 512), grid=(8, 8))
    lout = _layout((512, 512), grid=(2, 2))
    out = d2m.empty(lout)
    mb, nb = _blocks_per_core((512, 512), (1, 1), (2, 2))
    add(d2m.to_layout(lhs, lin), d2m.to_layout(rhs, lin), out, mb, nb, grid=(2, 2))
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)


# --- intended divergences from device (§9) -----------------------------------


def test_empty_is_zero():
    """Device `empty` contents are undefined; in sim they are zero, which makes
    runs deterministic. A shared test could not assert this -- see §9. (`zeros`
    and `full` are ordinary shared behavior and live in test_zeros_full_where.py.)"""
    layout = _layout((32, 32))
    assert torch.count_nonzero(d2m.empty(layout).to_host()).item() == 0


def test_matmul_into_empty_needs_no_prefill():
    """Because sim `empty` is zero, matmul into a raw `empty` output is the
    correct product. On device this hits the undefined-accumulator bug (TODO §1),
    which is why the device matmul tests prefill with `zeros` or carry an
    accumulator -- so this assertion is sim-only by construction (§9)."""
    lhs, rhs = torch.randn(32, 32), torch.randn(32, 32)
    layout = _layout((32, 32))
    out = d2m.empty(layout)  # deliberately not zeros()
    matmul_kernel(
        d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, grid=(1, 1)
    )
    assert_pcc(lhs @ rhs, out.to_host())


# --- simulator internals -----------------------------------------------------


def test_kernel_zeros_block_is_zero_and_f32():
    from d2m_jit._src.sim.ops import zeros as sim_zeros

    block = sim_zeros([2, 3])
    assert block.tile_grid == (2, 3)
    assert block.tiles.dtype == torch.float32  # device tile type is f32
    assert torch.count_nonzero(block.tiles).item() == 0
    with pytest.raises(NotImplementedError, match="rank-2"):
        sim_zeros([1, 1, 1])


# The audit that every device-registered in-kernel name has a sim backing needs
# the device registry, so it lives in test_backend_switch.py rather than carving
# a bindings-dependent exception out of this file's runtime-free guarantee.


# --- simulator-only rejections -----------------------------------------------


def test_kernel_rejects_scalar_before_tensor():
    """Sim raises the arg-order TypeError from its own validator. The device
    equivalent in test_errors.py asserts `D2mJitError` and is `device_only`, so
    this path is only covered here."""
    layout = _layout((32, 32))
    a = d2m.to_layout(torch.randn(32, 32), layout)
    out = d2m.empty(layout)
    with pytest.raises(TypeError, match="must precede scalars"):
        add(a, 1, out, 1, 1, grid=(1, 1))


def test_kernel_rejects_declarative_form():
    layout = _layout((32, 32))
    a = d2m.to_layout(torch.randn(32, 32), layout)
    out = d2m.empty(layout)
    with pytest.raises(NotImplementedError):
        add(a, a, out, 1, 1, grid=(1, 1), iterator_types=["parallel", "parallel"])


# --- async / semaphores ------------------------------------------------------


@d2m.kernel
async def add_async(lhs, rhs, out, m_blocks, n_blocks):
    sem = Semaphore(0)
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            a = await a
            sem.inc(1)
            sem.wait(1, reset=0)
            remote_store(out, [m_off + m, n_off + n], a + b)


def test_async_kernel_await_and_semaphore():
    """An `async def` body (await + Semaphore no-ops) is driven to completion
    and matches the synchronous result. Semaphores are ordering-only, so they
    do not affect numerics in the functional simulator. No device test uses an
    async body, so the sim re-run never reaches this path."""
    lin = _layout((64, 64), grid=(2, 2))
    lhs, rhs = torch.randn(64, 64), torch.randn(64, 64)
    out = d2m.empty(lin)
    add_async(d2m.to_layout(lhs, lin), d2m.to_layout(rhs, lin), out, 1, 1, grid=(2, 2))
    assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5)


@d2m.kernel
async def gen_kernel(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    yield x
    remote_store(out_t, [0, 0], x)


def test_async_generator_kernel_rejected():
    """`async def` + `yield` (multi-thread producer/consumer) needs an ordering
    model the simulator omits; it must fail loudly rather than silently no-op."""
    layout = _layout((32, 32))
    t = torch.randn(32, 32)
    out = d2m.empty(layout)
    with pytest.raises(NotImplementedError, match="async-generator"):
        gen_kernel(d2m.to_layout(t, layout), out, grid=(1, 1))


# --- runtime-free import -----------------------------------------------------


# Run in a subprocess so blocking the bindings cannot disturb this interpreter,
# which has them loaded. The blocker makes `ttmlir` / `_ttmlir_runtime`
# unimportable, standing in for a plain python+torch image with no tt-metal
# build; the sim must import and run a kernel anyway (SIMULATOR_SPEC.md §2).
_NO_BINDINGS_SCRIPT = """
import importlib.abc, sys


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("ttmlir", "_ttmlir_runtime"):
            raise ImportError("blocked for test: " + name)
        return None


sys.meta_path.insert(0, Blocker())

import torch

import d2m_jit.sim as d2m

leaked = [m for m in sys.modules if m.split(".")[0] in ("ttmlir", "_ttmlir_runtime")]
assert not leaked, "sim import pulled in the bindings: %s" % leaked

layout = d2m.Layout(
    shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
)


@d2m.kernel
def add_one_block(lhs, rhs, out):
    remote_store(out, [0, 0], remote_load(lhs, [0, 0]) + remote_load(rhs, [0, 0]))


lhs, rhs = torch.randn(32, 32), torch.randn(32, 32)
out = d2m.empty(layout)
add_one_block(
    d2m.to_layout(lhs, layout), d2m.to_layout(rhs, layout), out, grid=(1, 1)
)
assert torch.allclose(lhs + rhs, out.to_host(), atol=1e-5), "wrong result"
print("OK")
"""


def test_sim_imports_and_runs_without_mlir_bindings():
    """`import d2m_jit.sim` must not require `ttmlir` / `_ttmlir_runtime`.

    This is the property that lets the sim suite run on a plain python+torch
    image. It is easy to regress by adding a module-scope import to any module
    on the sim import path (`d2m_jit/__init__.py`, `_src/tensor_layout.py`,
    `_src/sim/*`), so it is asserted rather than documented.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _NO_BINDINGS_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"sim import/run failed without the MLIR bindings:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "OK" in proc.stdout
