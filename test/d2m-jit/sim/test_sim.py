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
from d2m_jit._src.layout_math import current_mesh
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


# --- mesh: declaration + per-device storage (SPEC §14.1-14.2) -----------------
#
# The mesh mirror is process-global (no graph lifecycle in sim); the autouse
# `_clear_mesh_mirror` fixture in the parent conftest clears it between tests,
# so declarations here cannot leak per-device allocation into the rest of the
# suite.


@pytest.fixture
def mesh_1x2():
    d2m.mesh((1, 2), topology=("linear", "ring"))


def test_mesh_declaration_validates_like_device():
    """Validation is the shared `validate_mesh_decl`, so shapes/topologies the
    device builder rejects fail identically here."""
    with pytest.raises(ValueError, match="positive integers"):
        d2m.mesh((1, 0))
    with pytest.raises(ValueError, match="rank-2 mesh"):
        d2m.mesh((2,))
    with pytest.raises(ValueError, match="one entry per mesh dimension"):
        d2m.mesh((1, 2), topology=("linear",))
    with pytest.raises(ValueError, match="'disabled', 'linear'"):
        d2m.mesh((1, 2), topology=("linear", "mobius"))
    assert current_mesh() is None  # rejected declarations record nothing

    d2m.mesh((1, 2), topology=("linear", "ring"))
    assert current_mesh()["shape"] == [1, 2]
    # No graph lifecycle in sim: redeclaration replaces (SPEC §14.2).
    d2m.mesh((2, 2))
    assert current_mesh()["shape"] == [2, 2]
    assert current_mesh()["topology"] is None


def test_allocations_are_per_device_under_mesh(mesh_1x2):
    layout = _layout((32, 32))
    t = d2m.zeros(layout)
    assert len(t.buffers) == 2
    # Distinct buffers: mutating one device's copy leaves the other intact.
    t.buffers[0] += 1.0
    assert torch.count_nonzero(t.buffers[1]).item() == 0
    # No single `.buffer` outside a kernel -- host code must be mesh-aware,
    # so a bare read cannot silently alias shard 0 (SPEC §14.1).
    with pytest.raises(RuntimeError, match="per-device buffers"):
        _ = t.buffer
    with pytest.raises(RuntimeError, match="per-device buffers"):
        t.to_host()


def test_host_ops_apply_per_device(mesh_1x2):
    layout = _layout((32, 64))
    data = torch.randn(32, 64)
    t = d2m.to_layout(data, layout)  # replicated onto both devices
    assert len(t.buffers) == 2
    assert torch.equal(t.buffers[0], t.buffers[1])
    assert t.buffers[0] is not t.buffers[1]

    v = d2m.permute(t, 1, 0)  # views permute every device's buffer
    assert len(v.buffers) == 2
    assert v.buffers[1].shape == (64, 32)

    r = d2m.reshape(t, 64, 32)  # host roundtrips run per device
    assert len(r.buffers) == 2
    assert torch.equal(r.buffers[1][:64, :32], data.reshape(64, 32))


def test_mesh_shard_places_chunks_like_the_runtime(mesh_1x2):
    """1x2 mesh, cols sharded: device 0 gets the left half, device 1 the right
    (row-major device order, meshshard_utils.cpp::shard placement)."""
    layout = _layout((32, 32))
    full = torch.randn(32, 64)
    shard = d2m.mesh_shard(full, layout, shard_dims=[0, 1], shard_shape=[1, 2])
    assert len(shard.buffers) == 2
    assert torch.equal(shard.buffers[0], full[:, :32])
    assert torch.equal(shard.buffers[1], full[:, 32:])
    assert shard.mesh.full_shape == [32, 64]
    assert shard.mesh.shard_dims == [0, 1]
    assert shard.mesh.shard_shape == [1, 2]


def test_mesh_shard_2x2_quadrants():
    d2m.mesh((2, 2))
    layout = _layout((32, 32))
    full = torch.randn(64, 64)
    shard = d2m.mesh_shard(full, layout, shard_dims=[0, 1], shard_shape=[2, 2])
    # Row-major over the mesh: (0,0) (0,1) (1,0) (1,1).
    assert torch.equal(shard.buffers[0], full[:32, :32])
    assert torch.equal(shard.buffers[1], full[:32, 32:])
    assert torch.equal(shard.buffers[2], full[32:, :32])
    assert torch.equal(shard.buffers[3], full[32:, 32:])


def test_mesh_shard_replicates_along_minus1_axes():
    """A `-1` mesh axis replicates: every device along it gets a copy of the
    same chunk, matching the runtime's replicate fill (SPEC §14.3)."""
    d2m.mesh((2, 2))
    layout = _layout((32, 32))
    full = torch.randn(32, 64)
    shard = d2m.mesh_shard(full, layout, shard_dims=[-1, 1], shard_shape=[1, 2])
    assert torch.equal(shard.buffers[0], full[:, :32])
    assert torch.equal(shard.buffers[1], full[:, 32:])
    assert torch.equal(shard.buffers[2], full[:, :32])
    assert torch.equal(shard.buffers[3], full[:, 32:])


def test_mesh_shard_validates_like_device(mesh_1x2):
    """The TypeError/RuntimeError/ValueError messages mirror
    `builder.mesh_shard`; the mapping checks are the shared
    `validate_mesh_mapping`/`shard_logical_shape`."""
    layout = _layout((32, 32))
    full = torch.randn(32, 64)
    with pytest.raises(TypeError, match="expects a torch.Tensor"):
        d2m.mesh_shard([[1.0]], layout, shard_dims=[0, 1], shard_shape=[1, 2])
    with pytest.raises(ValueError, match="does not match mesh"):
        d2m.mesh_shard(full, layout, shard_dims=[0, 1], shard_shape=[2, 2])
    with pytest.raises(ValueError, match="not divisible"):
        d2m.mesh_shard(
            torch.randn(32, 63), layout, shard_dims=[0, 1], shard_shape=[1, 2]
        )
    with pytest.raises(ValueError, match="expected per-device shape"):
        d2m.mesh_shard(full, _layout((32, 64)), shard_dims=[0, 1], shard_shape=[1, 2])


def test_mesh_shard_requires_a_mesh():
    with pytest.raises(RuntimeError, match="requires a preceding mesh"):
        d2m.mesh_shard(
            torch.randn(32, 64),
            _layout((32, 32)),
            shard_dims=[0, 1],
            shard_shape=[1, 2],
        )


def test_kernel_over_per_device_tensors_rejected_for_now(mesh_1x2):
    """Per-device *execution* is the SPEC §14.4 step of #9202; until it lands,
    kernels over mesh tensors fail loud instead of computing shard 0 only."""
    layout = _layout((32, 32))
    a = d2m.to_layout(torch.randn(32, 32), layout)
    out = d2m.empty(layout)
    with pytest.raises(NotImplementedError, match="per-device"):
        add(a, a, out, 1, 1, grid=(1, 1))


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
