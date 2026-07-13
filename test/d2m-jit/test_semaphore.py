# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit semaphore ops (`semaphore_inc` / `semaphore_set`) in the unified
split.

`@d2m.kernel` is unified-only, so a kernel that uses explicit semaphore
mutations is split into compute + datamovement threads like any other. Two
things must hold for the mutation to be correct (see
tools/d2m-jit/unified_semaphore_design.md):

  1. it must stay on a *datamovement* thread -- compute (TRISC) has no
     NOC/fabric access, and `split-unified-thread-v2` used to move it there; and
  2. it must run *exactly once* -- so a kernel containing an explicit mutation is
     kept on a single DM thread (ScheduleDMA's Option D), rather than cloned
     across NOC-processor threads.

`test_semaphore_inc_pinned_to_single_dm_thread` checks both at the IR level (no
device); `test_matmul_semaphore_barrier_runs` runs the kernel end-to-end. The
kernel pairs a matmul (which would otherwise drive a reader/writer DM split)
with a `semaphore_inc` + `semaphore_wait` barrier, so it exercises exactly the
case the split used to get wrong.
"""

import re

import pytest

from utils import assert_pcc

import d2m_jit as d2m
from d2m_jit._src.builder import (
    _Builder,
    _build_pipeline,
    _emit_returns_and_finalise,
    _get_system_desc_path,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from _ttmlir_runtime import runtime
except (ModuleNotFoundError, ImportError):
    runtime = None

from ttmlir.passmanager import PassManager


def _num_devices():
    return runtime.get_num_available_devices() if runtime is not None else 0


@d2m.kernel
def matmul_barrier(lhs, rhs, out, sem):
    # Local per-tile matmul followed by an explicit semaphore barrier. The
    # matmul (load lhs/rhs -> compute -> store out) is what would push the
    # datamovement ops onto separate reader/writer NOC threads; the barrier is
    # the explicit semaphore_inc/wait that must land on a single DM thread.
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    c = a @ b
    cy = core_index(0)
    cx = core_index(1)
    remote_store(out, [0, 0], c)
    semaphore_inc(sem, 1, core=[cy, cx])
    semaphore_wait(sem, 1)


def _layout():
    return d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_semaphore_inc_pinned_to_single_dm_thread():
    """Lower the matmul+barrier kernel and assert the explicit semaphore_inc
    lands on a single datamovement thread (never compute, never duplicated)."""
    L = _layout()
    lhs = torch.randn(32, 32, dtype=torch.float32)
    rhs = torch.randn(32, 32, dtype=torch.float32)
    out_d = d2m.empty(L)
    sem = d2m.global_semaphore(init=0)
    matmul_barrier(
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, sem, grid=(1, 1)
    )

    b = _Builder.get()
    _emit_returns_and_finalise(b, [out_d._resolve()])
    sd = _get_system_desc_path()
    if not sd:
        pytest.skip("no system descriptor available")
    PassManager.parse(
        f"builtin.module(ttcore-register-device{{system-desc-path={sd}}})",
        context=b.ctx,
    ).run(b.module.operation)
    # Run the backend up to (not including) the d2m->ttkernel conversion, where
    # the per-thread region structure is final but still inspectable as d2m.
    passes = _build_pipeline().split(",")
    cut = next(
        i
        for i, p in enumerate(passes)
        if p.startswith("d2m-to-ttkernel-pre-emitc-pipeline")
    )
    PassManager.parse(f"builtin.module({','.join(passes[:cut])})", context=b.ctx).run(
        b.module.operation
    )
    ir = str(b.module)

    # Exactly one explicit semaphore_inc survives (not cloned across threads).
    # (remote_store here carries no semaphore, so this is the only inc.)
    assert ir.count("d2m.semaphore_inc") == 1, ir

    # Every func that holds a semaphore mutation is a datamovement thread.
    func_re = re.compile(
        r"func\.func private @(\w+)\(\) attributes \{d2m\.thread = "
        r"#d2m\.thread<(\w+)"
    )
    starts = [(m.group(1), m.group(2), m.start()) for m in func_re.finditer(ir)]
    for i, (name, thread, start) in enumerate(starts):
        end = starts[i + 1][2] if i + 1 < len(starts) else len(ir)
        body = ir[start:end]
        if "d2m.semaphore_inc" in body or "d2m.semaphore_set" in body:
            assert thread == "datamovement", (
                f"semaphore mutation landed on a {thread} thread ({name}); it "
                f"must be on a datamovement thread"
            )


@d2m.kernel
def gated_fabric(in0, out0, start_sem, end_sem):
    # Multi-core generic: per-core local passthrough, with the cross-device
    # barrier gated to a single core via scf.if on the core index.
    dy = mesh_position(0)
    cy = core_index(0)
    cx = core_index(1)
    buf = remote_load(in0, [cy, cx])
    remote_store(out0, [cy, cx], buf)
    if cx == 0:
        device_synchronize(
            start_sem,
            start_device=[dy, 0],
            mcast_shape=[1, 2],
            num_receivers=1,
            core_indices=[cy, cx],
        )
        semaphore_wait(end_sem, 1)


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_scf_if_gated_fabric_lands_on_dm_thread():
    """A fabric op (device_synchronize) gated behind `if core==0` in a
    multi-core generic must lower onto a *datamovement* thread, not compute.

    The thread-split passes recurse into scf.for and scf.if when classifying
    ops; without the scf.if recursion the whole guarded conditional is treated
    as compute-resident and the fabric op lands on compute (TRISC, no fabric
    access). This lowers the kernel and asserts the placement at the IR level."""
    d2m.mesh((1, 2), topology=("linear", "linear"))
    L = d2m.Layout(
        shape=(32, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 2]
    )
    full = torch.randn(32, 128, dtype=torch.float32)
    in_s = d2m.mesh_shard(full, L, shard_dims=[0, 1], shard_shape=[1, 2])
    out_s = d2m.empty(L)
    ss = d2m.global_semaphore()
    es = d2m.global_semaphore()
    gated_fabric(
        in_s,
        out_s,
        ss,
        es,
        grid=(1, 2),
        fabric=d2m.fabric_config(
            cluster_axis=1, topology="linear", routing="bidir_line_mesh"
        ),
    )
    out = d2m.mesh_gather(out_s, shard_dims=[0, 1], shard_shape=[1, 2])

    b = _Builder.get()
    _emit_returns_and_finalise(b, [out._resolve()])
    sd = _get_system_desc_path()
    if not sd:
        pytest.skip("no system descriptor available")
    PassManager.parse(
        f"builtin.module(ttcore-register-device{{system-desc-path={sd} "
        f"mesh-shape=1,2 mesh-topology=linear,linear}})",
        context=b.ctx,
    ).run(b.module.operation)
    passes = _build_pipeline().split(",")
    cut = next(
        i
        for i, p in enumerate(passes)
        if p.startswith("d2m-to-ttkernel-pre-emitc-pipeline")
    )
    PassManager.parse(f"builtin.module({','.join(passes[:cut])})", context=b.ctx).run(
        b.module.operation
    )
    ir = str(b.module)

    func_re = re.compile(
        r"func\.func private @(\w+)\(\) attributes \{d2m\.thread = "
        r"#d2m\.thread<(\w+)"
    )
    starts = [(m.group(1), m.group(2), m.start()) for m in func_re.finditer(ir)]
    saw_sync = False
    for i, (name, thread, start) in enumerate(starts):
        end = starts[i + 1][2] if i + 1 < len(starts) else len(ir)
        if "d2m.device_synchronize" in ir[start:end]:
            saw_sync = True
            assert thread == "datamovement", (
                f"device_synchronize landed on a {thread} thread ({name}); a "
                f"fabric op must be on a datamovement thread"
            )
    assert saw_sync, "no func with device_synchronize found"


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 1, reason="requires a device")
def test_matmul_semaphore_barrier_runs():
    """End-to-end: the matmul+barrier kernel runs on device and the explicit
    semaphore_inc/wait barrier completes (it would deadlock if the inc were
    misplaced onto compute or split across threads). Output is the matmul."""
    L = _layout()
    lhs = torch.randn(32, 32, dtype=torch.float32)
    rhs = torch.randn(32, 32, dtype=torch.float32)
    out_d = d2m.empty(L)
    sem = d2m.global_semaphore(init=0)
    matmul_barrier(
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, sem, grid=(1, 1)
    )
    result = out_d.to_host()
    assert tuple(result.shape) == (32, 32)
    assert_pcc(lhs @ rhs, result)


@d2m.kernel
def core_read_gather(in0, out0, barrier):
    # grid (1,2): each core loads its 32x32 shard into a local buffer and
    # signals a barrier; then core 0 core_reads core 1's buffer (a cross-core
    # NoC read of a peer's local L1) and stores it. Exercises core_read +
    # cross-core semaphore_inc end to end.
    cy = core_index(0)
    cx = core_index(1)
    buf = empty([1, 1])
    buf = remote_load(buf, in0, [cy, cx])
    semaphore_inc(barrier, 1, core=[0, 0])  # signal core 0 (cross-core for core 1)
    if cx == 0:
        semaphore_wait(barrier, 2)
        peer = empty([1, 1])
        peer = core_read(peer, buf, core=[0, 1])  # read core 1's buffer
        remote_store(out0, [0, 0], peer)
    else:
        remote_store(out0, [cy, cx], buf)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 1, reason="requires a device")
def test_core_read_cross_core_gather():
    """End-to-end core_read: on a grid (1,2), core 0 reads core 1's local L1
    buffer over the NoC (gated by a cross-core semaphore barrier). out[0,0] is
    core 1's shard (gathered), out[0,1] is core 1's own shard."""
    L = d2m.Layout(
        shape=(32, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 2]
    )
    full = torch.randn(32, 64, dtype=torch.float32)
    out_d = d2m.empty(L)
    bar = d2m.global_semaphore(init=0)
    core_read_gather(d2m.to_layout(full, L), out_d, bar, grid=(1, 2))
    result = out_d.to_host()

    half1 = full[:, 32:]  # core 1's shard
    expected = torch.cat([half1, half1], dim=1)
    assert tuple(result.shape) == (32, 64)
    assert_pcc(expected, result)


@d2m.kernel
def core_write_scatter(in0, out0, done):
    # grid (1,2): core 0 writes its shard into core 1's buffer over the NoC
    # (core_write, the push dual of core_read), signals, then core 1 stores
    # that buffer. The receive buffer is also loaded locally so it has a local
    # CB producer; core_write overwrites its data.
    cy = core_index(0)
    cx = core_index(1)
    src = empty([1, 1])
    src = remote_load(src, in0, [cy, cx])
    recv = empty([1, 1])
    recv = remote_load(recv, in0, [cy, cx])
    if cx == 0:
        core_write(src, recv, core=[0, 1])  # write core 0's shard into core 1's recv
        semaphore_inc(done, 1, core=[0, 1])
    else:
        semaphore_wait(done, 1)
        remote_store(out0, [cy, cx], recv)  # core 1 stores recv (= core 0's shard)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 1, reason="requires a device")
def test_core_write_cross_core_scatter():
    """End-to-end core_write: on a grid (1,2), core 0 writes its local shard
    into core 1's local L1 buffer over the NoC (gated by a cross-core
    semaphore). out[0,1] then holds core 0's shard."""
    L = d2m.Layout(
        shape=(32, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 2]
    )
    full = torch.randn(32, 64, dtype=torch.float32)
    out_d = d2m.empty(L)
    done = d2m.global_semaphore(init=0)
    core_write_scatter(d2m.to_layout(full, L), out_d, done, grid=(1, 2))
    result = out_d.to_host()
    # out's second column (core 1) holds core 0's shard, delivered via core_write.
    assert tuple(result.shape) == (32, 64)
    assert_pcc(full[:, :32], result[:, 32:])


@d2m.kernel
def matmul_core_read_gather(lhs, rhs, out, ready):
    # grid (1,2): each core matmuls its own shard (a @ b on the compute thread),
    # then increments a producer-done semaphore with compute=True -- the backend
    # keeps that inc on the datamovement thread but *fences* it behind this
    # core's matmul output CB, so the signal can't fire before the compute
    # finishes. The injector (core 0) waits for both matmuls, then gathers its
    # own tile and core 1's tile via core_read and stores them. A fused matmul ->
    # cross-core all-gather in a single kernel.
    cy = core_index(0)
    cx = core_index(1)
    a = remote_load(lhs, [cy, cx])
    b = remote_load(rhs, [cy, cx])
    c = a @ b
    semaphore_inc(ready, 1, core=[0, 0], compute=True)  # producer-done (fenced)
    if cx == 0:
        semaphore_wait(ready, 2)
        g0 = empty([1, 1])
        g0 = core_read(g0, c, core=[0, 0])  # own tile
        g1 = empty([1, 1])
        g1 = core_read(g1, c, core=[0, 1])  # core 1's tile (cross-core)
        remote_store(out, [0, 0], g0)
        remote_store(out, [0, 1], g1)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
@pytest.mark.skipif(_num_devices() < 1, reason="requires a device")
def test_matmul_core_read_gather():
    """End-to-end fused matmul -> core_read all-gather on a grid (1,2): each core
    matmuls its shard and signals a *producer-done* semaphore (compute=True, so
    the inc is fenced behind the matmul on the datamovement thread); core 0 then
    gathers both tiles via core_read and stores them. Exercises two fixes
    together: the producer-done fence (the readiness signal must not fire before
    the compute -- otherwise the gather reads stale L1) and the aliased-store
    copy-elision skip for core_read dsts (otherwise the two gather buffers alias
    the injector's single local output shard and clobber each other)."""
    L = d2m.Layout(
        shape=(32, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 2]
    )
    a_full = torch.randn(32, 64, dtype=torch.float32)
    b_full = torch.randn(32, 64, dtype=torch.float32)
    out_d = d2m.empty(L)
    ready = d2m.global_semaphore(init=0)
    matmul_core_read_gather(
        d2m.to_layout(a_full, L), d2m.to_layout(b_full, L), out_d, ready, grid=(1, 2)
    )
    result = out_d.to_host()
    exp0 = a_full[:, :32] @ b_full[:, :32]
    exp1 = a_full[:, 32:] @ b_full[:, 32:]
    expected = torch.cat([exp0, exp1], dim=1)
    assert tuple(result.shape) == (32, 64)
    assert_pcc(expected, result)
