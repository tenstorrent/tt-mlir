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
    sem = d2m.global_semaphore(grid_shape=(8, 8), init=0)
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
    cut = passes.index("d2m-to-ttkernel-pre-emitc-pipeline")
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
    sem = d2m.global_semaphore(grid_shape=(8, 8), init=0)
    matmul_barrier(
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, sem, grid=(1, 1)
    )
    result = out_d.to_host()
    assert tuple(result.shape) == (32, 32)
    assert_pcc(lhs @ rhs, result)
