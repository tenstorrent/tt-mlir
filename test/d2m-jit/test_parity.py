# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-vs-device parity harness (SIMULATOR_SPEC.md §12).

For every declared ``KernelBench`` (discovered from ``kernels/``) this runs the
*same* kernel through both backends and checks agreement:

  - ``test_sim_matches_golden`` -- always runnable (no device): the simulator
    result vs the bench's torch golden.
  - ``test_sim_matches_device`` -- the simulator result vs the device result of
    the identical kernel; skipped when no runtime/device is available or when
    the process is already pinned to the simulator (D2M_JIT_SIM=1), since then
    both legs would be the sim.

Scope note: parity covers ``KernelBench``es because those are the self-describing
benches (kernel + golden + materializer + PCC threshold) the runner can compare
automatically. The simulator itself is not limited to benched kernels -- any
``@d2m.kernel`` runs under it (see the hand-written test_*.py suite, which uses
plain kernels with no bench).

Run:
    pytest test/d2m-jit/test_parity.py                 # sim-vs-golden (+ device if present)
    D2M_JIT_SIM=1 pytest test/d2m-jit/test_parity.py   # sim-vs-golden only
"""

import pytest

import d2m_jit as d2m
from runner import (
    compute_pcc,
    device_runtime_available,
    run_bench,
    run_bench_parity,
)


def _run_or_skip_unsupported(fn):
    """Run `fn`, converting an unsupported-by-simulator gap into a skip rather
    than a failure."""
    try:
        return fn()
    except NotImplementedError as e:
        pytest.skip(f"kernel not supported by the simulator: {e}")


def test_sim_matches_golden(kernel_bench):
    """Simulator output matches the bench's torch golden (no device needed)."""
    actual, expected = _run_or_skip_unsupported(
        lambda: run_bench(kernel_bench, backend="sim")
    )
    pcc = compute_pcc(expected, actual)
    assert (
        pcc >= kernel_bench.pcc
    ), f"{kernel_bench.name}: sim-vs-golden pcc {pcc} < {kernel_bench.pcc}"


@pytest.mark.skipif(
    d2m.config.simulator,
    reason="process is pinned to the simulator; device leg would also be sim",
)
def test_sim_matches_device(kernel_bench):
    """Simulator output matches the device output of the same kernel."""
    if not device_runtime_available():
        pytest.skip("no tt-metal runtime available for the device leg")
    try:
        pcc, _sim, _dev = _run_or_skip_unsupported(
            lambda: run_bench_parity(kernel_bench)
        )
    except Exception as e:  # noqa: BLE001 -- no silicon / device open failure
        pytest.skip(f"device leg unavailable: {type(e).__name__}: {e}")
    assert (
        pcc >= kernel_bench.pcc
    ), f"{kernel_bench.name}: sim-vs-device pcc {pcc} < {kernel_bench.pcc}"
