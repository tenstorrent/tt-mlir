# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parity: sim-vs-device PCC parity test (runs a kernel on both backends; "
        "requires a device). Select with `-m parity`, skip with `-m 'not parity'`.",
    )
    config.addinivalue_line(
        "markers",
        "device_only: asserts device-specific behavior -- an intended simulator "
        "divergence (SIMULATOR_SPEC.md §9), a host API the backend switch does "
        "not dispatch, or a device-specific error type (§8). Skipped when the "
        "suite is re-run with D2M_JIT_BACKEND=sim.",
    )


def _sim_backend_requested():
    return os.environ.get("D2M_JIT_BACKEND", "device") == "sim"


def pytest_collection_modifyitems(config, items):
    """Skip `device_only` tests when the whole suite is re-run on the simulator.

    CI runs this directory twice: once on the device, once with
    D2M_JIT_BACKEND=sim (see .github/test_scripts/d2m_jit.sh). Skipping by
    marker rather than by deselecting paths keeps the exclusions visible in the
    junit report instead of silently narrowing what the sim lane covers.
    """
    if not _sim_backend_requested():
        return
    skip_sim = pytest.mark.skip(
        reason="device_only: asserts device behavior that the simulator does "
        "not reproduce (see the marker docs in conftest.py)"
    )
    for item in items:
        if item.get_closest_marker("device_only"):
            item.add_marker(skip_sim)


@pytest.fixture(scope="function", autouse=True)
def _set_seed():
    """Deterministic torch RNG per-test for reproducibility."""
    torch.manual_seed(0)


@pytest.fixture(scope="function", autouse=True)
def _reset_builder():
    """Drop the process-level builder singleton between tests so a failed
    compile (negative tests) doesn't leak MLIR state into the next test."""
    yield
    # Imported here, not at module scope: the simulator suite runs with no MLIR
    # bindings at all, and there is no builder state to drop in that case.
    try:
        from d2m_jit._src.builder import _Builder
    except ImportError:
        return
    _Builder.reset()


def pytest_generate_tests(metafunc):
    """Parametrize the generic pattern tests over every spec declared in the
    bundled pattern files (test/d2m-jit/patterns/*.py). Adding a pattern file with
    PATTERN_TESTS / KERNEL_BENCHES is picked up here with no harness edits."""
    # `runner` imports the MLIR bindings, so only reach for it when a test
    # actually asks for one of the parametrized fixtures -- otherwise collecting
    # the device-free simulator suite would require a tt-metal build.
    if not {"pattern_test", "kernel_bench", "e2e_spec"} & set(metafunc.fixturenames):
        return

    from runner import discover

    pattern_tests, kernel_benches = discover()
    if "pattern_test" in metafunc.fixturenames:
        metafunc.parametrize(
            "pattern_test", pattern_tests, ids=[t.name for t in pattern_tests]
        )
    if "kernel_bench" in metafunc.fixturenames:
        metafunc.parametrize(
            "kernel_bench", kernel_benches, ids=[b.name for b in kernel_benches]
        )
    if "e2e_spec" in metafunc.fixturenames:
        # `golden` is optional: a spec with no golden cross-checks against the
        # ttnn device baseline of its original TTIR (see run_e2e).
        e2e = [t for t in pattern_tests if t.e2e]
        metafunc.parametrize("e2e_spec", e2e, ids=[t.name for t in e2e])


@pytest.fixture(scope="function")
def e2e_device():
    """An in-process mesh-device handle for one e2e test, opened lazily on first
    use and closed afterwards. Function-scoped so at most one device is open at
    a time (the in-process builder device tests open/close their own per call),
    avoiding any cross-test device contention — no subprocess, no marker split.

    For large-scale CI, prefer a single batch driver that opens one device and
    loops over all specs in-process, rather than one pytest case per pattern."""
    from runner import E2EDevice

    holder = E2EDevice()
    yield holder
    holder.close()
