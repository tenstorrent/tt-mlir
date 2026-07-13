# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from d2m_jit._src.builder import _Builder, _close_cached_device


@pytest.fixture(scope="function", autouse=True)
def _set_seed():
    """Deterministic torch RNG per-test for reproducibility."""
    torch.manual_seed(0)


@pytest.fixture(scope="function", autouse=True)
def _reset_builder():
    """Drop the process-level builder singleton between tests so a failed
    compile (negative tests) doesn't leak MLIR state into the next test."""
    yield
    _Builder.reset()


def pytest_generate_tests(metafunc):
    """Parametrize the generic pattern tests over every spec declared in the
    bundled pattern files (test/d2m-jit/patterns/*.py). Adding a pattern file with
    PATTERN_TESTS / KERNEL_BENCHES is picked up here with no harness edits."""
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


# --- Cached mesh-device lifecycle --------------------------------------------
#
# `builder._execute` reuses one open mesh device across tests rather than
# opening + closing on every run (avoids hammering the n300 ARC startup; see
# `_get_cached_device`). Mirroring test/python/golden/conftest.py, we close that
# cached device at session end and after a failing test -- the hardware may be
# in an undefined state after a runtime failure, so the next test should open a
# fresh one.


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        _close_cached_device()


def pytest_sessionfinish(session, exitstatus):
    _close_cached_device()
