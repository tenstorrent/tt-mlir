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
