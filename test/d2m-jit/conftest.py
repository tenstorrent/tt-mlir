# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from d2m_jit._src.builder import _Builder


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
    bundled pattern files (d2m_jit/patterns/*.py). Adding a pattern file with
    PATTERN_TESTS / KERNEL_BENCHES is picked up here with no harness edits."""
    from d2m_jit.testing import discover

    pattern_tests, kernel_benches = discover()
    if "pattern_test" in metafunc.fixturenames:
        metafunc.parametrize(
            "pattern_test", pattern_tests, ids=[t.name for t in pattern_tests]
        )
    if "kernel_bench" in metafunc.fixturenames:
        metafunc.parametrize(
            "kernel_bench", kernel_benches, ids=[b.name for b in kernel_benches]
        )
