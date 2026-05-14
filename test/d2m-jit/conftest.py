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
