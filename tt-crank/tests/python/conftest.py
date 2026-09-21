# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Top-level pytest conftest for tt-crank Python tests.

Only what every frontend's tests share lives here. Frontend-specific fixtures
sit in their own subdirectory conftest (torch/conftest.py, onnx/conftest.py).
"""

import os
import sys

# TT_CRANK_USE_SIMULATOR is read in a static initializer when libtt_crank.so is
# dlopened (see src/engine/sim_env.cpp) — whether via `import tt_crank.torch` or
# the ONNX Runtime plugin — so it must be set before that happens.
# pytest_addoption runs after conftest import, hence the sys.argv sniff here.
if "--sim" in sys.argv:
    os.environ["TT_CRANK_USE_SIMULATOR"] = "1"

import pytest  # noqa: E402

import bench  # noqa: E402
from bench import (  # noqa: E402, F401
    _test_report,
    accuracy,
    cpu_baseline,
    iters,
    opt_level,
    record_bench,
    warmup,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sim",
        action="store_true",
        default=False,
        help="Route the runtime through ttsim by setting TT_CRANK_USE_SIMULATOR=1.",
    )
    bench.add_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "benchmark: marks a test as a benchmark (deselect with '-m \"not benchmark\"').",
    )


def pytest_terminal_summary(
    terminalreporter, exitstatus: int, config: pytest.Config
) -> None:
    bench.terminal_summary(terminalreporter, config)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    bench.session_finish(session)
