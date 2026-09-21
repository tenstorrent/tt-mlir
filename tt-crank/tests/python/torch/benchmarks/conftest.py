# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Torch benchmark fixtures: the torch-only options, and the per-test report that
folds compile stats in. Shared plumbing lives in tests/python/bench.py.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

from bench import Measurement, TestReport, _results_bucket


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("benchmark", "Benchmark suite options")
    group.addoption(
        "--mode",
        action="store",
        default="compile",
        choices=["eager", "compile"],
        help="Model execution mode. 'compile' wraps the model with "
        "torch.compile(backend='tt'); 'eager' runs the model as-is.",
    )
    group.addoption(
        "--strict-no-fallback",
        action="store_true",
        default=False,
        help="Enforce that every benchmark op runs natively on tt: any op that "
        "would route through the CPU fallback raises instead. Eager mode only.",
    )
    group.addoption(
        "--profiler",
        action="store_true",
        default=False,
        help="Capture a torch.profiler Chrome trace per benchmark "
        "(one JSON file per test).",
    )
    group.addoption(
        "--profile-dir",
        action="store",
        default=".data/profile_data",
        help="Where --profiler writes its Chrome traces.",
    )
    group.addoption(
        "--llm-batch-size",
        action="store",
        type=int,
        default=32,
        help="Batch size for the LLM decode benchmark.",
    )
    group.addoption(
        "--llm-max-output-tokens",
        action="store",
        type=int,
        default=None,
        help="Override the generate-step count for the LLM decode benchmark. "
        "Default fills the 128-slot KV cache; use a small value for smoke runs.",
    )
    group.addoption(
        "--llm-num-layers",
        action="store",
        type=int,
        default=None,
        help="Truncate the model to this many decoder layers. Default keeps the full model.",
    )


@pytest.fixture(scope="session")
def mode(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--mode")


@pytest.fixture(scope="session")
def profile_enabled(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--profiler"))


@pytest.fixture(scope="session")
def profile_dir(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--profile-dir"))


@pytest.fixture(scope="session")
def llm_batch_size(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--llm-batch-size"))


@pytest.fixture(scope="session")
def llm_max_output_tokens(request: pytest.FixtureRequest) -> int | None:
    value = request.config.getoption("--llm-max-output-tokens")
    return None if value is None else int(value)


@pytest.fixture(autouse=True)
def _strict_no_fallback(request: pytest.FixtureRequest) -> Any:
    """When --strict-no-fallback is set, run each benchmark with the CPU
    fallback flipped to raise, so any op that isn't natively implemented on tt
    fails the benchmark instead of silently running on host.
    """
    if not request.config.getoption("--strict-no-fallback"):
        yield
        return
    from tt_crank.torch.testing import strict_no_fallback

    with strict_no_fallback():
        yield


@pytest.fixture(autouse=True)
def _test_report(request: pytest.FixtureRequest) -> Any:
    """Per-test benchmark report, wrapped in an artifacts collection named after
    the test (param id included, so parametrized runs don't collide).

    On teardown, folds the collection's compile stats (total engine compile
    time, graph and cache-hit counts) into the results the test recorded, then
    flushes them into the session-wide list the reporting hooks read.
    """
    from tt_crank.torch._artifacts import collect_artifacts

    report = TestReport()
    try:
        # A failing test still dumps its artifacts on exit from the `with`
        # block, so the IR collected up to the failure survives for debugging.
        with collect_artifacts(request.node.name) as collection:
            yield report
            stats = collection.compile_stats()

        if stats.num_graphs > 0 and report.device_result is not None:
            report.device_result.measurements.extend(
                [
                    Measurement("compile_total_ms", stats.total_duration_ms, "ms"),
                    Measurement("num_graphs", stats.num_graphs, "count"),
                    Measurement("num_cache_hits", stats.num_cache_hits, "count"),
                ]
            )
    finally:
        # Flush even when the artifacts dump throws: the results themselves
        # are fine, and dropping them would hide a finished measurement.
        _results_bucket(request.config).extend(report.results())


@pytest.fixture(scope="session")
def llm_num_layers(request: pytest.FixtureRequest) -> int | None:
    value = request.config.getoption("--llm-num-layers")
    return None if value is None else int(value)
