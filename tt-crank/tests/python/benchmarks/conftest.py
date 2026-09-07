# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark-suite-specific fixtures, result recording, and reporting hooks.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform as _platform
import socket
import subprocess
import sys
from collections.abc import Callable
from typing import Any

import pytest
import torch

from ._runner import BenchmarkResult, Measurement


_RESULTS_KEY = "_tt_kurbla_bench_results"


def _results_bucket(config: pytest.Config) -> list[BenchmarkResult]:
    """The session-wide result list, created on first use."""
    bucket: list[BenchmarkResult] = getattr(config, _RESULTS_KEY, [])
    setattr(config, _RESULTS_KEY, bucket)
    return bucket


class _TestReport:
    """One test's benchmark report: the device result, plus the optional
    --cpu-baseline row. One of each, enforced, so the compile stats folded in
    at teardown can't be misattributed; a test wanting several device results
    should be parametrized into several tests instead.
    """

    def __init__(self) -> None:
        self.device_result: BenchmarkResult | None = None
        self.cpu_baseline: BenchmarkResult | None = None

    def record(self, result: BenchmarkResult) -> None:
        if result.device == "cpu":
            assert (
                self.cpu_baseline is None
            ), "this test already recorded a cpu baseline"
            self.cpu_baseline = result
        else:
            assert (
                self.device_result is None
            ), "this test already recorded a device result"
            self.device_result = result

    def results(self) -> list[BenchmarkResult]:
        return [r for r in (self.device_result, self.cpu_baseline) if r is not None]


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
        "--warmup",
        action="store",
        type=int,
        default=3,
        help="Number of warmup iterations before timing starts.",
    )
    group.addoption(
        "--iters",
        action="store",
        type=int,
        default=20,
        help="Number of timed iterations per benchmark.",
    )
    group.addoption(
        "--strict-no-fallback",
        action="store_true",
        default=False,
        help="Enforce that every benchmark op runs natively on tt: any op that "
        "would route through the CPU fallback raises instead. Eager mode only.",
    )
    group.addoption(
        "--cpu-baseline",
        action="store_true",
        default=False,
        help="Also time each benchmark on CPU for a relative comparison row.",
    )
    group.addoption(
        "--accuracy",
        action="store_true",
        default=False,
        help="Build a CPU reference model and emit pcc_before_warmup / "
        "pcc_after_warmup measurements alongside the perf numbers.",
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
        default="./profile_data",
        help="Where --profiler writes its Chrome traces.",
    )
    group.addoption(
        "--benchmark-json",
        action="store",
        default="benchmark_results.json",
        help="Path to write the per-test benchmark JSON results to.",
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
    group.addoption(
        "--opt-level",
        action="store",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="tt-mlir optimizer level for compile mode (CompileOption.OPT_LEVEL): "
        "0 = optimizer off (bare TTIRToTTNN), 1 = optimizer on without sharding, "
        "2 = optimizer on with memory-layout/sharding analysis. "
        "When unset, each benchmark uses its own default.",
    )


@pytest.fixture(scope="session")
def mode(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--mode")


@pytest.fixture(scope="session")
def warmup(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--warmup"))


@pytest.fixture(scope="session")
def iters(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("--iters"))


@pytest.fixture(scope="session")
def cpu_baseline(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--cpu-baseline"))


@pytest.fixture(scope="session")
def accuracy(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--accuracy"))


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
    from tt_kurbla.torch.testing import strict_no_fallback

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
    from tt_kurbla.torch._artifacts import collect_artifacts

    report = _TestReport()
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


@pytest.fixture(scope="session")
def opt_level(request: pytest.FixtureRequest) -> int | None:
    value = request.config.getoption("--opt-level")
    return None if value is None else int(value)


@pytest.fixture
def record_bench(_test_report: _TestReport) -> Callable[[BenchmarkResult], None]:
    """Record a BenchmarkResult on this test's report, for terminal + JSON
    reporting once the report is finalized."""
    return _test_report.record


def _git_sha() -> str:
    # Prefer the sha the Actions runner provides.
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            # Pin to this file's repo; the pytest invocation dir may be outside it.
            cwd=os.path.dirname(__file__),
        ).strip()
    except Exception:
        return ""


def _device_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "arch": torch.tt.arch(),
        "num_chips": torch.tt.num_chips(),
        "platform": _platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
    }


def _collect(config: pytest.Config) -> list[BenchmarkResult]:
    return list(getattr(config, _RESULTS_KEY, []))


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    results = _collect(config)
    if not results:
        return
    tr = terminalreporter
    tr.write_sep("=", "benchmark results")
    tr.write_line(f"device arch: {torch.tt.arch()}, chips: {torch.tt.num_chips()}")
    tr.write_line("")
    for r in results:
        for line in r.format_card().split("\n"):
            tr.write_line(line)
        tr.write_line("")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    results = _collect(session.config)
    if not results:
        return
    out_path = session.config.getoption("--benchmark-json")
    payload = {
        "session": {
            "git_sha": _git_sha(),
            "simulator": os.environ.get("TT_KURBLA_USE_SIMULATOR") == "1",
            "mode": session.config.getoption("--mode"),
            "warmup": session.config.getoption("--warmup"),
            "iters": session.config.getoption("--iters"),
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "device_info": _device_info(),
        },
        "results": [r.as_dict() for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
