# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark plumbing shared by the torch and onnx suites: result types, the
per-test report, the common CLI options and fixtures, and the reporting hooks.
Frontend-specific pieces (torch's --mode, profiler and LLM flags) stay in the
suite's own benchmarks/conftest.py.
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
from dataclasses import asdict, dataclass, field
from typing import Any

import pytest


@dataclass
class Measurement:
    name: str
    value: float
    unit: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    label: str
    mode: str
    device: str
    warmup: int
    iters: int
    measurements: list[Measurement] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": self.mode,
            "device": self.device,
            "warmup": self.warmup,
            "iters": self.iters,
            "measurements": [m.as_dict() for m in self.measurements],
        }

    def format_card(self) -> str:
        """Render this result as a multi-line card for terminal output."""
        header = (
            f"{self.label}  "
            f"[mode={self.mode}, device={self.device}, "
            f"warmup={self.warmup}, iters={self.iters}]"
        )
        rows = [header]
        for m in self.measurements:
            value = f"{m.value:>12.0f}" if m.unit == "count" else f"{m.value:>12.3f}"
            rows.append(f"  {m.name:<24} {value} {m.unit}")
        return "\n".join(rows)


_RESULTS_KEY = "_tt_crank_bench_results"


def _results_bucket(config: pytest.Config) -> list[BenchmarkResult]:
    """The session-wide result list, created on first use."""
    bucket: list[BenchmarkResult] = getattr(config, _RESULTS_KEY, [])
    setattr(config, _RESULTS_KEY, bucket)
    return bucket


class TestReport:
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


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("benchmark", "Benchmark suite options")
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
        "--benchmark-json",
        action="store",
        default=".data/benchmark_results.json",
        help="Path to write the per-test benchmark JSON results to.",
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
def opt_level(request: pytest.FixtureRequest) -> int | None:
    value = request.config.getoption("--opt-level")
    return None if value is None else int(value)


@pytest.fixture(autouse=True)
def _test_report(request: pytest.FixtureRequest) -> Any:
    """Per-test benchmark report, flushed into the session-wide list at teardown."""
    report = TestReport()
    try:
        yield report
    finally:
        _results_bucket(request.config).extend(report.results())


@pytest.fixture
def record_bench(_test_report: TestReport) -> Callable[[BenchmarkResult], None]:
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
    import torch  # tt_crank.torch backs the device queries; imported lazily for onnx-only runs
    import tt_crank.torch  # noqa: F401

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


def terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    results = _collect(config)
    if not results:
        return
    tr = terminalreporter
    tr.write_sep("=", "benchmark results")
    info = _device_info()
    tr.write_line(f"device arch: {info['arch']}, chips: {info['num_chips']}")
    tr.write_line("")
    for r in results:
        for line in r.format_card().split("\n"):
            tr.write_line(line)
        tr.write_line("")


def session_finish(session: pytest.Session) -> None:
    results = _collect(session.config)
    if not results:
        return
    out_path = session.config.getoption("--benchmark-json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "session": {
            "git_sha": _git_sha(),
            "simulator": os.environ.get("TT_CRANK_USE_SIMULATOR") == "1",
            "mode": session.config.getoption("--mode", None),
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
