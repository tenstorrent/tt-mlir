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

from ._runner import BenchmarkResult


_RESULTS_KEY = "_tt_kurbla_bench_results"


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


@pytest.fixture(scope="session")
def llm_num_layers(request: pytest.FixtureRequest) -> int | None:
    value = request.config.getoption("--llm-num-layers")
    return None if value is None else int(value)


@pytest.fixture
def record_bench(request: pytest.FixtureRequest) -> Callable[[BenchmarkResult], None]:
    """Stash a BenchmarkResult on the session config for terminal + JSON reporting."""
    bucket: list[BenchmarkResult] = getattr(request.config, _RESULTS_KEY, [])
    setattr(request.config, _RESULTS_KEY, bucket)

    def _record(result: BenchmarkResult) -> None:
        bucket.append(result)

    return _record


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def _device_info() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
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
