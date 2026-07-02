# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""D2M-JIT pattern perf benchmarks: d2m (ttmetal) vs ttnn kernel duration.

Auto-discovers ``PatternTest`` entries tagged ``"perf"`` from all pattern
files under ``d2m_jit/patterns/``. Each bench compiles the same TTIR
through both the d2m pattern-rewrite pipeline and the standard TTIR→TTNN
pipeline, runs both on device with the profiler, and records kernel
durations.

Run:
    pytest test/d2m-jit/perf_ci/perf_tests.py -v
"""

import sys

import pytest


def _discover_perf_specs():
    from d2m_jit.testing import discover

    pattern_tests, _kb = discover()
    return [t for t in pattern_tests if "perf" in t.tags]


_PERF_SPECS = _discover_perf_specs()


@pytest.mark.parametrize(
    "spec",
    _PERF_SPECS,
    ids=[s.name for s in _PERF_SPECS],
)
def test_d2m_vs_ttnn(spec, perf_runner):
    """Run d2m and ttnn paths, record kernel duration for dashboard."""
    d2m_ns, ttnn_ns = perf_runner(spec)

    ratio = round(ttnn_ns / d2m_ns, 4) if d2m_ns > 0 else float("inf")
    print(
        f"\n  {spec.name}: d2m={d2m_ns}ns  ttnn={ttnn_ns}ns  "
        f"ratio(ttnn/d2m)={ratio}",
        file=sys.stderr,
    )

    assert d2m_ns > 0, f"d2m profiler returned 0 for {spec.name}"
    assert ttnn_ns > 0, f"ttnn profiler returned 0 for {spec.name}"
