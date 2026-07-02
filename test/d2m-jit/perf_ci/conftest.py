# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Perf collection fixtures for d2m-jit pattern benchmarks.

Accumulates per-test kernel duration results into ``perf_results.json``
(written to ``$TT_METAL_PROFILER_DIR``). Each test runs both the d2m
(ttmetal) and ttnn compilation paths on device with the profiler enabled.

Requires ``TT_METAL_DEVICE_PROFILER=1``,
``TT_METAL_PROFILER_MID_RUN_DUMP=1``, and
``TT_METAL_PROFILER_CPP_POST_PROCESS=1``.
"""

import json
import os
import re
import sys

import pytest

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

_perf_results: list[dict] = []


def _flush_results():
    profiler_dir = os.environ.get("TT_METAL_PROFILER_DIR")
    if not profiler_dir or not _perf_results:
        return
    os.makedirs(profiler_dir, exist_ok=True)
    results_path = os.path.join(profiler_dir, "perf_results.json")
    with open(results_path, "w") as f:
        json.dump(_perf_results, f, indent=2)


def _read_profiler_duration(runtime, device):
    """Trigger profiler read and sum kernel durations across all programs."""
    total_ns = 0
    num_programs = 0
    try:
        import ttnn

        runtime.read_device_profiler_results(device)
        data = ttnn.get_all_programs_perf_data()
        for programs in data.values():
            for program in programs:
                num_programs += 1
                result = program.program_analyses_results.get(_DURATION_KEY)
                if result:
                    total_ns += result.duration
    except Exception as exc:
        print(f"Warning: profiler read failed: {exc}", file=sys.stderr)
    return total_ns, num_programs


def _run_and_profile(runtime, fbb, inputs):
    """Open a device, execute the fbb, read profiler, close. Returns duration_ns."""
    from d2m_jit.testing import execute_ttm_in_process

    runtime.set_compatible_device_runtime(fbb)
    opts = runtime.MeshDeviceOptions()
    opts.mesh_shape = fbb.get_program_mesh_shape(0)
    device = runtime.open_mesh_device(opts)
    try:
        execute_ttm_in_process(fbb, inputs, device)
        duration_ns, num_programs = _read_profiler_duration(runtime, device)
    finally:
        runtime.close_mesh_device(device)
    return duration_ns, num_programs


def _extract_shape_label(ttir_text):
    """Extract shape string from TTIR func signature for reporting."""
    match = re.search(r"tensor<(\d+x\d+(?:x\d+)*)x\w+>", ttir_text)
    return match.group(1) if match else "unknown"


def _extract_dtype_label(ttir_text):
    """Extract element type from TTIR func signature for reporting."""
    match = re.search(r"tensor<[\dx]+x(\w+)>", ttir_text)
    if not match:
        return "unknown"
    mlir_to_label = {"f32": "FLOAT32", "bf16": "BFLOAT16", "f16": "FLOAT16"}
    return mlir_to_label.get(match.group(1), match.group(1))


@pytest.fixture
def perf_runner():
    """Fixture that provides the run-and-profile function.

    Each test calls ``perf_runner(spec)`` which compiles both d2m and ttnn
    flatbuffers, runs each on device with profiler, and records results."""
    from _ttmlir_runtime import runtime

    def run(spec):
        from d2m_jit.testing import (
            compile_perf_d2m_fbb,
            compile_perf_ttnn_fbb,
            perf_inputs,
        )

        inputs = perf_inputs(spec)
        d2m_fbb = compile_perf_d2m_fbb(spec)
        ttnn_fbb = compile_perf_ttnn_fbb(spec)

        d2m_ns, d2m_progs = _run_and_profile(runtime, d2m_fbb, inputs)
        ttnn_ns, ttnn_progs = _run_and_profile(runtime, ttnn_fbb, inputs)

        shape_label = _extract_shape_label(spec.ttir)
        dtype_label = _extract_dtype_label(spec.ttir)

        _perf_results.append(
            {
                "test_name": spec.name,
                "pattern": spec.name,
                "shape": shape_label,
                "dtype": dtype_label,
                "d2m_duration_ns": d2m_ns,
                "d2m_num_programs": d2m_progs,
                "ttnn_duration_ns": ttnn_ns,
                "ttnn_num_programs": ttnn_progs,
            }
        )
        _flush_results()
        return d2m_ns, ttnn_ns

    return run


def pytest_sessionfinish(session, exitstatus):
    _flush_results()
