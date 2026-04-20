# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import pytest
import ttnn

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

_perf_results: list[dict] = []


def _serialize_fidelity(val):
    """Serialize math_fidelity to a string, handling both enums and strings."""
    if hasattr(val, "name"):
        return val.name
    return str(val)


def _flush_results():
    """Write the current results to disk so they survive a later crash."""
    profiler_dir = os.environ.get("TT_METAL_PROFILER_DIR")
    if not profiler_dir or not _perf_results:
        return
    os.makedirs(profiler_dir, exist_ok=True)
    results_path = os.path.join(profiler_dir, "perf_results.json")
    with open(results_path, "w") as f:
        json.dump(_perf_results, f, indent=2)


@pytest.fixture
def perf_device(request):
    """Open a device, yield it, then read profiler data and close.

    After the test body runs, we synchronize + read the device profiler so
    ``get_latest_programs_perf_data`` returns only the ops from *this* test.
    Requires TT_METAL_DEVICE_PROFILER=1, TT_METAL_PROFILER_MID_RUN_DUMP=1,
    and TT_METAL_PROFILER_CPP_POST_PROCESS=1.
    """
    device = ttnn.open_device(device_id=0)
    yield device

    total_duration_ns = 0
    num_programs = 0
    try:
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        data = ttnn.get_latest_programs_perf_data()
        for programs in data.values():
            for program in programs:
                num_programs += 1
                result = program.program_analyses_results.get(_DURATION_KEY)
                if result:
                    total_duration_ns += result.duration
    except Exception as exc:
        print(
            f"Warning: profiler read failed for {request.node.nodeid}: {exc}",
            file=sys.stderr,
        )

    params = getattr(request.node, "callspec", None)
    p = params.params if params else {}
    op_func = p.get("op")
    op_name = getattr(op_func, "__name__", str(op_func)) if op_func else "unknown"
    ttnn_dtype = p.get("ttnn_dtype")
    dtype_str = ttnn_dtype.name if ttnn_dtype is not None else ""

    mem_cfg_id = p.get("memory_config_id", "")
    _perf_results.append(
        {
            "test_node_id": request.node.nodeid,
            "jit": bool(p.get("jit_enabled", False)),
            "op": op_name,
            "dtype": dtype_str,
            "memory_config_id": mem_cfg_id,
            "input_a_mem": p.get("input_a_mem", mem_cfg_id),
            "input_b_mem": p.get("input_b_mem", mem_cfg_id),
            "math_fidelity": _serialize_fidelity(p.get("math_fidelity", "HiFi4")),
            "h": p.get("h", 0),
            "w": p.get("w", 0),
            "m": p.get("m", 0),
            "k": p.get("k", 0),
            "n": p.get("n", 0),
            "duration_ns": total_duration_ns,
            "num_programs": num_programs,
        }
    )
    _flush_results()

    try:
        ttnn.close_device(device)
    except Exception as exc:
        print(
            f"Warning: close_device failed for {request.node.nodeid}: {exc}",
            file=sys.stderr,
        )


def pytest_sessionfinish(session, exitstatus):
    _flush_results()
