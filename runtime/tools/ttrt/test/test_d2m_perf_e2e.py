# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import shutil
import pathlib
import json

from ttrt.common.util import *
from ttrt.common.api import API

from util import *

TT_MLIR_HOME = pathlib.Path(os.environ.get("TT_MLIR_HOME", os.getcwd())).resolve()
BUILD_DIR = pathlib.Path(TT_MLIR_HOME) / "build"
FB_PATH = BUILD_DIR / "test/ttmlir/Silicon/TTMetal/n150/Output/simple_add.mlir.tmp.ttm"

REQUIRED_COLUMNS = [
    "GLOBAL CALL COUNT",
    "OP CODE",
    "HOST DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
]


def test_d2m_perf_e2e():
    # Preconditions: built tt-mlir with runtime and perf enabled, ttrt installed in PATH, and silicon present
    ttrt_exec = shutil.which("ttrt")
    if not ttrt_exec:
        pytest.skip("ttrt executable not found in PATH")

    if not FB_PATH.exists():
        pytest.skip(f"Missing flatbuffer file: {FB_PATH}")

    # Run ttrt perf on the produced binary. Enable device profiling to produce device CSVs.
    perf_results = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args = {}
    custom_args["--result-file"] = perf_results
    custom_args["binary"] = str(FB_PATH)
    API.initialize_apis()
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    # Validate outputs: perf folder under artifacts/<binary_name>/perf should contain CSVs
    binary_name = FB_PATH.name
    perf_dir = TT_MLIR_HOME / "ttrt-artifacts" / binary_name / "perf"
    assert perf_dir.exists(), f"Missing perf directory: {perf_dir}"

    # Tracy CSVs
    tracy_times = perf_dir / "tracy_ops_times.csv"
    tracy_data = perf_dir / "tracy_ops_data.csv"
    assert tracy_times.exists(), "Missing tracy_ops_times.csv"
    assert tracy_data.exists(), "Missing tracy_ops_data.csv"

    # Profiler CSV produced by post-processing, copied into perf folder
    # The main file lives under tt-metal generated/profiler/reports, but ttrt copies into perf dir for the binary
    # The expected filename is ops_perf_results.csv and ops_perf_results_minus_const_eval.csv
    ops_csv = perf_dir / "ops_perf_results.csv"
    ops_minus_const_eval = perf_dir / "ops_perf_results_minus_const_eval.csv"
    assert ops_csv.exists(), "Missing ops_perf_results.csv"
    assert (
        ops_minus_const_eval.exists()
    ), "Missing ops_perf_results_minus_const_eval.csv"

    # Basic schema checks
    import csv

    with ops_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in REQUIRED_COLUMNS:
            assert (
                col in reader.fieldnames
            ), f"Column {col} missing in ops_perf_results.csv"
        # Also ensure at least one row exists
        first = next(reader, None)
        assert first is not None, "ops_perf_results.csv has no result rows"

    # Validate perf result json saved
    assert os.path.exists(perf_results), "Missing perf result json"
    with open(perf_results, "r") as f:
        data = json.load(f)
        assert isinstance(data, list), "perf result json should be a list"
        assert all(
            d.get("result") == "pass" for d in data
        ), "Perf did not pass on all binaries"
