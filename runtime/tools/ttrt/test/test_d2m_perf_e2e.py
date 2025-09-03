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
BUILD_DIR = TT_MLIR_HOME / "build"
BIN_DIR = BUILD_DIR / "bin"
TTMLIR_OPT = BIN_DIR / "ttmlir-opt"
TTMLIR_TRANSLATE = BIN_DIR / "ttmlir-translate"

# Pick a simple TTM mlir that compiles and runs on silicon
SIMPLE_MLIR = TT_MLIR_HOME / "test/ttmlir/Silicon/TTMetal/n150/simple_add.mlir"

REQUIRED_COLUMNS = [
    "GLOBAL CALL COUNT",
    "OP CODE",
    "HOST DURATION [ns]",
    "DEVICE KERNEL DURATION [ns]",
]


def _which_or(path: str) -> str:
    return shutil.which(path) or path


def test_d2m_perf_e2e():
    # Preconditions: built tt-mlir with runtime and perf enabled, ttrt installed in PATH, and silicon present
    ttrt_exec = shutil.which("ttrt")
    if not ttrt_exec:
        pytest.skip("ttrt executable not found in PATH")

    if not TTMLIR_OPT.exists() or not TTMLIR_TRANSLATE.exists():
        pytest.skip("ttmlir tools not found; build tt-mlir before running this test")

    if not SIMPLE_MLIR.exists():
        pytest.skip(f"Missing MLIR test file: {SIMPLE_MLIR}")

    artifacts_dir = TT_MLIR_HOME / "ttrt-artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1) Query system desc
    query_cmd = f"{ttrt_exec} query --save-artifacts"
    sub_process_command(query_cmd)

    system_desc = artifacts_dir / "system_desc.ttsys"
    if not system_desc.exists():
        pytest.skip(
            "system_desc.ttsys was not generated (no silicon or runtime not enabled)"
        )

    # 2) Compile MLIR to TTM MLIR with system-desc
    ttm_mlir = TT_MLIR_HOME / "ttm_perf_e2e.mlir"
    opt_cmd = (
        f'{_which_or(str(TTMLIR_OPT))} --ttir-to-ttmetal-pipeline="system-desc-path={system_desc}" '
        f"{SIMPLE_MLIR} -o {ttm_mlir}"
    )
    sub_process_command(opt_cmd)
    assert ttm_mlir.exists(), "ttmlir-opt did not produce TTM mlir"

    # 3) Translate TTM MLIR to flatbuffer (.ttm)
    fb_path = TT_MLIR_HOME / "ttm_perf_e2e.ttm"
    trans_cmd = f"{_which_or(str(TTMLIR_TRANSLATE))} --ttmetal-to-flatbuffer {ttm_mlir} -o {fb_path}"
    sub_process_command(trans_cmd)
    assert fb_path.exists(), "ttmlir-translate did not produce flatbuffer"

    # 4) Run ttrt perf on the produced binary. Enable device profiling (host-only off) to produce device CSVs.
    perf_results = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args = {}
    custom_args["--result-file"] = perf_results
    custom_args["artifact-dir"] = artifacts_dir
    custom_args["binary"] = str(fb_path)
    API.initialize_apis()
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    # 5) Validate outputs: perf folder under artifacts/<binary_name>/perf should contain CSVs
    binary_name = fb_path.name
    perf_dir = artifacts_dir / binary_name / "perf"
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

    with ops_minus_const_eval.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in REQUIRED_COLUMNS:
            assert (
                col in reader.fieldnames
            ), f"Column {col} missing in ops_perf_results_minus_const_eval.csv"

    # 6) Validate perf result json saved
    assert os.path.exists(perf_results), "Missing perf result json"
    with open(perf_results, "r") as f:
        data = json.load(f)
        assert isinstance(data, list), "perf result json should be a list"
        assert all(
            d.get("result") == "pass" for d in data
        ), "Perf did not pass on all binaries"
