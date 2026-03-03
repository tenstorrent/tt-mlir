# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test module for compiling and executing MLIR model snippets.
# Discovers all MLIR files in the mlir_snippets/models directory,
# compiles them for the TTMetal backend, and executes them.
# Each function in an MLIR file becomes its own test case.

import pytest
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb
from builder.base.builder_utils import get_artifact_dir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")

PERF_EXEC_ITERATIONS = max(
    1, int(os.environ.get("TT_MLIR_SNIPPET_PERF_EXEC_ITERS", "1"))
)
PERF_WARMUP_ITERATIONS = max(
    0, int(os.environ.get("TT_MLIR_SNIPPET_PERF_WARMUP_ITERS", "0"))
)
PERF_TIMINGS: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
PERF_SUMMARY_SORT_MODE = "snippet"
PERF_SUMMARY_KEYWORD = ""
PERF_SUMMARY_METRIC = "exec"
PERF_MARK_TTMETAL_FASTER = False
PERF_SHOW_EXEC_TIMES = False
PERF_TIME_UNIT = "ms"
LOG_WIDTH = 92
SNIPPET_COL_WIDTH = 44
DELTA_RATIO_COL_WIDTH = 27
EXEC_TIMES_COL_WIDTH = 33
TIME_VALUE_WIDTH = 16
SORT_METRICS = {"compile", "exec"}

# Extract individual functions from an MLIR module.
def extract_functions_from_mlir(mlir_content: str) -> List[Tuple[str, str, int]]:
    functions = []

    # Find func.func declarations and extract with brace matching
    func_start_pattern = re.compile(r"func\.func\s+@(\w+)\s*\(")

    for match in func_start_pattern.finditer(mlir_content):
        func_name = match.group(1)
        start_pos = match.start()

        # Find the opening brace of the function body
        brace_pos = mlir_content.find("{", match.end())
        if brace_pos == -1:
            continue

        # Count braces to find matching closing brace
        brace_count = 1
        pos = brace_pos + 1
        while pos < len(mlir_content) and brace_count > 0:
            if mlir_content[pos] == "{":
                brace_count += 1
            elif mlir_content[pos] == "}":
                brace_count -= 1
            pos += 1

        if brace_count == 0:
            func_body = mlir_content[start_pos:pos]
            # Wrap each function in its own module
            func_mlir = f"module {{\n  {func_body}\n}}"
            line_number = mlir_content.count("\n", 0, start_pos) + 1
            functions.append((func_name, func_mlir, line_number))

    return functions


# Model IDs (path under mlir_snippets/models without .mlir) to exclude from
# discovery here; they are tested in test_d2m_fusion_with_optimizer.py instead.
SNIPPETS_TO_SKIP = {
    "gpt_oss_20b/gate_up",
    "gpt_oss_20b/rope_embedding",
}


# Discover all MLIR files and extract each function as a separate snippet.
def discover_model_mlir_snippets() -> Dict[str, Dict[str, Any]]:
    models_dir = os.path.join(os.path.dirname(__file__), "mlir_snippets/models")
    snippets = {}

    if not os.path.exists(models_dir):
        return snippets

    for root, _, files in os.walk(models_dir):
        for filename in sorted(files):
            if filename.endswith(".mlir"):
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, models_dir)
                model_id = rel_path.replace(".mlir", "")
                if model_id in SNIPPETS_TO_SKIP:
                    continue
                with open(file_path, "r") as f:
                    content = f.read().strip()

                if content:
                    functions = extract_functions_from_mlir(content)
                    for func_name, func_mlir, line_number in functions:
                        snippet_id = f"{model_id}/{func_name}"
                        snippets[snippet_id] = {
                            "path": file_path,
                            "content": func_mlir,
                            "func_name": func_name,
                            "model_id": model_id,
                            "line_number": line_number,
                        }

    return snippets


MODEL_MLIR_SNIPPETS = discover_model_mlir_snippets()


def get_snippet_ids() -> List[str]:
    return list(MODEL_MLIR_SNIPPETS.keys())


def _format_time(seconds: float) -> str:
    scale = 1000.0 if PERF_TIME_UNIT == "ms" else 1_000_000.0
    suffix = "ms" if PERF_TIME_UNIT == "ms" else "us"
    return f"{seconds * scale:.3f} {suffix}"


def _ns_to_s(ns: int) -> float:
    return ns / 1_000_000_000.0


def _format_signed_time(seconds: float) -> str:
    sign = "+" if seconds >= 0 else "-"
    scale = 1000.0 if PERF_TIME_UNIT == "ms" else 1_000_000.0
    suffix = "ms" if PERF_TIME_UNIT == "ms" else "us"
    return f"{sign}{abs(seconds) * scale:.3f} {suffix}"


def _format_delta_ratio_cell(delta_s: float, ratio: float) -> str:
    return f"{_format_signed_time(delta_s):>{TIME_VALUE_WIDTH}} / {ratio:>7.3f}x"


def _format_exec_times_cell(ttmetal_s: float, ttnn_s: float) -> str:
    return (
        f"{_format_time(ttmetal_s):>{TIME_VALUE_WIDTH}} / "
        f"{_format_time(ttnn_s):>{TIME_VALUE_WIDTH}}"
    )


def _target_delta_and_ratio(ttmetal_s: float, ttnn_s: float) -> Tuple[float, float]:
    delta = ttmetal_s - ttnn_s
    ratio = (ttmetal_s / ttnn_s) if ttnn_s > 0 else float("inf")
    return delta, ratio


def _format_ratio_description(ratio: float) -> str:
    if ratio == float("inf"):
        return "infx"
    if ratio >= 1.0:
        return f"{ratio:.3f}x slower"
    return f"{(1.0 / ratio):.3f}x faster"


def _parse_sort_spec(sort_spec: str) -> Tuple[str, str, str]:
    normalized = sort_spec.strip()
    if not normalized:
        return "snippet", "", "exec"
    if normalized == "snippet":
        return "snippet", "", "exec"

    # Allow metric override for time/ratio: time:exec, ratio=compile, etc.
    for mode in ("time", "ratio"):
        if normalized == mode:
            return mode, "", "exec"
        if normalized.startswith(f"{mode}:") or normalized.startswith(f"{mode}="):
            metric = (
                normalized.split(":", 1)[1]
                if ":" in normalized
                else normalized.split("=", 1)[1]
            )
            metric = metric.strip().lower()
            if metric not in SORT_METRICS:
                metric = "exec"
            return mode, "", metric

    if normalized == "keyword":
        return "keyword", "", "exec"
    return "snippet", "", "exec"


def _get_source_ref(snippet_id: str) -> str:
    source_info = MODEL_MLIR_SNIPPETS.get(snippet_id, {})
    if source_info:
        return f"{source_info['path']}:{source_info['line_number']}"
    return "unknown"


def _sort_single_target_rows(rows: List[Dict[str, object]]) -> None:
    if PERF_SUMMARY_SORT_MODE == "time":
        key_name = "exec_s" if PERF_SUMMARY_METRIC == "exec" else "compile_s"
        rows.sort(key=lambda row: row[key_name], reverse=True)
    else:
        rows.sort(key=lambda row: row["snippet_id"])


def _sort_comparison_rows(rows: List[Dict[str, object]]) -> None:
    if PERF_SUMMARY_SORT_MODE == "time":
        if PERF_SUMMARY_METRIC == "exec":
            rows.sort(
                key=lambda row: max(row["ttmetal_exec_s"], row["ttnn_exec_s"]),
                reverse=True,
            )
        else:
            rows.sort(
                key=lambda row: max(row["ttmetal_compile_s"], row["ttnn_compile_s"]),
                reverse=True,
            )
    elif PERF_SUMMARY_SORT_MODE == "ratio":
        ratio_key = "exec_ratio" if PERF_SUMMARY_METRIC == "exec" else "compile_ratio"
        rows.sort(key=lambda row: row[ratio_key], reverse=True)
    else:
        rows.sort(key=lambda row: row["snippet_id"])


def _print_run_header(
    snippet_id: str,
    model_id: str,
    func_name: str,
    target: str,
    mlir_content: str,
) -> None:
    print("\n" + "=" * LOG_WIDTH)
    print("Snippet benchmark")
    print("-" * LOG_WIDTH)
    print(f"snippet : {snippet_id}")
    print(f"model   : {model_id}")
    print(f"func    : {func_name}")
    print(f"target  : {target}")
    print(f"iters   : exec={PERF_EXEC_ITERATIONS}, warmup={PERF_WARMUP_ITERATIONS}")
    print("-" * LOG_WIDTH)
    print("MLIR:")
    print(mlir_content)
    print("=" * LOG_WIDTH)


def _print_perf_summary() -> None:
    if not PERF_TIMINGS:
        return

    rows = []
    single_target_rows = []
    targets_seen = set()
    for snippet_id in sorted(PERF_TIMINGS.keys()):
        timing_entry = PERF_TIMINGS[snippet_id]
        targets_seen.update(timing_entry.keys())
        if "ttmetal" not in timing_entry or "ttnn" not in timing_entry:
            if len(timing_entry) == 1:
                target = next(iter(timing_entry.keys()))
                metrics = timing_entry[target]
                single_target_rows.append(
                    {
                        "snippet_id": snippet_id,
                        "source_ref": _get_source_ref(snippet_id),
                        "compile_s": metrics["compile_s"],
                        "exec_s": metrics["exec_s"],
                    }
                )
            continue

        ttmetal = timing_entry["ttmetal"]
        ttnn = timing_entry["ttnn"]

        compile_delta, compile_ratio = _target_delta_and_ratio(
            ttmetal["compile_s"], ttnn["compile_s"]
        )
        exec_delta, exec_ratio = _target_delta_and_ratio(
            ttmetal["exec_s"], ttnn["exec_s"]
        )
        source_ref = _get_source_ref(snippet_id)

        rows.append(
            {
                "snippet_id": snippet_id,
                "source_ref": source_ref,
                "compile_delta": compile_delta,
                "compile_ratio": compile_ratio,
                "exec_delta": exec_delta,
                "exec_ratio": exec_ratio,
                "ttmetal_compile_s": ttmetal["compile_s"],
                "ttnn_compile_s": ttnn["compile_s"],
                "ttmetal_exec_s": ttmetal["exec_s"],
                "ttnn_exec_s": ttnn["exec_s"],
            }
        )

    _sort_comparison_rows(rows)

    print(f"\n{'='*LOG_WIDTH}")
    print(f"Exec iterations: {PERF_EXEC_ITERATIONS}, warmup: {PERF_WARMUP_ITERATIONS}")
    print(f"Time unit: {PERF_TIME_UNIT}")
    sort_label = PERF_SUMMARY_SORT_MODE
    if PERF_SUMMARY_SORT_MODE in {"time", "ratio"}:
        sort_label += f" ({PERF_SUMMARY_METRIC})"
    if PERF_SUMMARY_SORT_MODE == "keyword":
        sort_label += f" ({PERF_SUMMARY_KEYWORD or 'no keyword provided'})"
    print(f"Sort: {sort_label}")
    print(f"{'='*LOG_WIDTH}")
    if rows:
        print("Snippet perf summary (ttmetal vs ttnn)")
        print(f"{'='*LOG_WIDTH}")
        header_parts = [
            f"{'snippet':<{SNIPPET_COL_WIDTH}}",
            f"{'exec (delta/ratio)':>{DELTA_RATIO_COL_WIDTH}}",
        ]
        if PERF_SHOW_EXEC_TIMES:
            header_parts.append(f"{'exec (ttmetal/ttnn)':>{EXEC_TIMES_COL_WIDTH}}")
        header_parts.append(f"{'compile (delta/ratio)':>{DELTA_RATIO_COL_WIDTH}}")
        header_parts.append("source")
        header = "  ".join(header_parts)
        print(header)
        separator = "-" * len(header)
        print(separator)

        # A negative delta means ttmetal < ttnn for that metric (ttmetal faster).
        delta_key = "exec_delta" if PERF_SUMMARY_METRIC == "exec" else "compile_delta"
        faster_marker_printed = False

        for row in rows:
            if (
                PERF_MARK_TTMETAL_FASTER
                and not faster_marker_printed
                and row[delta_key] < 0
            ):
                print(separator)
                print(f"--- ttmetal faster below ({PERF_SUMMARY_METRIC} delta < 0) ---")
                print(separator)
                faster_marker_printed = True

            row_parts = [
                f"{row['snippet_id'][:SNIPPET_COL_WIDTH]:<{SNIPPET_COL_WIDTH}}",
                f"{_format_delta_ratio_cell(row['exec_delta'], row['exec_ratio']):<{DELTA_RATIO_COL_WIDTH}}",
            ]
            if PERF_SHOW_EXEC_TIMES:
                row_parts.append(
                    f"{_format_exec_times_cell(row['ttmetal_exec_s'], row['ttnn_exec_s']):<{EXEC_TIMES_COL_WIDTH}}"
                )
            row_parts.append(
                f"{_format_delta_ratio_cell(row['compile_delta'], row['compile_ratio']):<{DELTA_RATIO_COL_WIDTH}}"
            )
            row_parts.append(f"{row['source_ref']}")
            row_text = "  ".join(row_parts)
            print(row_text)
    else:
        _sort_single_target_rows(single_target_rows)
        target_name = next(iter(targets_seen), "target")
        print(f"Snippet perf summary ({target_name} only)")
        print(f"{'='*LOG_WIDTH}")
        header = "  ".join(
            [
                f"{'snippet':<{SNIPPET_COL_WIDTH}}",
                f"{'exec':>{TIME_VALUE_WIDTH}}",
                f"{'compile':>{TIME_VALUE_WIDTH}}",
                "source",
            ]
        )
        print(header)
        print("-" * len(header))
        for row in single_target_rows:
            print(
                "  ".join(
                    [
                        f"{row['snippet_id'][:SNIPPET_COL_WIDTH]:<{SNIPPET_COL_WIDTH}}",
                        f"{_format_time(row['exec_s']):>{TIME_VALUE_WIDTH}}",
                        f"{_format_time(row['compile_s']):>{TIME_VALUE_WIDTH}}",
                        f"{row['source_ref']}",
                    ]
                )
            )

    print(f"{'='*LOG_WIDTH}\n")


@pytest.fixture(scope="session", autouse=True)
def snippet_perf_summary(pytestconfig):
    global PERF_SUMMARY_SORT_MODE, PERF_SUMMARY_KEYWORD, PERF_SUMMARY_METRIC
    global PERF_MARK_TTMETAL_FASTER, PERF_SHOW_EXEC_TIMES, PERF_TIME_UNIT
    sort_spec = pytestconfig.getoption("--sort")
    if not sort_spec:
        sort_spec = os.environ.get("TT_MLIR_SNIPPET_PERF_SORT", "snippet")
    (
        PERF_SUMMARY_SORT_MODE,
        PERF_SUMMARY_KEYWORD,
        PERF_SUMMARY_METRIC,
    ) = _parse_sort_spec(sort_spec)
    PERF_MARK_TTMETAL_FASTER = pytestconfig.getoption("--mark-ttmetal-faster")
    PERF_SHOW_EXEC_TIMES = pytestconfig.getoption("--show-exec-times")
    PERF_TIME_UNIT = pytestconfig.getoption("--time-unit")
    yield
    _print_perf_summary()


@pytest.mark.parametrize("snippet_id", get_snippet_ids())
@pytest.mark.parametrize("target", ["ttmetal", "ttnn"])
def test_model_snippet_compile_execute(
    snippet_id: str,
    target: str,
    request,
    device,
):
    # Test that compiles and executes a single MLIR function snippet.
    kwargs = get_request_kwargs(request)
    system_desc_path = kwargs.get(
        "system_desc_path", "ttrt-artifacts/system_desc.ttsys"
    )
    output_root = kwargs.get("output_root", ".")
    save_artifacts = kwargs.get("save_artifacts", False)

    print_ir = kwargs.get("print_ir", False)
    skip_exec = kwargs.get("skip_exec", False)

    snippet_info = MODEL_MLIR_SNIPPETS[snippet_id]
    mlir_content = snippet_info["content"]
    func_name = snippet_info["func_name"]
    model_id = snippet_info["model_id"]

    artifact_dir = get_artifact_dir(
        output_root, f"model_snippets/{snippet_id}", target, save_artifacts
    )

    _print_run_header(snippet_id, model_id, func_name, target, mlir_content)

    # Compile
    compile_start = time.perf_counter()
    module, builder = load_mlir_file(mlir_content, target="ttir")
    (
        compiled_bin,
        input_output_goldens,
        intermediate_goldens,
    ) = compile_ttir_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc_path,
        artifact_dir=artifact_dir,
        target=target,
        save_artifacts=save_artifacts,
        print_ir=print_ir,
    )
    compile_s = time.perf_counter() - compile_start
    print("Compile:")
    print("  - status : success")
    print(f"  - time   : {_format_time(compile_s)}")

    exec_s = 0.0

    if skip_exec:
        print("Execute:")
        print("  - status : skipped (--skip-exec)")
        PERF_TIMINGS[snippet_id][target] = {
            "compile_s": compile_s,
            "exec_s": 0.0,
            "total_s": compile_s,
        }
        return

    # Optional warmup to reduce one-time effects before measured iterations.
    for _ in range(PERF_WARMUP_ITERATIONS):
        execute_fb(
            compiled_bin,
            input_output_goldens=input_output_goldens,
            intermediate_goldens=intermediate_goldens,
            device=device,
        )

    # Execute using the standard runtime helper and average over iterations.
    exec_s_acc = 0.0
    for _ in range(PERF_EXEC_ITERATIONS):
        _, _, perf_report = execute_fb(
            compiled_bin,
            input_output_goldens=input_output_goldens,
            intermediate_goldens=intermediate_goldens,
            device=device,
            return_perf=True,
        )
        exec_s_acc += _ns_to_s(perf_report["totals"]["runtime_ns"])
    exec_s = exec_s_acc / PERF_EXEC_ITERATIONS
    total_s = compile_s + exec_s

    print("Execute:")
    print("  - status : success")
    print(f"  - avg    : {_format_time(exec_s)}")
    print("Totals:")
    print(f"  - compile + execute(avg) : {_format_time(total_s)}")

    PERF_TIMINGS[snippet_id][target] = {
        "compile_s": compile_s,
        "exec_s": exec_s,
        "total_s": total_s,
    }

    if "ttmetal" in PERF_TIMINGS[snippet_id] and "ttnn" in PERF_TIMINGS[snippet_id]:
        ttmetal = PERF_TIMINGS[snippet_id]["ttmetal"]
        ttnn = PERF_TIMINGS[snippet_id]["ttnn"]
        compile_delta, compile_ratio = _target_delta_and_ratio(
            ttmetal["compile_s"], ttnn["compile_s"]
        )
        exec_delta, exec_ratio = _target_delta_and_ratio(
            ttmetal["exec_s"], ttnn["exec_s"]
        )
        total_delta, total_ratio = _target_delta_and_ratio(
            ttmetal["total_s"], ttnn["total_s"]
        )
        print("Comparison (ttmetal - ttnn):")
        print(
            f"  - compile : {_format_signed_time(compile_delta)} ({_format_ratio_description(compile_ratio)})"
        )
        print(
            f"  - execute : {_format_signed_time(exec_delta)} ({_format_ratio_description(exec_ratio)})"
        )
        print(
            f"  - total   : {_format_signed_time(total_delta)} ({_format_ratio_description(total_ratio)})"
        )

    print("=" * LOG_WIDTH + "\n")
