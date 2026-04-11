# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute every func.func in a single ops.mlir-style module (non-pytest).

Mirrors the flow in test_snippets.py: split the module into one-module-per-function
snippets, compile each to TTMetal (default), and execute on device.

Usage (from tt-mlir repo root, after ``source env/activate``)::

    python tools/scripts/run_ops_mlir_snippets.py path/to/ops.mlir

Requires a system descriptor (same as pytest golden tests), e.g.::

    ttrt query --save-artifacts
    export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Repo layout: tools/scripts/<this file> -> tt-mlir root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

import math

import _ttmlir_runtime as tt_runtime  # noqa: E402

from builder.base.builder_apis import (  # noqa: E402
    compile_ttir_module_to_flatbuffer,
    load_mlir_file,
)
from builder.base.builder_runtime import execute_fb  # noqa: E402
from builder.base.builder_utils import get_artifact_dir  # noqa: E402


def extract_functions_from_mlir(mlir_content: str) -> List[Tuple[str, str]]:
    """Extract each top-level func.func as its own wrapped module (same as test_snippets)."""
    functions: List[Tuple[str, str]] = []
    func_start_pattern = re.compile(r"func\.func\s+@(\w+)\s*\(")

    for match in func_start_pattern.finditer(mlir_content):
        func_name = match.group(1)
        start_pos = match.start()
        brace_pos = mlir_content.find("{", match.end())
        if brace_pos == -1:
            continue

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
            func_mlir = f"module {{\n  {func_body}\n}}"
            functions.append((func_name, func_mlir))

    return functions


def _open_mesh_device(
    target: str,
    mesh_shape: Tuple[int, int],
    *,
    disable_eth_dispatch: bool,
):
    """Open a mesh device for ttmetal or ttnn (same logic as golden conftest)."""
    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    if disable_eth_dispatch:
        mesh_options.dispatch_core_type = tt_runtime.runtime.DispatchCoreType.WORKER
    mesh_options.mesh_shape = mesh_shape

    if target == "ttnn":
        device_runtime_enum = tt_runtime.runtime.DeviceRuntime.TTNN
    elif target == "ttmetal":
        device_runtime_enum = tt_runtime.runtime.DeviceRuntime.TTMetal
    else:
        raise ValueError(f"unsupported target: {target}")

    tt_runtime.runtime.set_current_device_runtime(device_runtime_enum)
    if math.prod(mesh_shape) > 1:
        tt_runtime.runtime.set_fabric_config(tt_runtime.runtime.FabricConfig.FABRIC_1D)
    device = tt_runtime.runtime.open_mesh_device(mesh_options)
    print(f"Device opened for target {target}, mesh shape {mesh_options.mesh_shape}.")
    return device


def _close_device(device) -> None:
    if device is None:
        return
    try:
        tt_runtime.runtime.close_mesh_device(device)
        tt_runtime.runtime.set_fabric_config(tt_runtime.runtime.FabricConfig.DISABLED)
    except Exception as e:
        print(f"Warning: failed to close device: {e}", file=sys.stderr)


@contextlib.contextmanager
def _capture_fd_stderr():
    """Capture C-level stderr (fd 2) so MLIR diagnostics are preserved."""
    sys.stderr.flush()
    old_fd = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    captured: List[str] = []
    try:
        yield captured
    finally:
        sys.stderr.flush()
        try:
            os.fsync(2)
        except OSError:
            pass
        os.dup2(old_fd, 2)
        os.close(old_fd)
        tmp.seek(0)
        captured.append(tmp.read().decode("utf-8", errors="replace"))
        tmp.close()


@dataclass
class SnippetResult:
    func_name: str
    compile_ok: bool = False
    compile_error: Optional[str] = None
    compile_diagnostics: str = ""
    exec_ok: Optional[bool] = None
    exec_error: Optional[str] = None
    exec_diagnostics: str = ""


def _status_label(r: SnippetResult) -> str:
    if not r.compile_ok:
        return "COMPILE FAIL"
    if r.exec_ok is False:
        return "EXEC FAIL"
    return "PASSED"


def _write_report(
    results: List[SnippetResult],
    report_path: Path,
    *,
    target: str,
    ops_path: Path,
    skip_exec: bool,
) -> None:
    """Write a human-readable report summarizing compile/execute results."""
    mode = "compile-only" if skip_exec else "compile+execute"
    total = len(results)
    name_width = max((len(r.func_name) for r in results), default=12)
    name_width = max(name_width, 9)
    status_width = 12  # len("COMPILE FAIL")

    n_passed = sum(1 for r in results if r.compile_ok and r.exec_ok is not False)
    n_compile_fail = sum(1 for r in results if not r.compile_ok)
    n_exec_fail = sum(1 for r in results if r.exec_ok is False)

    failed_results = [r for r in results if not r.compile_ok or r.exec_ok is False]

    sep = "─" * 72
    lines: List[str] = []

    # -- summary --
    lines.append(f"target:  {target}")
    lines.append(f"input:   {ops_path}")
    lines.append(f"mode:    {mode}")
    lines.append(f"total:   {total} ops")
    lines.append("")
    lines.append(f"  PASSED:       {n_passed}")
    lines.append(f"  COMPILE FAIL: {n_compile_fail}")
    if not skip_exec:
        lines.append(f"  EXEC FAIL:    {n_exec_fail}")
    lines.append("")

    # -- failure details (before the full table) --
    if failed_results:
        lines.append(sep)
        lines.append("")
        lines.append(f"  Failure details ({len(failed_results)})")
        for idx, r in enumerate(failed_results, 1):
            label = _status_label(r)
            err = r.compile_error if not r.compile_ok else r.exec_error
            diag = r.compile_diagnostics if not r.compile_ok else r.exec_diagnostics
            lines.append("")
            lines.append(f"  [{idx}] {r.func_name} — {label}")
            if err:
                lines.append(f"      exception: {err}")
            diag_lines = _extract_diagnostics(diag)
            if diag_lines:
                lines.append("      diagnostics:")
                for dl in diag_lines:
                    lines.append(f"        {dl}")
        lines.append("")

    # -- full per-op table (at the bottom) --
    lines.append(sep)
    lines.append("")
    hdr_name = "func_name".ljust(name_width)
    lines.append(f"  {hdr_name}  status")
    lines.append(f"  {'─' * name_width}  {'─' * status_width}")

    for r in results:
        col_name = r.func_name.ljust(name_width)
        label = _status_label(r)
        lines.append(f"  {col_name}  {label}")

    lines.append("")
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")


_DIAG_PATTERNS = [
    re.compile(r"error:.*", re.IGNORECASE),
    re.compile(r"can't find feasible allocation.*"),
    re.compile(r"required L1 memory.*"),
    re.compile(r"usable space is.*"),
    re.compile(r"No parser found.*"),
]


def _extract_diagnostics(raw: str) -> List[str]:
    """Pull meaningful lines from captured stderr, dedup and filter noise."""
    if not raw:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip ANSI escape-only lines and common non-diagnostic noise.
        clean = re.sub(r"\x1b\[[0-9;]*m", "", stripped).strip()
        if not clean:
            continue
        if clean.startswith("Warning:"):
            continue
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


@dataclass
class FileResults:
    """Aggregated results for one ops.mlir file."""

    ops_path: Path
    results: List[SnippetResult] = field(default_factory=list)
    failed: int = 0


def _write_combined_report(
    dir_path: Path,
    file_results: List[FileResults],
    *,
    target: str,
    skip_exec: bool,
) -> None:
    """Write a directory-level ``ops-run-report.txt`` aggregating all files."""
    mode = "compile-only" if skip_exec else "compile+execute"
    sep = "\u2500" * 72

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(dir_path))
        except ValueError:
            return str(p)

    grand_total = sum(len(fr.results) for fr in file_results)
    grand_passed = sum(
        sum(1 for r in fr.results if r.compile_ok and r.exec_ok is not False)
        for fr in file_results
    )
    grand_compile_fail = sum(
        sum(1 for r in fr.results if not r.compile_ok) for fr in file_results
    )
    grand_exec_fail = sum(
        sum(1 for r in fr.results if r.exec_ok is False) for fr in file_results
    )

    rel_names = [_rel(fr.ops_path) for fr in file_results]
    name_w = max((len(n) for n in rel_names), default=12)
    name_w = max(name_w, 9)

    lines: List[str] = [
        f"Directory: {dir_path.name}/",
        f"target:    {target}",
        f"mode:      {mode}",
        f"files:     {len(file_results)}",
        f"total ops: {grand_total}",
        "",
        f"  PASSED:       {grand_passed}",
        f"  COMPILE FAIL: {grand_compile_fail}",
    ]
    if not skip_exec:
        lines.append(f"  EXEC FAIL:    {grand_exec_fail}")
    lines += ["", sep, "", "Per-file summary:"]

    for fr, rname in zip(file_results, rel_names):
        n = len(fr.results)
        p = sum(1 for r in fr.results if r.compile_ok and r.exec_ok is not False)
        cf = sum(1 for r in fr.results if not r.compile_ok)
        ef = sum(1 for r in fr.results if r.exec_ok is False)
        fail_str = f"{cf} compile" + (f", {ef} exec" if ef else "")
        lines.append(
            f"  {rname:{name_w}s}  {n:4d} ops"
            f"   {p:4d} passed"
            f"   {n - p:4d} failed ({fail_str})"
        )

    all_failures = [
        (_rel(fr.ops_path), r)
        for fr in file_results
        for r in fr.results
        if not r.compile_ok or r.exec_ok is False
    ]
    if all_failures:
        lines += ["", sep, "", f"  All failures ({len(all_failures)})"]
        for idx, (rname, r) in enumerate(all_failures, 1):
            label = _status_label(r)
            err = r.compile_error if not r.compile_ok else r.exec_error
            diag = r.compile_diagnostics if not r.compile_ok else r.exec_diagnostics
            lines.append("")
            lines.append(f"  [{idx}] {rname}/{r.func_name} \u2014 {label}")
            if err:
                lines.append(f"      exception: {err}")
            diag_lines = _extract_diagnostics(diag)
            if diag_lines:
                lines.append("      diagnostics:")
                for dl in diag_lines:
                    lines.append(f"        {dl}")

    lines.append("")
    report = dir_path / "ops-run-report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCombined report written to {report}")


def _process_file(
    ops_path: Path,
    args,
    device,
) -> tuple[List[SnippetResult], int, object, bool]:
    """Process a single ops.mlir file.

    Returns (results, failed_count, device, should_stop).
    ``device`` may differ from the input if it was re-opened after an exec failure.
    """
    content = ops_path.read_text().strip()
    if not content:
        print(f"warning: empty file {ops_path}, skipping", file=sys.stderr)
        return [], 0, device, False

    snippets = extract_functions_from_mlir(content)
    if hasattr(args, "func") and args.func:
        snippets = [
            (name, mlir) for name, mlir in snippets if args.func in name
        ]
    if not snippets:
        print(
            f"warning: no matching func.func definitions in {ops_path}, skipping",
            file=sys.stderr,
        )
        return [], 0, device, False

    rel_label = ops_path.name
    results: List[SnippetResult] = []
    failed = 0
    should_stop = False

    for func_name, func_mlir in snippets:
        snippet_id = f"{rel_label}/{func_name}"
        artifact_dir = get_artifact_dir(
            args.output_root,
            f"ops_mlir_snippets/{snippet_id}",
            args.target,
            args.save_artifacts,
        )
        print(f"\n{'=' * 60}")
        print(f"Snippet: {snippet_id}")
        print(f"{'=' * 60}")

        result = SnippetResult(func_name=func_name)

        capture_ctx = (
            contextlib.nullcontext([])
            if args.print_ir
            else _capture_fd_stderr()
        )
        with capture_ctx as captured:
            try:
                module, builder = load_mlir_file(func_mlir, target="ttir")
                (
                    compiled_bin,
                    input_output_goldens,
                    intermediate_goldens,
                ) = compile_ttir_module_to_flatbuffer(
                    module,
                    builder,
                    system_desc_path=args.sys_desc,
                    artifact_dir=artifact_dir,
                    target=args.target,
                    save_artifacts=args.save_artifacts,
                    print_ir=args.print_ir,
                )
                result.compile_ok = True
            except Exception as e:
                result.compile_error = str(e)

        result.compile_diagnostics = captured[0] if captured else ""

        if result.compile_ok:
            print("  compile: ok")
        else:
            print(f"  compile: FAILED: {result.compile_error}", file=sys.stderr)
            failed += 1
            results.append(result)
            if args.fail_fast:
                should_stop = True
                break
            continue

        if args.skip_exec:
            results.append(result)
            continue

        capture_ctx = (
            contextlib.nullcontext([])
            if args.print_ir
            else _capture_fd_stderr()
        )
        with capture_ctx as captured:
            try:
                execute_fb(
                    compiled_bin,
                    input_output_goldens=input_output_goldens,
                    intermediate_goldens=intermediate_goldens,
                    device=device,
                    save_artifacts=args.save_artifacts,
                    artifact_dir=artifact_dir,
                )
                result.exec_ok = True
            except Exception as e:
                result.exec_ok = False
                result.exec_error = str(e)

        result.exec_diagnostics = captured[0] if captured else ""

        if result.exec_ok:
            print("  execute: ok")
        else:
            print(f"  execute: FAILED: {result.exec_error}", file=sys.stderr)
            failed += 1
            _close_device(device)
            device = _open_mesh_device(
                args.target,
                (1, 1),
                disable_eth_dispatch=args.disable_eth_dispatch,
            )
            if args.fail_fast:
                results.append(result)
                should_stop = True
                break

        results.append(result)

    return results, failed, device, should_stop


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and run every function in an ops.mlir-style snippet file "
        "(or every .mlir file in a directory)."
    )
    parser.add_argument(
        "ops_mlir",
        type=Path,
        nargs="+",
        help="One or more ops.mlir files (or directories of .mlir files)",
    )
    parser.add_argument(
        "--target",
        default="ttmetal",
        choices=["ttmetal", "ttnn"],
        help="Compile target (default: ttmetal, same as snippet tests)",
    )
    parser.add_argument(
        "--sys-desc",
        default=os.environ.get("SYSTEM_DESC_PATH", "ttrt-artifacts/system_desc.ttsys"),
        help="System descriptor path (default: SYSTEM_DESC_PATH env or ttrt-artifacts/...)",
    )
    parser.add_argument(
        "--output-root",
        default=".",
        help="Root for artifact dirs (default: current directory)",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Keep flatbuffers / mlir under the artifact dir",
    )
    parser.add_argument("--print-ir", action="store_true", help="Print compiled MLIR")
    parser.add_argument(
        "--skip-exec",
        action="store_true",
        help="Compile only; do not run on device",
    )
    parser.add_argument(
        "--disable-eth-dispatch",
        action="store_true",
        help="Same as pytest --disable-eth-dispatch",
    )
    parser.add_argument(
        "--func",
        "-f",
        help="Only run the function with this name (substring match)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first compile or execution failure",
    )
    args = parser.parse_args()

    ops_files: List[Path] = []
    for p in args.ops_mlir:
        p = p.resolve()
        if p.is_dir():
            expanded = sorted(p.glob("*.mlir"))
            if not expanded:
                print(f"warning: no .mlir files in {p}", file=sys.stderr)
            ops_files.extend(expanded)
        elif p.is_file():
            ops_files.append(p)
        else:
            print(f"warning: skipping non-existent path: {p}", file=sys.stderr)
    if not ops_files:
        print("error: no .mlir files found", file=sys.stderr)
        return 2

    device = None
    if not args.skip_exec:
        device = _open_mesh_device(
            args.target,
            (1, 1),
            disable_eth_dispatch=args.disable_eth_dispatch,
        )

    total_failed = 0
    total_snippets = 0
    all_file_results: List[FileResults] = []
    try:
        for ops_path in ops_files:
            if len(ops_files) > 1:
                print(f"\n{'#' * 60}")
                print(f"# File: {ops_path.name}")
                print(f"{'#' * 60}")

            results, failed, device, should_stop = _process_file(ops_path, args, device)

            if results:
                report_path = ops_path.parent / f"{ops_path.stem}-run-report.txt"
                _write_report(
                    results,
                    report_path,
                    target=args.target,
                    ops_path=ops_path,
                    skip_exec=args.skip_exec,
                )
                all_file_results.append(
                    FileResults(ops_path=ops_path, results=results, failed=failed)
                )

            total_failed += failed
            total_snippets += len(results)

            if should_stop:
                break
    finally:
        _close_device(device)

    if len(all_file_results) > 1:
        common = Path(
            os.path.commonpath([fr.ops_path.parent for fr in all_file_results])
        )
        _write_combined_report(
            common,
            all_file_results,
            target=args.target,
            skip_exec=args.skip_exec,
        )

    if total_failed:
        print(
            f"\nDone: {total_failed} snippet(s) failed"
            f" across {len(ops_files)} file(s).",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nDone: all {total_snippets} snippet(s) succeeded"
        f" across {len(ops_files)} file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
