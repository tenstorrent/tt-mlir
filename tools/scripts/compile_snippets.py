# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
compile_snippets.py - Compile MLIR snippet functions in isolation.

Each func.func in the input module is compiled in a separate subprocess
so that crashes/hangs in one snippet do not block the rest. Produces
per-component logs and a stable JSON results file.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Event

SCRIPT_PATH = os.path.abspath(__file__)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_FUNC_DEF_RE = re.compile(
    r'\s*func\.func(?:\s+(?:public|private|nested))?\s+@(?P<symbol>"(?:\\.|[^"\\])*"|[^\s(]+)'
)


def _strip_ansi(text):
    return _ANSI_RE.sub("", text)


def _extract_error_summary(stderr_text):
    """Extract error summary from stderr following strict precedence.

    Priority:
      1. first line containing ``error:``
      2. first assertion-like line (``*.cpp:<line>: ... Assertion ... failed``)
      3. last non-empty line
      4. fallback string
    """
    text = _strip_ansi(stderr_text)
    lines = text.strip().splitlines()

    for line in lines:
        if "error:" in line:
            return line.strip()

    for line in lines:
        if re.search(r"\S+\.cpp:\d+:.*Assertion.*failed", line):
            return line.strip()

    for line in reversed(lines):
        if line.strip():
            return line.strip()

    return "No stderr captured"


def _update_brace_depth(line, depth):
    in_string = False
    prev_ch = None
    saw_open = False
    for ch in line:
        if ch == '"' and prev_ch != "\\":
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
                saw_open = True
            elif ch == "}":
                depth -= 1
        prev_ch = ch
    return depth, saw_open


def _sanitize_symbol_name(symbol):
    raw = symbol
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("._")
    return safe or "func"


def _extract_functions_from_text(mlir_text):
    """Extract ``func.func`` ops from module text.

    Returns a list of *(func_name, standalone_module_text, dir_name)* tuples.
    """
    lines = mlir_text.split("\n")
    funcs = []
    used_dir_names = set()

    def unique_dir_name(symbol):
        base = _sanitize_symbol_name(symbol)
        if base not in used_dir_names:
            used_dir_names.add(base)
            return base
        idx = 2
        while f"{base}_{idx}" in used_dir_names:
            idx += 1
        name = f"{base}_{idx}"
        used_dir_names.add(name)
        return name

    module_start = None
    module_end = None
    for i, line in enumerate(lines):
        if re.match(r"\s*module\b", line):
            depth = 0
            found_brace = False
            for j in range(i, len(lines)):
                depth, saw_open = _update_brace_depth(lines[j], depth)
                found_brace = found_brace or saw_open
                if found_brace and depth == 0:
                    module_start = i
                    module_end = j
                    break
            if module_start is not None:
                break

    if module_start is not None and module_end is not None:
        module_funcs = []
        depth = 0
        i = module_start
        while i <= module_end:
            if depth == 1:
                match = _FUNC_DEF_RE.match(lines[i])
                if match:
                    symbol = match.group("symbol")
                    start = i
                    func_depth = 0
                    found_brace = False
                    while i <= module_end:
                        func_depth, saw_open = _update_brace_depth(lines[i], func_depth)
                        found_brace = found_brace or saw_open
                        if found_brace and func_depth == 0:
                            i += 1
                            break
                        i += 1
                    module_funcs.append((symbol, start, i))
                    continue
            depth, _ = _update_brace_depth(lines[i], depth)
            i += 1

        for idx, (func_name, start, end) in enumerate(module_funcs):
            remove_lines = set()
            for other_idx, (_, other_start, other_end) in enumerate(module_funcs):
                if other_idx != idx:
                    remove_lines.update(range(other_start, other_end))
            snippet_lines = [
                lines[line_idx]
                for line_idx in range(module_start, module_end + 1)
                if line_idx not in remove_lines
            ]
            funcs.append(
                (func_name, "\n".join(snippet_lines), unique_dir_name(func_name))
            )
        return funcs

    i = 0
    while i < len(lines):
        match = _FUNC_DEF_RE.match(lines[i])
        if not match:
            i += 1
            continue
        func_name = match.group("symbol")
        depth = 0
        start = i
        found_brace = False
        while i < len(lines):
            depth, saw_open = _update_brace_depth(lines[i], depth)
            found_brace = found_brace or saw_open
            if found_brace and depth == 0:
                i += 1
                break
            i += 1
        func_text = "\n".join(lines[start:i])
        snippet = f"module {{\n{func_text}\n}}"
        funcs.append((func_name, snippet, unique_dir_name(func_name)))
    return funcs


def _collect_artifacts(comp_dir, mlir_path):
    """Return absolute paths of compile artifacts (anything that isn't the
    input MLIR or the log files)."""
    skip = {
        os.path.basename(mlir_path),
        "compile_stdout.txt",
        "compile_stderr.txt",
    }
    artifacts = []
    for name in sorted(os.listdir(comp_dir)):
        if name not in skip:
            artifacts.append(os.path.abspath(os.path.join(comp_dir, name)))
    return artifacts


def _compile_one(
    func_name, dir_name, snippet_text, output_dir, system_desc, timeout_sec, log
):
    """Compile a single snippet in an isolated subprocess.

    Returns *(func_name, result_dict)*.
    """
    comp_dir = os.path.join(output_dir, dir_name)
    os.makedirs(comp_dir, exist_ok=True)

    mlir_path = os.path.join(comp_dir, f"{dir_name}.mlir")
    stdout_path = os.path.join(comp_dir, "compile_stdout.txt")
    stderr_path = os.path.join(comp_dir, "compile_stderr.txt")

    with open(mlir_path, "w", encoding="utf-8") as f:
        f.write(snippet_text)

    result = {
        "status": "infra_error",
        "duration_sec": 0.0,
        "exit_code": None,
        "signal": None,
        "dir": os.path.abspath(comp_dir),
        "snippet_path": os.path.abspath(mlir_path),
        "stdout_path": os.path.abspath(stdout_path),
        "stderr_path": os.path.abspath(stderr_path),
        "error_summary": None,
        "artifact_paths": [],
    }

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                SCRIPT_PATH,
                "--worker",
                mlir_path,
                "--system-desc",
                system_desc,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        elapsed = time.monotonic() - start
        result["duration_sec"] = round(elapsed, 3)
        result["exit_code"] = proc.returncode

        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout or "")
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(proc.stderr or "")

        if proc.returncode < 0:
            result["status"] = "crash"
            result["signal"] = -proc.returncode
            result["error_summary"] = _extract_error_summary(proc.stderr or "")
        elif proc.returncode == 0 and "COMPILE_OK" in (proc.stdout or ""):
            result["status"] = "pass"
            result["artifact_paths"] = _collect_artifacts(comp_dir, mlir_path)
        else:
            result["status"] = "fail"
            result["error_summary"] = _extract_error_summary(proc.stderr or "")

    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        result["duration_sec"] = round(elapsed, 3)
        result["status"] = "timeout"
        result["error_summary"] = f"Compilation timeout (> {timeout_sec}s)"
        stdout_data = exc.stdout or ""
        stderr_data = exc.stderr or ""
        if isinstance(stdout_data, bytes):
            stdout_data = stdout_data.decode("utf-8", errors="replace")
        if isinstance(stderr_data, bytes):
            stderr_data = stderr_data.decode("utf-8", errors="replace")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(stdout_data)
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(stderr_data)

    except Exception as exc:
        elapsed = time.monotonic() - start
        result["duration_sec"] = round(elapsed, 3)
        result["error_summary"] = f"Infrastructure error: {exc}"
        for p in (stdout_path, stderr_path):
            if not os.path.exists(p):
                with open(p, "w", encoding="utf-8") as f:
                    pass

    status_tag = result["status"].upper()
    log.info("  [%s] %s (%.1fs)", status_tag, func_name, result["duration_sec"])
    return func_name, result


def _run_worker(mlir_path, system_desc):
    """Subprocess entry-point: load and compile a single MLIR snippet."""
    from builder.base.builder_apis import (
        compile_ttir_module_to_flatbuffer,
        load_mlir_file,
    )

    with open(mlir_path, "r", encoding="utf-8") as f:
        mlir_text = f.read()

    artifact_dir = os.path.dirname(os.path.abspath(mlir_path))
    module, builder = load_mlir_file(mlir_text, target="ttir")
    compile_ttir_module_to_flatbuffer(
        module,
        builder,
        system_desc_path=system_desc,
        artifact_dir=artifact_dir,
        target="ttmetal",
        save_artifacts=True,
    )
    print("COMPILE_OK")


def _build_summary(components):
    summary = {
        "total": len(components),
        "passed": 0,
        "failed": 0,
        "crashed": 0,
        "timed_out": 0,
        "infra_errors": 0,
    }
    _STATUS_KEY = {
        "pass": "passed",
        "fail": "failed",
        "crash": "crashed",
        "timeout": "timed_out",
        "infra_error": "infra_errors",
    }
    for comp in components.values():
        key = _STATUS_KEY.get(comp["status"])
        if key:
            summary[key] += 1
    return summary


def _write_results(path, args_snippets_mlir, args_system_desc, timeout_sec, components):
    summary = _build_summary(components)
    data = {
        "schema_version": 1,
        "input_snippets_mlir": os.path.abspath(args_snippets_mlir),
        "system_desc": os.path.abspath(args_system_desc),
        "target": "ttmetal",
        "timeout_sec": timeout_sec,
        "summary": summary,
        "components": components,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compile MLIR snippet functions in isolation"
    )

    parser.add_argument(
        "--worker",
        type=str,
        metavar="MLIR_PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "snippets_mlir",
        nargs="?",
        help="MLIR module containing func.func snippets",
    )
    parser.add_argument(
        "--system-desc",
        required=True,
        help="Path to system descriptor",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Per-component timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--stop-on-keyboard-interrupt",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Stop scheduling on Ctrl-C and write partial results (default: true)",
    )

    args = parser.parse_args()

    # ── Worker mode (internal) ──────────────────────────────────────────
    if args.worker:
        _run_worker(args.worker, args.system_desc)
        return

    # ── Normal mode ─────────────────────────────────────────────────────
    if not args.snippets_mlir:
        parser.error("snippets_mlir is required")
    if not os.path.isfile(args.snippets_mlir):
        print(f"Error: file not found: {args.snippets_mlir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.system_desc):
        print(
            f"Error: system descriptor not found: {args.system_desc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_dir = os.path.dirname(os.path.abspath(args.snippets_mlir))
        output_dir = os.path.join(input_dir, f"compiled_snippets_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    log = logging.getLogger("compile_snippets")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    console_h = logging.StreamHandler(sys.stderr)
    console_h.setFormatter(fmt)
    log.addHandler(console_h)

    file_h = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
    file_h.setFormatter(fmt)
    log.addHandler(file_h)

    log.info("=" * 60)
    log.info("compile_snippets: Isolated snippet compilation")
    log.info("=" * 60)
    log.info("  Input:       %s", args.snippets_mlir)
    log.info("  System desc: %s", args.system_desc)
    log.info("  Output dir:  %s", output_dir)
    log.info("  Timeout:     %ds", args.timeout_sec)
    log.info("  Jobs:        %d", args.jobs)
    log.info("")

    # ── Step 1: Parse ───────────────────────────────────────────────────
    log.info("[1/2] Parsing MLIR and extracting functions ...")
    with open(args.snippets_mlir, "r", encoding="utf-8") as f:
        mlir_text = f.read()

    try:
        funcs = _extract_functions_from_text(mlir_text)
    except Exception as exc:
        log.error("  Failed to parse MLIR: %s", exc)
        sys.exit(1)

    log.info("  Found %d function(s)", len(funcs))
    if not funcs:
        log.info("  Nothing to compile.")
        results_path = os.path.join(output_dir, "compile_results.json")
        _write_results(
            results_path, args.snippets_mlir, args.system_desc, args.timeout_sec, {}
        )
        log.info("  Results: %s", results_path)
        return

    # ── Step 2: Compile ─────────────────────────────────────────────────
    log.info("[2/2] Compiling %d snippet(s) ...", len(funcs))
    components = {}
    cancel_event = Event()

    executor = ThreadPoolExecutor(max_workers=args.jobs)
    futures = {}
    interrupted = False
    try:
        for func_name, snippet, dir_name in funcs:
            if cancel_event.is_set():
                break
            fut = executor.submit(
                _compile_one,
                func_name,
                dir_name,
                snippet,
                output_dir,
                args.system_desc,
                args.timeout_sec,
                log,
            )
            futures[fut] = (func_name, dir_name)

        for fut in as_completed(futures):
            try:
                name, result = fut.result()
                components[name] = result
            except Exception as exc:
                name, dir_name = futures[fut]
                log.error("  [INFRA_ERROR] %s: %s", name, exc)
                comp_dir = os.path.join(output_dir, dir_name)
                components[name] = {
                    "status": "infra_error",
                    "duration_sec": 0.0,
                    "exit_code": None,
                    "signal": None,
                    "dir": os.path.abspath(comp_dir),
                    "snippet_path": os.path.abspath(
                        os.path.join(comp_dir, f"{dir_name}.mlir")
                    ),
                    "stdout_path": os.path.abspath(
                        os.path.join(comp_dir, "compile_stdout.txt")
                    ),
                    "stderr_path": os.path.abspath(
                        os.path.join(comp_dir, "compile_stderr.txt")
                    ),
                    "error_summary": f"Infrastructure error: {exc}",
                    "artifact_paths": [],
                }
    except KeyboardInterrupt:
        if args.stop_on_keyboard_interrupt:
            cancel_event.set()
            interrupted = True
            for fut in futures:
                fut.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            log.warning("  Keyboard interrupt – writing partial results ...")
            for fut, (name, _) in futures.items():
                if not fut.done() or name in components:
                    continue
                if fut.cancelled():
                    continue
                try:
                    done_name, result = fut.result()
                    components[done_name] = result
                except Exception as exc:
                    log.error("  [INFRA_ERROR] %s: %s", name, exc)
        else:
            raise
    finally:
        if not interrupted:
            executor.shutdown(wait=True)

    # ── Write results ───────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "compile_results.json")
    summary = _write_results(
        results_path,
        args.snippets_mlir,
        args.system_desc,
        args.timeout_sec,
        components,
    )

    log.info("")
    log.info("Summary:")
    log.info("  Total:        %d", summary["total"])
    log.info("  Passed:       %d", summary["passed"])
    log.info("  Failed:       %d", summary["failed"])
    log.info("  Crashed:      %d", summary["crashed"])
    log.info("  Timed out:    %d", summary["timed_out"])
    log.info("  Infra errors: %d", summary["infra_errors"])
    log.info("  Results:      %s", results_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
