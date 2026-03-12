#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate and serve an Add-Op review page.

Collects test results, git diffs, and emitted code for a newly added op,
bundles everything into a self-contained HTML page, and serves it via a
local HTTP server.

Usage:
    python generate_review.py --op-name gather_dim \\
        [--ttnn-test-dir test/ttmlir/Dialect/TTNN/gather] \\
        [--emitc-test-dir test/ttmlir/EmitC/TTNN/gather] \\
        [--pytest-filter gather_dim] \\
        [--emitpy-input test/ttmlir/Dialect/TTNN/gather/simple_gather.mlir] \\
        [--emitc-input test/ttmlir/Dialect/TTNN/gather/simple_gather.mlir] \\
        [--port 3118]

    python generate_review.py --op-name gather_dim --static review.html

No dependencies beyond the Python stdlib are required.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


def run_cmd(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    """Run a command and return (returncode, combined stdout+stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return result.returncode, output.strip()
    except subprocess.TimeoutExpired:
        return 1, f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


def collect_tests(
    op_name: str,
    ttnn_test_dir: str | None,
    emitc_test_dir: str | None,
    pytest_filter: str | None,
) -> dict:
    """Run tests and collect results."""
    tests = {}

    # 1. Lit TTNN conversion test
    if ttnn_test_dir and Path(ttnn_test_dir).exists():
        rc, output = run_cmd(["llvm-lit", ttnn_test_dir, "-v"])
        passed, failed = parse_lit_output(output)
        tests["lit_ttnn"] = {
            "status": "pass" if rc == 0 else "fail",
            "output": output,
            "passed": passed,
            "failed": failed,
            "count": f"{passed}/{passed + failed}",
        }

    # 2. Lit EmitC pipeline test
    if emitc_test_dir and Path(emitc_test_dir).exists():
        rc, output = run_cmd(["llvm-lit", emitc_test_dir, "-v"])
        passed, failed = parse_lit_output(output)
        tests["lit_emitc"] = {
            "status": "pass" if rc == 0 else "fail",
            "output": output,
            "passed": passed,
            "failed": failed,
            "count": f"{passed}/{passed + failed}",
        }

    # 3. Pytest builder/golden tests
    if pytest_filter:
        rc, output = run_cmd(
            [
                "pytest",
                "test/python/golden/test_ttir_ops.py",
                "-k",
                pytest_filter,
                "-v",
                "--tb=short",
                "--no-header",
            ],
            timeout=300,
        )
        passed, failed = parse_pytest_output(output)
        tests["pytest"] = {
            "status": "pass" if rc == 0 else "fail",
            "output": output,
            "passed": passed,
            "failed": failed,
        }

    return tests


def parse_lit_output(output: str) -> tuple[int, int]:
    """Parse llvm-lit output to count passed/failed tests."""
    passed = len(re.findall(r"^PASS:", output, re.MULTILINE))
    failed = len(re.findall(r"^FAIL:", output, re.MULTILINE))
    # Also check the summary line
    m = re.search(r"Passed:\s*(\d+)", output)
    if m:
        passed = max(passed, int(m.group(1)))
    m = re.search(r"Failed:\s*(\d+)", output)
    if m:
        failed = max(failed, int(m.group(1)))
    return passed, failed


def parse_pytest_output(output: str) -> tuple[int, int]:
    """Parse pytest output to count passed/failed tests."""
    passed = failed = 0
    m = re.search(r"(\d+) passed", output)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        failed = int(m.group(1))
    return passed, failed


def collect_diff(op_name: str = "") -> tuple[str, str]:
    """Collect git diff for tracked and untracked files related to the op."""
    # Tracked changes
    _, tracked = run_cmd(["git", "diff"])
    _, stat = run_cmd(["git", "diff", "--stat"])

    # Untracked new files — only include files in key directories
    # that are likely related to the new op (runtime, test, include)
    RELEVANT_DIRS = [
        "runtime/",
        "test/ttmlir/",
        "test/python/",
        "include/",
        "lib/",
        "tools/",
    ]
    RELEVANT_EXTS = {".cpp", ".h", ".hpp", ".py", ".mlir", ".fbs", ".td", ".txt"}

    _, untracked_files = run_cmd(["git", "ls-files", "--others", "--exclude-standard"])
    untracked_diff = ""
    untracked_count = 0
    for f in untracked_files.strip().split("\n"):
        f = f.strip()
        if not f:
            continue
        # Must be in a relevant directory
        if not any(f.startswith(d) for d in RELEVANT_DIRS):
            continue
        # Must have a relevant extension
        if not any(f.endswith(ext) for ext in RELEVANT_EXTS):
            continue
        try:
            content = Path(f).read_text()
            lines = content.split("\n")
            untracked_diff += f"diff --git a/{f} b/{f}\n"
            untracked_diff += "new file mode 100644\n"
            untracked_diff += f"--- /dev/null\n"
            untracked_diff += f"+++ b/{f}\n"
            untracked_diff += f"@@ -0,0 +1,{len(lines)} @@\n"
            for line in lines:
                untracked_diff += f"+{line}\n"
            untracked_count += 1
        except (OSError, UnicodeDecodeError):
            pass

    full_diff = tracked
    if untracked_diff:
        if full_diff:
            full_diff += "\n"
        full_diff += untracked_diff

    if untracked_count > 0:
        stat += f"\n + {untracked_count} new untracked file(s)"

    return full_diff, stat


def collect_emitted_python(input_mlir: str | None) -> str:
    """Generate emitted Python code via the EmitPy pipeline."""
    if not input_mlir or not Path(input_mlir).exists():
        return ""

    # Stage 1: TTIR -> TTNN
    rc1, _ = run_cmd(
        [
            "ttmlir-opt",
            "--ttir-to-ttnn-backend-pipeline",
            "-o",
            "/tmp/addop_review_ttnn.mlir",
            input_mlir,
        ]
    )
    if rc1 != 0:
        return f"# Error: TTIR-to-TTNN pipeline failed\n# {_}"

    # Stage 2: TTNN -> EmitPy -> Python
    rc2, output = run_cmd(
        [
            "bash",
            "-c",
            "ttmlir-opt --ttnn-to-emitpy-pipeline /tmp/addop_review_ttnn.mlir 2>/dev/null"
            " | ttmlir-translate --mlir-to-python 2>&1",
        ]
    )
    if rc2 != 0:
        return f"# Error: EmitPy pipeline failed\n{output}"

    return output


def collect_emitted_cpp(input_mlir: str | None) -> str:
    """Generate emitted C++ code via the EmitC pipeline."""
    if not input_mlir or not Path(input_mlir).exists():
        return ""

    # Run as a single shell pipeline to inherit PATH properly
    rc, output = run_cmd(
        [
            "bash",
            "-c",
            f"ttmlir-opt --ttir-to-ttnn-backend-pipeline -o /tmp/addop_review_ttnn_c.mlir {input_mlir} 2>/dev/null"
            f" && ttmlir-opt --ttnn-to-emitc-device-pipeline -o /tmp/addop_review_emitc.mlir /tmp/addop_review_ttnn_c.mlir 2>/dev/null"
            f" && ttmlir-translate --mlir-to-cpp /tmp/addop_review_emitc.mlir 2>&1",
        ]
    )
    if rc != 0:
        return f"// Error: EmitC pipeline failed\n{output}"

    return output


def collect_all(args: argparse.Namespace) -> dict:
    """Collect all review data."""
    print("  Collecting test results...", flush=True)
    tests = collect_tests(
        args.op_name,
        args.ttnn_test_dir,
        args.emitc_test_dir,
        args.pytest_filter or args.op_name,
    )

    print("  Collecting git diff...", flush=True)
    diff, diff_stat = collect_diff(args.op_name)

    print("  Generating emitted Python...", flush=True)
    emitted_python = collect_emitted_python(args.emitpy_input)

    print("  Generating emitted C++...", flush=True)
    emitted_cpp = collect_emitted_cpp(args.emitc_input)

    return {
        "op_name": args.op_name,
        "tests": tests,
        "diff": diff,
        "diff_stat": diff_stat,
        "emitted_python": emitted_python,
        "emitted_cpp": emitted_cpp,
    }


def generate_html(data: dict) -> str:
    """Generate the complete standalone HTML page with embedded data."""
    template_path = Path(__file__).parent / "viewer.html"
    vendor_dir = Path(__file__).parent / "vendor"
    template = template_path.read_text()

    # Inline vendor CSS
    vendor_css = ""
    for name in ["diff2html.min.css", "github-dark.min.css"]:
        p = vendor_dir / name
        if p.exists():
            vendor_css += p.read_text() + "\n"
    template = template.replace("/*__VENDOR_CSS__*/", vendor_css)

    # Inline vendor JS
    vendor_js = ""
    for name in ["diff2html.min.js", "highlight.min.js", "python.min.js", "cpp.min.js"]:
        p = vendor_dir / name
        if p.exists():
            vendor_js += p.read_text() + "\n"
    template = template.replace("/*__VENDOR_JS__*/", vendor_js)

    # Inline data
    data_json = json.dumps(data)
    return template.replace(
        "/*__EMBEDDED_DATA__*/",
        f"const EMBEDDED_DATA = {data_json};",
    )


def generate_shell_html(op_name: str) -> str:
    """Generate the HTML shell with no data — page loads instantly."""
    data = {
        "op_name": op_name,
        "tests": {},
        "diff": "",
        "diff_stat": "",
        "emitted_python": "",
        "emitted_cpp": "",
        "_loading": {
            "test_lit_ttnn": True,
            "test_lit_emitc": True,
            "test_pytest": True,
            "diff": True,
            "emitted_python": True,
            "emitted_cpp": True,
        },
    }
    return generate_html(data)


# ---------------------------------------------------------------------------
# HTTP server — async data collection
# ---------------------------------------------------------------------------


class AsyncDataCollector:
    """Runs data collection in background threads."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._data: dict = {
            "op_name": args.op_name,
            "tests": {},
            "diff": "",
            "diff_stat": "",
            "emitted_python": "",
            "emitted_cpp": "",
        }
        self._lock = threading.Lock()
        self._generation = 0

    def start_refresh(self) -> None:
        """Start a background refresh of all data."""
        with self._lock:
            self._generation += 1
            gen = self._generation
            # Build loading flags — one per test suite + other sections
            loading = {
                "diff": True,
                "emitted_python": True,
                "emitted_cpp": True,
            }
            if self.args.ttnn_test_dir:
                loading["test_lit_ttnn"] = True
            if self.args.emitc_test_dir:
                loading["test_lit_emitc"] = True
            if self.args.pytest_filter or self.args.op_name:
                loading["test_pytest"] = True
            self._data = {
                "op_name": self.args.op_name,
                "tests": {},
                "diff": "",
                "diff_stat": "",
                "emitted_python": "",
                "emitted_cpp": "",
                "_loading": loading,
            }

        # Launch collection tasks in parallel threads
        threading.Thread(target=self._collect_diff, args=(gen,), daemon=True).start()
        threading.Thread(target=self._collect_emitpy, args=(gen,), daemon=True).start()
        threading.Thread(target=self._collect_emitcpp, args=(gen,), daemon=True).start()
        # Launch each test suite in its own thread
        if self.args.ttnn_test_dir:
            threading.Thread(
                target=self._collect_test_lit_ttnn, args=(gen,), daemon=True
            ).start()
        if self.args.emitc_test_dir:
            threading.Thread(
                target=self._collect_test_lit_emitc, args=(gen,), daemon=True
            ).start()
        if self.args.pytest_filter or self.args.op_name:
            threading.Thread(
                target=self._collect_test_pytest, args=(gen,), daemon=True
            ).start()

    def _is_stale(self, gen: int) -> bool:
        with self._lock:
            return gen != self._generation

    def _collect_test_lit_ttnn(self, gen: int) -> None:
        print("  [bg] Running lit TTNN tests...", flush=True)
        test_dir = self.args.ttnn_test_dir
        if not test_dir or not Path(test_dir).exists():
            return
        cmd = ["llvm-lit", test_dir, "-v"]
        rc, output = run_cmd(cmd)
        passed, failed = parse_lit_output(output)
        if self._is_stale(gen):
            return
        with self._lock:
            self._data["tests"]["lit_ttnn"] = {
                "status": "pass" if rc == 0 else "fail",
                "output": output,
                "passed": passed,
                "failed": failed,
                "count": f"{passed}/{passed + failed}",
                "cmd": " ".join(cmd),
            }
            if isinstance(self._data.get("_loading"), dict):
                self._data["_loading"]["test_lit_ttnn"] = False

    def _collect_test_lit_emitc(self, gen: int) -> None:
        print("  [bg] Running lit EmitC tests...", flush=True)
        test_dir = self.args.emitc_test_dir
        if not test_dir or not Path(test_dir).exists():
            return
        cmd = ["llvm-lit", test_dir, "-v"]
        rc, output = run_cmd(cmd)
        passed, failed = parse_lit_output(output)
        if self._is_stale(gen):
            return
        with self._lock:
            self._data["tests"]["lit_emitc"] = {
                "status": "pass" if rc == 0 else "fail",
                "output": output,
                "passed": passed,
                "failed": failed,
                "count": f"{passed}/{passed + failed}",
                "cmd": " ".join(cmd),
            }
            if isinstance(self._data.get("_loading"), dict):
                self._data["_loading"]["test_lit_emitc"] = False

    def _collect_test_pytest(self, gen: int) -> None:
        print("  [bg] Running pytest...", flush=True)
        pytest_filter = self.args.pytest_filter or self.args.op_name
        cmd = [
            "pytest",
            "test/python/golden/test_ttir_ops.py",
            "-k",
            pytest_filter,
            "-v",
            "--tb=short",
            "--no-header",
        ]
        rc, output = run_cmd(cmd, timeout=300)
        passed, failed = parse_pytest_output(output)
        if self._is_stale(gen):
            return
        with self._lock:
            self._data["tests"]["pytest"] = {
                "status": "pass" if rc == 0 else "fail",
                "output": output,
                "passed": passed,
                "failed": failed,
                "cmd": " ".join(cmd),
            }
            if isinstance(self._data.get("_loading"), dict):
                self._data["_loading"]["test_pytest"] = False

    def _collect_diff(self, gen: int) -> None:
        print("  [bg] Collecting diff...", flush=True)
        diff, diff_stat = collect_diff(self.args.op_name)
        if self._is_stale(gen):
            return
        with self._lock:
            self._data["diff"] = diff
            self._data["diff_stat"] = diff_stat
            if isinstance(self._data.get("_loading"), dict):
                self._data["_loading"]["diff"] = False

    def _collect_emitpy(self, gen: int) -> None:
        print("  [bg] Generating emitted Python...", flush=True)
        code = collect_emitted_python(self.args.emitpy_input)
        if self._is_stale(gen):
            return
        with self._lock:
            self._data["emitted_python"] = code
            if isinstance(self._data.get("_loading"), dict):
                self._data["_loading"]["emitted_python"] = False

    def _collect_emitcpp(self, gen: int) -> None:
        print("  [bg] Generating emitted C++...", flush=True)
        code = collect_emitted_cpp(self.args.emitc_input)
        if self._is_stale(gen):
            return
        with self._lock:
            self._data["emitted_cpp"] = code
            if isinstance(self._data.get("_loading"), dict):
                self._data["_loading"]["emitted_cpp"] = False

    def get_data(self) -> dict:
        with self._lock:
            return dict(self._data)


def _kill_port(port: int) -> None:
    """Kill any process listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for pid_str in result.stdout.strip().split("\n"):
            if pid_str.strip():
                try:
                    os.kill(int(pid_str.strip()), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
        if result.stdout.strip():
            time.sleep(0.5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


class ReviewHandler(BaseHTTPRequestHandler):
    """Serves the review HTML shell instantly, data via async API."""

    def __init__(
        self,
        args: argparse.Namespace,
        collector: AsyncDataCollector,
        *handler_args,
        **handler_kwargs,
    ):
        self.review_args = args
        self.collector = collector
        super().__init__(*handler_args, **handler_kwargs)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            # Each page load triggers a fresh data collection
            self.collector.start_refresh()
            html = generate_shell_html(self.review_args.op_name)
            content = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/data":
            data = self.collector.get_data()
            body = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/refresh":
            self.collector.start_refresh()
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and serve an Add-Op review page",
    )
    parser.add_argument(
        "--op-name",
        required=True,
        help="Name of the op being reviewed (e.g. gather_dim)",
    )
    parser.add_argument(
        "--ttnn-test-dir",
        help="Path to TTNN lit test directory (e.g. test/ttmlir/Dialect/TTNN/gather)",
    )
    parser.add_argument(
        "--emitc-test-dir",
        help="Path to EmitC lit test directory (e.g. test/ttmlir/EmitC/TTNN/gather)",
    )
    parser.add_argument(
        "--pytest-filter",
        help="pytest -k filter (defaults to --op-name)",
    )
    parser.add_argument(
        "--emitpy-input",
        help="MLIR file for EmitPy code generation",
    )
    parser.add_argument(
        "--emitc-input",
        help="MLIR file for EmitC code generation",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=3118,
        help="Server port (default: 3118)",
    )
    parser.add_argument(
        "--static",
        "-s",
        type=Path,
        default=None,
        help="Write standalone HTML to this path instead of starting server",
    )
    args = parser.parse_args()

    print(f"\n  Add-Op Review: {args.op_name}")
    print(f"  {'=' * 40}")

    if args.static:
        data = collect_all(args)
        html = generate_html(data)
        args.static.parent.mkdir(parents=True, exist_ok=True)
        args.static.write_text(html)
        print(f"\n  Static review written to: {args.static}\n")
        sys.exit(0)

    # Start server immediately, collect data in background
    collector = AsyncDataCollector(args)
    collector.start_refresh()

    port = args.port
    _kill_port(port)
    handler = partial(ReviewHandler, args, collector)
    try:
        server = HTTPServer(("127.0.0.1", port), handler)
    except OSError:
        server = HTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]

    url = f"http://localhost:{port}"
    print(f"\n  URL:  {url}")
    print(f"\n  Refresh the page to re-run tests and collect latest changes.")
    print(f"  Press Ctrl+C to stop.\n")

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
