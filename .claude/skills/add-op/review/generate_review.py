#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate and serve an Add-Op review page.

Collects test results, git diffs, and emitted code for one or more newly added
ops, bundles everything into a self-contained HTML page, and serves it via a
local HTTP server.

Multi-op usage (repeat --op blocks):
    python generate_review.py \\
      --op gather_dim \\
        --ttnn-test-dir test/ttmlir/Dialect/TTNN/gather \\
        --emitc-test-dir test/ttmlir/EmitC/TTNN/gather \\
        --pytest-filter gather_dim \\
        --emitpy-input test/ttmlir/Dialect/TTNN/gather/simple_gather_dim.mlir \\
        --emitc-input test/ttmlir/Dialect/TTNN/gather/simple_gather_dim.mlir \\
      --op all_reduce_async \\
        --ttnn-test-dir test/ttmlir/Dialect/TTNN/all_reduce_async \\
        ... \\
      [--port 3118]

Single-op usage (backward compatible):
    python generate_review.py --op-name gather_dim \\
        [--ttnn-test-dir ...] [--port 3118]

    python generate_review.py --op-name gather_dim --static review.html

No dependencies beyond the Python stdlib are required.
"""

import argparse
import glob as globmod
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional


@dataclass
class OpConfig:
    """Configuration for a single op to review."""

    op_name: str
    ttnn_op_name: Optional[str] = None  # op name in tt-metal C++ (if different)
    ttnn_test_dir: Optional[str] = None
    emitc_test_dir: Optional[str] = None
    pytest_filter: Optional[str] = None
    emitpy_input: Optional[str] = None
    emitc_input: Optional[str] = None


@dataclass
class GlobalConfig:
    """Global (non-per-op) configuration."""

    port: int = 3118
    static: Optional[Path] = None
    ops: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI parsing — split argv on --op boundaries
# ---------------------------------------------------------------------------


def _make_op_parser() -> argparse.ArgumentParser:
    """Parser for per-op arguments."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("op_name", nargs="?", default=None)
    p.add_argument("--ttnn-op-name", default=None)
    p.add_argument("--ttnn-test-dir", default=None)
    p.add_argument("--emitc-test-dir", default=None)
    p.add_argument("--pytest-filter", default=None)
    p.add_argument("--emitpy-input", default=None)
    p.add_argument("--emitc-input", default=None)
    return p


def parse_args(argv: list[str] | None = None) -> GlobalConfig:
    """Parse CLI args, supporting both old single-op and new multi-op syntax."""
    if argv is None:
        argv = sys.argv[1:]

    # Detect old-style --op-name (backward compatibility)
    if "--op-name" in argv:
        return _parse_legacy(argv)

    # Split argv on --op boundaries
    global_args: list[str] = []
    op_groups: list[list[str]] = []
    current: list[str] | None = None

    for arg in argv:
        if arg == "--op":
            if current is not None:
                op_groups.append(current)
            current = []
        elif current is not None:
            current.append(arg)
        else:
            global_args.append(arg)

    if current is not None:
        op_groups.append(current)

    # Parse global args
    gp = argparse.ArgumentParser(add_help=False)
    gp.add_argument("--port", "-p", type=int, default=3118)
    gp.add_argument("--static", "-s", type=Path, default=None)
    g, _ = gp.parse_known_args(global_args)

    cfg = GlobalConfig(port=g.port, static=g.static)

    # Parse per-op args
    op_parser = _make_op_parser()
    for group in op_groups:
        ns, _ = op_parser.parse_known_args(group)
        if not ns.op_name:
            print(f"Error: --op requires an op name", file=sys.stderr)
            sys.exit(1)
        cfg.ops.append(
            OpConfig(
                op_name=ns.op_name,
                ttnn_op_name=ns.ttnn_op_name,
                ttnn_test_dir=ns.ttnn_test_dir,
                emitc_test_dir=ns.emitc_test_dir,
                pytest_filter=ns.pytest_filter,
                emitpy_input=ns.emitpy_input,
                emitc_input=ns.emitc_input,
            )
        )

    if not cfg.ops:
        print("Error: at least one --op is required", file=sys.stderr)
        sys.exit(1)

    return cfg


def _parse_legacy(argv: list[str]) -> GlobalConfig:
    """Parse old-style --op-name single-op arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--ttnn-op-name", default=None)
    parser.add_argument("--ttnn-test-dir", default=None)
    parser.add_argument("--emitc-test-dir", default=None)
    parser.add_argument("--pytest-filter", default=None)
    parser.add_argument("--emitpy-input", default=None)
    parser.add_argument("--emitc-input", default=None)
    parser.add_argument("--port", "-p", type=int, default=3118)
    parser.add_argument("--static", "-s", type=Path, default=None)
    args = parser.parse_args(argv)

    return GlobalConfig(
        port=args.port,
        static=args.static,
        ops=[
            OpConfig(
                op_name=args.op_name,
                ttnn_op_name=args.ttnn_op_name,
                ttnn_test_dir=args.ttnn_test_dir,
                emitc_test_dir=args.emitc_test_dir,
                pytest_filter=args.pytest_filter,
                emitpy_input=args.emitpy_input,
                emitc_input=args.emitc_input,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------


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


def parse_lit_output(output: str) -> tuple[int, int]:
    """Parse llvm-lit output to count passed/failed tests."""
    passed = len(re.findall(r"^PASS:", output, re.MULTILINE))
    failed = len(re.findall(r"^FAIL:", output, re.MULTILINE))
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


def collect_diff() -> tuple[str, str]:
    """Collect git diff for tracked and untracked files."""
    _, tracked = run_cmd(["git", "diff"])
    _, stat = run_cmd(["git", "diff", "--stat"])

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
        if not any(f.startswith(d) for d in RELEVANT_DIRS):
            continue
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


# ---------------------------------------------------------------------------
# Op info collection — C++ and Python signatures from tt-metal
# ---------------------------------------------------------------------------

TT_METAL_OPS_ROOT = Path("third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations")


def _find_op_files(op_name: str) -> tuple[Path | None, Path | None]:
    """Locate the .hpp and _nanobind.cpp for an op by grepping for its registration.

    Searches all .hpp files under the tt-metal operations dir for:
        constexpr auto {op_name} = ttnn::register_operation<...>();
    This avoids hardcoded file-name hints — works for any op.
    """
    base = TT_METAL_OPS_ROOT
    if not base.exists():
        return None, None

    # grep for the constexpr registration line
    pattern = rf"constexpr\s+auto\s+{re.escape(op_name)}\s*="
    rc, output = run_cmd(
        ["grep", "-rl", "-P", pattern, str(base), "--include=*.hpp"],
        timeout=10,
    )
    hpp_path = None
    if rc == 0 and output.strip():
        candidates = [Path(p) for p in output.strip().split("\n") if p.strip()]
        # Prefer non-tosa variants (tosa/ subdirectories are alternative implementations)
        non_tosa = [p for p in candidates if "/tosa/" not in str(p)]
        hpp_path = non_tosa[0] if non_tosa else candidates[0]

    # The nanobind file lives in the same directory as the hpp, named {stem}_nanobind.cpp
    nb_path = None
    if hpp_path:
        stem = hpp_path.stem  # e.g. "sdpa_decode", "gather", "all_reduce_async"
        candidate = hpp_path.parent / f"{stem}_nanobind.cpp"
        if candidate.exists():
            nb_path = candidate

    return hpp_path, nb_path


def _extract_cpp_signature(hpp_path: Path, op_name: str) -> list[str]:
    """Extract invoke() signatures for the struct registered under op_name."""
    text = hpp_path.read_text()

    # Find the constexpr registration line to get the struct name
    struct_name = None
    for m in re.finditer(
        r'constexpr\s+auto\s+(\w+)\s*=\s*ttnn::register_operation<\s*"[^"]*"\s*,\s*([\w:]+)\s*>',
        text,
    ):
        if m.group(1) == op_name:
            struct_name = m.group(2).split("::")[-1]
            break

    if not struct_name:
        return _extract_all_invokes(text)

    # Find the struct block and extract invoke signatures from it
    struct_pattern = rf"struct\s+{re.escape(struct_name)}\s*\{{(.*?)\}};"
    struct_match = re.search(struct_pattern, text, re.DOTALL)
    if not struct_match:
        return _extract_all_invokes(text)

    return _extract_all_invokes(struct_match.group(1))


def _extract_all_invokes(text: str) -> list[str]:
    """Extract all static ... invoke(...) signatures from a block of C++ text."""
    signatures = []
    for m in re.finditer(
        r"(static\s+[\w:<>,\s]+\s+invoke\s*\([^)]*(?:\([^)]*\)[^)]*)*\));",
        text,
        re.DOTALL,
    ):
        sig = _normalize_signature(m.group(1))
        signatures.append(sig)

    if not signatures:
        for m in re.finditer(r"(static\b.*?invoke\b.*?);", text, re.DOTALL):
            sig = _normalize_signature(m.group(1))
            signatures.append(sig)

    return signatures


def _normalize_signature(raw: str) -> str:
    """Normalize a C++ signature: dedent while preserving line breaks."""
    lines = raw.split("\n")
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in lines]
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    # Find minimum indentation (ignoring blank lines)
    min_indent = float("inf")
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            min_indent = min(min_indent, len(line) - len(stripped))
    if min_indent == float("inf"):
        min_indent = 0
    # Dedent
    return "\n".join(line[min_indent:] for line in lines)


def _extract_python_binding(nb_path: Path, op_name: str) -> list[str]:
    """Extract all nanobind binding blocks for a specific op.

    Returns a list of binding blocks (one per overload).
    """
    text = nb_path.read_text()
    lines = text.split("\n")

    blocks = []
    binding_markers = ["bind_registered_operation", f"bind_{op_name}"]

    i = 0
    while i < len(lines):
        line = lines[i]
        is_binding_start = any(marker in line for marker in binding_markers)
        if not is_binding_start:
            i += 1
            continue

        # Scan the block to the end (balanced parens or end of function)
        block_lines = []
        depth = 0
        found_op = False
        j = i
        while j < len(lines):
            bl = lines[j]
            block_lines.append(bl)
            if op_name in bl:
                found_op = True
            depth += bl.count("(") - bl.count(")")
            if depth <= 0 and len(block_lines) > 2:
                break
            j += 1

        if found_op:
            # Collect nb::arg lines, splitting on overload boundaries.
            # Each nanobind_overload_t block ends with "},", "});" or "})"
            # after the last nb::arg line — e.g.:
            #   nb::arg("subdevice_id") = nb::none()},
            current_overload: list[str] = []
            for bl in block_lines:
                stripped = bl.strip()
                if not (
                    stripped.startswith("nb::arg(")
                    or stripped.startswith("nb::kw_only()")
                ):
                    continue
                # Detect end-of-overload: line ends with }...; or }, or })
                is_last = bool(re.search(r"\}[);,\s]*$", stripped))
                # Strip the trailing }, }); etc.
                clean = re.sub(r"\}[);,\s]*$", "", stripped).rstrip(",")
                current_overload.append(clean)
                if is_last:
                    blocks.append("\n".join(current_overload))
                    current_overload = []
            if current_overload:
                blocks.append("\n".join(current_overload))
        i = j + 1

    return blocks


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to CamelCase: 'all_reduce_async' -> 'AllReduceAsync'."""
    return "".join(part.capitalize() for part in name.split("_"))


TTNN_OPS_TD = Path("include/ttmlir/Dialect/TTNN/IR/TTNNOps.td")
TTIR_OPS_TD = Path("include/ttmlir/Dialect/TTIR/IR/TTIROps.td")


def _extract_td_def(td_path: Path, def_name: str) -> tuple[str, str]:
    """Extract a full `def ...Op` block from a .td file.

    Returns (relative_path_with_line, definition_text).
    """
    if not td_path.exists():
        return "", ""

    text = td_path.read_text()
    # Find `def DEF_NAME :` or `def DEF_NAME :`
    pattern = rf"^(def\s+{re.escape(def_name)}\s*:.*)$"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return "", ""

    start = match.start()
    line_num = text[:start].count("\n") + 1

    # Walk from the start to find the matching closing brace
    # .td defs end with `}` at column 0 (or same indent as `def`)
    lines = text[start:].split("\n")
    depth = 0
    block_lines = []
    for line in lines:
        block_lines.append(line)
        depth += line.count("{") - line.count("}")
        if depth <= 0 and len(block_lines) > 1:
            break

    definition = "\n".join(block_lines)
    rel_path = f"{td_path}:{line_num}"
    return rel_path, definition


def collect_op_info(op_name: str) -> dict:
    """Collect C++ and Python signature info for an op from tt-metal."""
    hpp_path, nb_path = _find_op_files(op_name)

    result = {
        "cpp_path": "",
        "cpp_signatures": [],
        "py_path": "",
        "python_bindings": [],
        "ttnn_td_path": "",
        "ttnn_td_def": "",
        "ttir_td_path": "",
        "ttir_td_def": "",
    }

    if hpp_path and hpp_path.exists():
        result["cpp_path"] = str(hpp_path)
        result["cpp_signatures"] = _extract_cpp_signature(hpp_path, op_name)

    if nb_path and nb_path.exists():
        result["py_path"] = str(nb_path)
        result["python_bindings"] = _extract_python_binding(nb_path, op_name)

    # Extract .td definitions
    camel = _snake_to_camel(op_name)
    ttnn_def_name = f"TTNN_{camel}Op"
    ttir_def_name = f"TTIR_{camel}Op"

    result["ttnn_td_path"], result["ttnn_td_def"] = _extract_td_def(
        TTNN_OPS_TD, ttnn_def_name
    )
    result["ttir_td_path"], result["ttir_td_def"] = _extract_td_def(
        TTIR_OPS_TD, ttir_def_name
    )

    return result


def collect_emitted_python(op_name: str, input_mlir: str | None) -> dict:
    """Generate emitted Python code via the EmitPy pipeline."""
    if not input_mlir or not Path(input_mlir).exists():
        return {"code": "", "cmd": ""}

    tmp = f"/tmp/addop_review_{op_name}_ttnn_py.mlir"
    cmd = (
        f"ttmlir-opt --ttir-to-ttnn-common-pipeline -o {tmp} {input_mlir}\n"
        f"ttmlir-opt --ttnn-common-to-emitpy-pipeline {tmp} "
        f"| ttmlir-translate --mlir-to-python"
    )

    rc1, err = run_cmd(
        ["ttmlir-opt", "--ttir-to-ttnn-common-pipeline", "-o", tmp, input_mlir]
    )
    if rc1 != 0:
        return {"code": f"# Error: TTIR-to-TTNN pipeline failed\n# {err}", "cmd": cmd}

    rc2, output = run_cmd(
        [
            "bash",
            "-c",
            f"ttmlir-opt --ttnn-common-to-emitpy-pipeline {tmp} 2>/dev/null"
            " | ttmlir-translate --mlir-to-python 2>&1",
        ]
    )
    if rc2 != 0:
        return {"code": f"# Error: EmitPy pipeline failed\n{output}", "cmd": cmd}

    return {"code": output, "cmd": cmd}


def collect_emitted_cpp(op_name: str, input_mlir: str | None) -> dict:
    """Generate emitted C++ code via the EmitC pipeline."""
    if not input_mlir or not Path(input_mlir).exists():
        return {"code": "", "cmd": ""}

    tmp_ttnn = f"/tmp/addop_review_{op_name}_ttnn_c.mlir"
    tmp_emitc = f"/tmp/addop_review_{op_name}_emitc.mlir"
    cmd = (
        f"ttmlir-opt --ttir-to-ttnn-common-pipeline -o {tmp_ttnn} {input_mlir}\n"
        f"ttmlir-opt --ttnn-common-to-emitc-pipeline -o {tmp_emitc} {tmp_ttnn}\n"
        f"ttmlir-translate --mlir-to-cpp {tmp_emitc}"
    )

    rc, output = run_cmd(
        [
            "bash",
            "-c",
            f"ttmlir-opt --ttir-to-ttnn-common-pipeline -o {tmp_ttnn} {input_mlir} 2>/dev/null"
            f" && ttmlir-opt --ttnn-common-to-emitc-pipeline -o {tmp_emitc} {tmp_ttnn} 2>/dev/null"
            f" && ttmlir-translate --mlir-to-cpp {tmp_emitc} 2>&1",
        ]
    )
    if rc != 0:
        return {"code": f"// Error: EmitC pipeline failed\n{output}", "cmd": cmd}

    return {"code": output, "cmd": cmd}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


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
    for name in [
        "diff2html.min.js",
        "highlight.min.js",
        "python.min.js",
        "cpp.min.js",
    ]:
        p = vendor_dir / name
        if p.exists():
            vendor_js += p.read_text() + "\n"
    template = template.replace("/*__VENDOR_JS__*/", vendor_js)

    data_json = json.dumps(data)
    return template.replace(
        "/*__EMBEDDED_DATA__*/",
        f"const EMBEDDED_DATA = {data_json};",
    )


def generate_shell_html(op_names: list[str], cfg: "GlobalConfig | None" = None) -> str:
    """Generate the HTML shell with loading state — page loads instantly."""
    ops = {}
    for name in op_names:
        ops[name] = {
            "info": {},
            "tests": {},
            "emitted_python": {"code": "", "cmd": ""},
            "emitted_cpp": {"code": "", "cmd": ""},
            "_loading": {
                "info": True,
                "test_lit_ttnn": True,
                "test_lit_emitc": True,
                "test_pytest": True,
                "test_emitc_dylib": True,
                "emitted_python": True,
                "emitted_cpp": True,
            },
        }
    # Build alchemist metadata for the initial page load
    op_inputs = {}
    if cfg:
        for oc in cfg.ops:
            mlir = oc.emitpy_input or ""
            output_dir = f"/tmp/addop_alchemist_{oc.op_name}"
            cmd = (
                f"tt-alchemist generate-python {mlir} -o {output_dir} --local"
                if mlir
                else ""
            )
            op_inputs[oc.op_name] = {"input_mlir": mlir, "cmd": cmd}
    data = {
        "ops": ops,
        "diff": "",
        "diff_stat": "",
        "_loading": {"diff": True},
        "alchemist": {
            "available": check_alchemist_available(),
            "build_cmd": ALCHEMIST_BUILD_CMD,
            "ops": {},
            "inputs": op_inputs,
        },
    }
    return generate_html(data)


# ---------------------------------------------------------------------------
# HTTP server — async data collection
# ---------------------------------------------------------------------------


class AsyncDataCollector:
    """Runs data collection in background threads for multiple ops."""

    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._generation = 0
        self._data: dict = self._empty_data()

    def _empty_data(self) -> dict:
        ops = {}
        for oc in self.cfg.ops:
            ops[oc.op_name] = {
                "info": {},
                "tests": {},
                "emitted_python": {"code": "", "cmd": ""},
                "emitted_cpp": {"code": "", "cmd": ""},
            }
        return {
            "ops": ops,
            "diff": "",
            "diff_stat": "",
        }

    def start_refresh(self) -> None:
        """Start a background refresh of all data."""
        with self._lock:
            self._generation += 1
            gen = self._generation

            # Build per-op loading flags
            ops = {}
            for oc in self.cfg.ops:
                loading = {
                    "info": True,
                    "emitted_python": True,
                    "emitted_cpp": True,
                }
                if oc.ttnn_test_dir:
                    loading["test_lit_ttnn"] = True
                if oc.emitc_test_dir:
                    loading["test_lit_emitc"] = True
                if oc.pytest_filter or oc.op_name:
                    loading["test_pytest"] = True
                if oc.emitc_input:
                    loading["test_emitc_dylib"] = True
                ops[oc.op_name] = {
                    "info": {},
                    "tests": {},
                    "emitted_python": {"code": "", "cmd": ""},
                    "emitted_cpp": {"code": "", "cmd": ""},
                    "_loading": loading,
                }

            self._data = {
                "ops": ops,
                "diff": "",
                "diff_stat": "",
                "_loading": {"diff": True},
            }

        # Shared diff thread
        threading.Thread(target=self._collect_diff, args=(gen,), daemon=True).start()

        # Per-op threads
        for oc in self.cfg.ops:
            threading.Thread(
                target=self._collect_info, args=(gen, oc), daemon=True
            ).start()
            if oc.ttnn_test_dir:
                threading.Thread(
                    target=self._collect_test_lit_ttnn,
                    args=(gen, oc),
                    daemon=True,
                ).start()
            if oc.emitc_test_dir:
                threading.Thread(
                    target=self._collect_test_lit_emitc,
                    args=(gen, oc),
                    daemon=True,
                ).start()
            if oc.pytest_filter or oc.op_name:
                threading.Thread(
                    target=self._collect_test_pytest,
                    args=(gen, oc),
                    daemon=True,
                ).start()
            if oc.emitc_input:
                threading.Thread(
                    target=self._collect_test_emitc_dylib,
                    args=(gen, oc),
                    daemon=True,
                ).start()
            threading.Thread(
                target=self._collect_emitpy, args=(gen, oc), daemon=True
            ).start()
            threading.Thread(
                target=self._collect_emitcpp, args=(gen, oc), daemon=True
            ).start()

    def _is_stale(self, gen: int) -> bool:
        with self._lock:
            return gen != self._generation

    def _set_op_field(self, gen: int, op_name: str, key: str, value) -> None:
        """Safely set a field on a per-op data dict and clear its loading flag."""
        if self._is_stale(gen):
            return
        with self._lock:
            op_data = self._data.get("ops", {}).get(op_name)
            if not op_data:
                return
            op_data[key] = value
            loading = op_data.get("_loading")
            if isinstance(loading, dict) and key in loading:
                loading[key] = False

    def _set_op_test(
        self, gen: int, op_name: str, test_key: str, loading_key: str, result: dict
    ) -> None:
        if self._is_stale(gen):
            return
        with self._lock:
            op_data = self._data.get("ops", {}).get(op_name)
            if not op_data:
                return
            op_data["tests"][test_key] = result
            loading = op_data.get("_loading")
            if isinstance(loading, dict) and loading_key in loading:
                loading[loading_key] = False

    # ---- Diff (shared) ----

    def _collect_diff(self, gen: int) -> None:
        print("  [bg] Collecting diff...", flush=True)
        diff, diff_stat = collect_diff()
        if self._is_stale(gen):
            return
        with self._lock:
            self._data["diff"] = diff
            self._data["diff_stat"] = diff_stat
            loading = self._data.get("_loading")
            if isinstance(loading, dict):
                loading["diff"] = False

    # ---- Per-op collectors ----

    def _collect_info(self, gen: int, oc: OpConfig) -> None:
        lookup_name = oc.ttnn_op_name or oc.op_name
        print(
            f"  [bg] [{oc.op_name}] Collecting op info ({lookup_name})...", flush=True
        )
        info = collect_op_info(lookup_name)
        self._set_op_field(gen, oc.op_name, "info", info)

    def _collect_test_lit_ttnn(self, gen: int, oc: OpConfig) -> None:
        print(f"  [bg] [{oc.op_name}] Running lit TTNN tests...", flush=True)
        test_dir = oc.ttnn_test_dir
        if not test_dir or not Path(test_dir).exists():
            return
        cmd = ["llvm-lit", test_dir, "-v"]
        rc, output = run_cmd(cmd)
        passed, failed = parse_lit_output(output)
        self._set_op_test(
            gen,
            oc.op_name,
            "lit_ttnn",
            "test_lit_ttnn",
            {
                "status": "pass" if rc == 0 else "fail",
                "output": output,
                "passed": passed,
                "failed": failed,
                "count": f"{passed}/{passed + failed}",
                "cmd": " ".join(cmd),
            },
        )

    def _collect_test_lit_emitc(self, gen: int, oc: OpConfig) -> None:
        print(f"  [bg] [{oc.op_name}] Running lit EmitC tests...", flush=True)
        test_dir = oc.emitc_test_dir
        if not test_dir or not Path(test_dir).exists():
            return
        cmd = ["llvm-lit", test_dir, "-v"]
        rc, output = run_cmd(cmd)
        passed, failed = parse_lit_output(output)
        self._set_op_test(
            gen,
            oc.op_name,
            "lit_emitc",
            "test_lit_emitc",
            {
                "status": "pass" if rc == 0 else "fail",
                "output": output,
                "passed": passed,
                "failed": failed,
                "count": f"{passed}/{passed + failed}",
                "cmd": " ".join(cmd),
            },
        )

    def _collect_test_pytest(self, gen: int, oc: OpConfig) -> None:
        print(f"  [bg] [{oc.op_name}] Running pytest...", flush=True)
        pytest_filter = oc.pytest_filter or oc.op_name
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
        self._set_op_test(
            gen,
            oc.op_name,
            "pytest",
            "test_pytest",
            {
                "status": "pass" if rc == 0 else "fail",
                "output": output,
                "passed": passed,
                "failed": failed,
                "cmd": " ".join(cmd),
            },
        )

    def _collect_test_emitc_dylib(self, gen: int, oc: OpConfig) -> None:
        """Compile EmitC output to dylib and run via ttrt emitc."""
        print(f"  [bg] [{oc.op_name}] Running EmitC dylib test...", flush=True)

        # Check ttrt availability
        if not shutil.which("ttrt"):
            self._set_op_test(
                gen,
                oc.op_name,
                "emitc_dylib",
                "test_emitc_dylib",
                {
                    "status": "skip",
                    "output": (
                        "ttrt is not available in PATH.\n"
                        "Build with: cmake -G Ninja -B build "
                        "-DTTMLIR_ENABLE_RUNTIME=ON\n"
                        "cmake --build build --target ttrt"
                    ),
                    "passed": 0,
                    "failed": 0,
                    "cmd": "",
                },
            )
            return

        input_mlir = oc.emitc_input
        if not input_mlir or not Path(input_mlir).exists():
            self._set_op_test(
                gen,
                oc.op_name,
                "emitc_dylib",
                "test_emitc_dylib",
                {
                    "status": "skip",
                    "output": "No emitc_input MLIR file configured.",
                    "passed": 0,
                    "failed": 0,
                    "cmd": "",
                },
            )
            return

        tmp_dir = f"/tmp/addop_emitc_dylib_{oc.op_name}"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_ttnn = f"{tmp_dir}/ttnn.mlir"
        tmp_emitc = f"{tmp_dir}/emitc.mlir"
        tmp_cpp = f"{tmp_dir}/ttnn-dylib.cpp"
        standalone_src = str(
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "tools"
            / "ttnn-standalone"
        )
        build_dir = f"{tmp_dir}/build"
        so_path = f"{build_dir}/libttnn-dylib.so"

        # Step 1: EmitC pipeline (TTIR -> TTNN -> EmitC -> C++)
        pipeline_cmd = (
            f"ttmlir-opt --ttir-to-ttnn-common-pipeline -o {tmp_ttnn} {input_mlir}"
            f" && ttmlir-opt --ttnn-common-to-emitc-pipeline -o {tmp_emitc} {tmp_ttnn}"
            f" && ttmlir-translate --mlir-to-cpp {tmp_emitc} > {tmp_cpp}"
        )
        rc, output = run_cmd(["bash", "-c", pipeline_cmd], timeout=120)
        if rc != 0:
            self._set_op_test(
                gen,
                oc.op_name,
                "emitc_dylib",
                "test_emitc_dylib",
                {
                    "status": "fail",
                    "output": f"EmitC pipeline failed:\n{output}",
                    "passed": 0,
                    "failed": 1,
                    "cmd": pipeline_cmd,
                },
            )
            return

        # Step 2: Compile to dylib using per-op build dir
        # Copy required source files into tmp_dir so CMake -S works
        for fname in (
            "CMakeLists.txt",
            "ttnn-precompiled.hpp",
            "workarounds.hpp",
            "compile_so.cpp",
            "compile_so.hpp",
            "ttnn-standalone.cpp",
        ):
            src = os.path.join(standalone_src, fname)
            dst = os.path.join(tmp_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        compile_cmd = (
            f"cmake -G Ninja -B {build_dir} -S {tmp_dir}"
            f" -DCMAKE_BUILD_TYPE=Release"
            f" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
            f" && cmake --build {build_dir} -- ttnn-dylib"
        )
        rc, output_compile = run_cmd(["bash", "-c", compile_cmd], timeout=300)
        if rc != 0 or not Path(so_path).exists():
            self._set_op_test(
                gen,
                oc.op_name,
                "emitc_dylib",
                "test_emitc_dylib",
                {
                    "status": "fail",
                    "output": f"Dylib compilation failed:\n{output_compile}",
                    "passed": 0,
                    "failed": 1,
                    "cmd": f"{pipeline_cmd}\n{compile_cmd}",
                },
            )
            return

        # Step 3: Run via ttrt emitc (same as CI, no --flatbuffer)
        ttrt_cmd = f"ttrt emitc {so_path}"
        rc, output_run = run_cmd(["ttrt", "emitc", so_path], timeout=300)
        full_output = (
            f"=== EmitC Pipeline ===\n{output}\n\n"
            f"=== Compile Dylib ===\n{output_compile}\n\n"
            f"=== ttrt emitc ===\n{output_run}"
        )
        full_cmd = f"{pipeline_cmd}\n{compile_cmd}\n{ttrt_cmd}"

        self._set_op_test(
            gen,
            oc.op_name,
            "emitc_dylib",
            "test_emitc_dylib",
            {
                "status": "pass" if rc == 0 else "fail",
                "output": full_output,
                "passed": 1 if rc == 0 else 0,
                "failed": 0 if rc == 0 else 1,
                "cmd": full_cmd,
            },
        )

    def _collect_emitpy(self, gen: int, oc: OpConfig) -> None:
        print(f"  [bg] [{oc.op_name}] Generating emitted Python...", flush=True)
        code = collect_emitted_python(oc.op_name, oc.emitpy_input)
        self._set_op_field(gen, oc.op_name, "emitted_python", code)

    def _collect_emitcpp(self, gen: int, oc: OpConfig) -> None:
        print(f"  [bg] [{oc.op_name}] Generating emitted C++...", flush=True)
        code = collect_emitted_cpp(oc.op_name, oc.emitc_input)
        self._set_op_field(gen, oc.op_name, "emitted_cpp", code)

    def get_data(self) -> dict:
        with self._lock:
            # Deep-ish copy to avoid races
            import copy

            return copy.deepcopy(self._data)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# tt-alchemist integration
# ---------------------------------------------------------------------------

# CMake flags required to build tt-alchemist (for user guidance)
ALCHEMIST_BUILD_FLAGS = (
    "-DTTMLIR_ENABLE_ALCHEMIST=ON "
    "-DTTMLIR_ENABLE_ALCHEMIST_WHEEL=ON "
    "-DTTMLIR_ENABLE_RUNTIME=ON"
)
ALCHEMIST_BUILD_CMD = (
    f"cmake -G Ninja -B build {ALCHEMIST_BUILD_FLAGS}\n"
    "cmake --build build --target tt-alchemist-install-wheel-editable"
)


def check_alchemist_available() -> bool:
    """Check if tt-alchemist CLI is available in PATH."""
    return shutil.which("tt-alchemist") is not None


class AlchemistState:
    """Tracks per-op tt-alchemist generate/run state."""

    def __init__(self):
        self._lock = threading.Lock()
        # op_name -> { "phase": "idle"|"generating"|"generated"|"running"|"done"|"error",
        #              "output_dir": str, "log": str, "run_log": str }
        self._ops: dict[str, dict] = {}

    def get(self, op_name: str) -> dict:
        with self._lock:
            return dict(
                self._ops.get(
                    op_name,
                    {
                        "phase": "idle",
                        "output_dir": "",
                        "log": "",
                        "run_log": "",
                    },
                )
            )

    def get_all(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._ops.items()}

    def _set(self, op_name: str, **kwargs) -> None:
        with self._lock:
            if op_name not in self._ops:
                self._ops[op_name] = {
                    "phase": "idle",
                    "output_dir": "",
                    "log": "",
                    "run_log": "",
                }
            self._ops[op_name].update(kwargs)

    def _append_log(self, op_name: str, field: str, line: str) -> None:
        """Append a line to a log field (thread-safe, streaming)."""
        with self._lock:
            if op_name not in self._ops:
                return
            prev = self._ops[op_name].get(field, "")
            self._ops[op_name][field] = (prev + "\n" + line) if prev else line

    def _stream_process(
        self,
        cmd: list[str],
        op_name: str,
        log_field: str,
        timeout: int = 300,
        cwd: str | None = None,
    ) -> int:
        """Run a command, streaming stdout/stderr into log_field line-by-line.

        Returns the process exit code, or -1 on timeout/error.
        """
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            cwd=cwd,
        )
        deadline = time.monotonic() + timeout
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                self._append_log(op_name, log_field, line)
                if time.monotonic() > deadline:
                    proc.kill()
                    self._append_log(
                        op_name, log_field, f"\n[timed out after {timeout}s]"
                    )
                    return -1
            proc.wait(timeout=max(1, deadline - time.monotonic()))
            return proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            self._append_log(op_name, log_field, f"\n[timed out after {timeout}s]")
            return -1
        except Exception as e:
            proc.kill()
            self._append_log(op_name, log_field, f"\n[error: {e}]")
            return -1

    def start_generate(self, op_name: str, input_mlir: str) -> None:
        """Launch tt-alchemist generate-python in a background thread."""
        self._set(op_name, phase="generating", log="", run_log="")
        threading.Thread(
            target=self._run_generate,
            args=(op_name, input_mlir),
            daemon=True,
        ).start()

    def _run_generate(self, op_name: str, input_mlir: str) -> None:
        output_dir = f"/tmp/addop_alchemist_{op_name}"
        cmd = [
            "tt-alchemist",
            "generate-python",
            input_mlir,
            "-o",
            output_dir,
            "--local",
        ]
        print(f"  [alchemist] [{op_name}] Generating: {' '.join(cmd)}", flush=True)
        rc = self._stream_process(cmd, op_name, "log", timeout=300)
        if rc == 0:
            self._set(op_name, phase="generated", output_dir=output_dir)
            print(f"  [alchemist] [{op_name}] Generate succeeded", flush=True)
        else:
            self._set(op_name, phase="error")
            print(f"  [alchemist] [{op_name}] Generate failed (rc={rc})", flush=True)

    def start_run(self, op_name: str) -> None:
        """Launch the generated ./run script in a background thread."""
        state = self.get(op_name)
        output_dir = state.get("output_dir", "")
        if not output_dir:
            self._set(op_name, phase="error", run_log="No output directory")
            return
        self._set(op_name, phase="running", run_log="")
        threading.Thread(
            target=self._run_solution,
            args=(op_name, output_dir),
            daemon=True,
        ).start()

    def _run_solution(self, op_name: str, output_dir: str) -> None:
        run_script = os.path.join(output_dir, "run")
        if not os.path.exists(run_script):
            self._set(
                op_name, phase="error", run_log=f"Run script not found: {run_script}"
            )
            return
        print(f"  [alchemist] [{op_name}] Running: {run_script}", flush=True)
        rc = self._stream_process(
            ["bash", run_script],
            op_name,
            "run_log",
            timeout=600,
            cwd=output_dir,
        )
        if rc == 0:
            self._set(op_name, phase="done")
            print(f"  [alchemist] [{op_name}] Run succeeded", flush=True)
        else:
            self._set(op_name, phase="error")
            print(f"  [alchemist] [{op_name}] Run failed (rc={rc})", flush=True)


# Global alchemist state (shared across requests)
_alchemist_state = AlchemistState()


class ReviewHandler(BaseHTTPRequestHandler):
    """Serves the review HTML shell instantly, data via async API."""

    def __init__(
        self,
        cfg: GlobalConfig,
        collector: AsyncDataCollector,
        *handler_args,
        **handler_kwargs,
    ):
        self.cfg = cfg
        self.collector = collector
        super().__init__(*handler_args, **handler_kwargs)

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.collector.start_refresh()
            op_names = [oc.op_name for oc in self.cfg.ops]
            html = generate_shell_html(op_names, cfg=self.cfg)
            content = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/api/data":
            data = self.collector.get_data()
            # Inject alchemist state into response
            # Build per-op input map and generate commands
            op_inputs = {}
            for oc in self.cfg.ops:
                mlir = oc.emitpy_input or ""
                output_dir = f"/tmp/addop_alchemist_{oc.op_name}"
                cmd = (
                    f"tt-alchemist generate-python {mlir} -o {output_dir} --local"
                    if mlir
                    else ""
                )
                op_inputs[oc.op_name] = {"input_mlir": mlir, "cmd": cmd}
            data["alchemist"] = {
                "available": check_alchemist_available(),
                "build_cmd": ALCHEMIST_BUILD_CMD,
                "ops": _alchemist_state.get_all(),
                "inputs": op_inputs,
            }
            self._json_response(data)
        elif self.path == "/api/refresh":
            self.collector.start_refresh()
            self._json_response({"ok": True})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        if self.path == "/api/alchemist/generate":
            op_name = payload.get("op_name", "")
            if not op_name:
                self._json_response({"error": "op_name required"}, 400)
                return
            if not check_alchemist_available():
                self._json_response({"error": "tt-alchemist not available"}, 400)
                return
            # Find the emitpy_input for this op (TTIR mlir file)
            input_mlir = None
            for oc in self.cfg.ops:
                if oc.op_name == op_name:
                    input_mlir = oc.emitpy_input
                    break
            if not input_mlir or not Path(input_mlir).exists():
                self._json_response(
                    {"error": f"No MLIR input found for {op_name}"}, 400
                )
                return
            _alchemist_state.start_generate(op_name, input_mlir)
            self._json_response({"ok": True})

        elif self.path == "/api/alchemist/run":
            op_name = payload.get("op_name", "")
            if not op_name:
                self._json_response({"error": "op_name required"}, 400)
                return
            _alchemist_state.start_run(op_name)
            self._json_response({"ok": True})

        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    op_names = [oc.op_name for oc in cfg.ops]

    print(f"\n  Add-Op Review: {', '.join(op_names)}")
    print(f"  {'=' * 40}")

    if cfg.static:
        # Synchronous collection for static output
        ops = {}
        for oc in cfg.ops:
            print(f"  Collecting {oc.op_name}...", flush=True)
            tests = {}
            if oc.ttnn_test_dir and Path(oc.ttnn_test_dir).exists():
                rc, output = run_cmd(["llvm-lit", oc.ttnn_test_dir, "-v"])
                passed, failed = parse_lit_output(output)
                tests["lit_ttnn"] = {
                    "status": "pass" if rc == 0 else "fail",
                    "output": output,
                    "passed": passed,
                    "failed": failed,
                    "count": f"{passed}/{passed + failed}",
                }
            if oc.emitc_test_dir and Path(oc.emitc_test_dir).exists():
                rc, output = run_cmd(["llvm-lit", oc.emitc_test_dir, "-v"])
                passed, failed = parse_lit_output(output)
                tests["lit_emitc"] = {
                    "status": "pass" if rc == 0 else "fail",
                    "output": output,
                    "passed": passed,
                    "failed": failed,
                    "count": f"{passed}/{passed + failed}",
                }
            pf = oc.pytest_filter or oc.op_name
            if pf:
                rc, output = run_cmd(
                    [
                        "pytest",
                        "test/python/golden/test_ttir_ops.py",
                        "-k",
                        pf,
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
            lookup_name = oc.ttnn_op_name or oc.op_name
            ops[oc.op_name] = {
                "info": collect_op_info(lookup_name),
                "tests": tests,
                "emitted_python": collect_emitted_python(oc.op_name, oc.emitpy_input),
                "emitted_cpp": collect_emitted_cpp(oc.op_name, oc.emitc_input),
            }

        print("  Collecting git diff...", flush=True)
        diff, diff_stat = collect_diff()

        data = {"ops": ops, "diff": diff, "diff_stat": diff_stat}
        html = generate_html(data)
        cfg.static.parent.mkdir(parents=True, exist_ok=True)
        cfg.static.write_text(html)
        print(f"\n  Static review written to: {cfg.static}\n")
        sys.exit(0)

    # Start server immediately, collect data in background
    collector = AsyncDataCollector(cfg)
    collector.start_refresh()

    port = cfg.port
    _kill_port(port)
    handler = partial(ReviewHandler, cfg, collector)
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
