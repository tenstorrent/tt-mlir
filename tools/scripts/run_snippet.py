#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile (and optionally execute) individual MLIR function snippets.

Extracts a single function from a multi-function MLIR file (like ops.mlir),
wraps it in its own module, and runs the compilation pipeline for the
requested backend target.

Usage:
    # Compile a single function (compile-only, no device needed):
    python tools/run_snippet.py vllm_llama3_3b/graph_10_ttir/ops.mlir --func add_0 --target ttmetal

    # List all functions in the file:
    python tools/run_snippet.py vllm_llama3_3b/graph_10_ttir/ops.mlir --list

    # Compile all functions (stops on first failure by default):
    python tools/run_snippet.py vllm_llama3_3b/graph_10_ttir/ops.mlir --target ttmetal

    # Compile all functions, continue past failures:
    python tools/run_snippet.py vllm_llama3_3b/graph_10_ttir/ops.mlir --target ttmetal --keep-going

    # Also execute on device:
    python tools/run_snippet.py vllm_llama3_3b/graph_10_ttir/ops.mlir --func add_0 --target ttmetal --execute

    # Dump IR after each pass:
    python tools/run_snippet.py vllm_llama3_3b/graph_10_ttir/ops.mlir --func add_0 --target ttmetal --print-ir
"""

import argparse
import re
import sys
from typing import List, Tuple

from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb


def extract_functions_from_mlir(mlir_content: str) -> List[Tuple[str, str]]:
    """Extract individual functions from an MLIR module, each wrapped in its own module."""
    functions = []
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


def compile_snippet(
    mlir_content: str,
    func_name: str,
    target: str,
    system_desc_path: str,
    print_ir: bool,
    execute: bool,
    save_artifacts: bool,
    artifact_dir: str,
):
    """Compile (and optionally execute) a single MLIR snippet."""
    print(f"\n{'='*60}")
    print(f"Compiling: {func_name} -> {target}")
    print(f"{'='*60}")
    print(f"MLIR:\n{mlir_content}")
    print(f"{'='*60}")

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
    print("Compilation successful")

    if execute:
        execute_fb(
            compiled_bin,
            input_output_goldens=input_output_goldens,
            intermediate_goldens=intermediate_goldens,
        )
        print("Execution successful")


def main():
    parser = argparse.ArgumentParser(
        description="Compile individual MLIR function snippets."
    )
    parser.add_argument("mlir_file", help="Path to the MLIR file (e.g. ops.mlir)")
    parser.add_argument(
        "--func", "-f", help="Function name to compile (omit to compile all)"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List functions only")
    parser.add_argument(
        "--target",
        "-t",
        default="ttmetal",
        choices=["ttmetal", "ttnn"],
        help="Backend target (default: ttmetal)",
    )
    parser.add_argument(
        "--execute", "-e", action="store_true", help="Also execute on device"
    )
    parser.add_argument(
        "--print-ir", action="store_true", help="Dump IR after each pass"
    )
    parser.add_argument(
        "--system-desc",
        default="ttrt-artifacts/system_desc.ttsys",
        help="Path to system descriptor",
    )
    parser.add_argument(
        "--save-artifacts", action="store_true", help="Save compilation artifacts"
    )
    parser.add_argument(
        "--artifact-dir", default=".", help="Directory for saved artifacts"
    )
    parser.add_argument(
        "--keep-going",
        "-k",
        action="store_true",
        help="Continue past failures when compiling all functions",
    )
    args = parser.parse_args()

    with open(args.mlir_file, "r") as f:
        content = f.read().strip()

    functions = extract_functions_from_mlir(content)
    if not functions:
        print(f"No functions found in {args.mlir_file}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        for name, _ in functions:
            print(name)
        return

    if args.func:
        matches = [(n, m) for n, m in functions if n == args.func]
        if not matches:
            available = [n for n, _ in functions]
            print(
                f"Function '{args.func}' not found. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        targets = matches
    else:
        targets = functions

    passed, failed = 0, 0
    for func_name, func_mlir in targets:
        try:
            compile_snippet(
                func_mlir,
                func_name,
                target=args.target,
                system_desc_path=args.system_desc,
                print_ir=args.print_ir,
                execute=args.execute,
                save_artifacts=args.save_artifacts,
                artifact_dir=args.artifact_dir,
            )
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\nFAILED: {func_name}: {e}", file=sys.stderr)
            if not args.keep_going and len(targets) > 1:
                print("Stopping (use --keep-going to continue past failures)")
                break
            if len(targets) == 1:
                sys.exit(1)

    if len(targets) > 1:
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed out of {len(targets)}")
        print(f"{'='*60}")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
