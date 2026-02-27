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
from typing import Dict, List, Tuple

from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb
from builder.base.builder_utils import get_artifact_dir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")

# Extract individual functions from an MLIR module.
def extract_functions_from_mlir(mlir_content: str) -> List[Tuple[str, str]]:
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
            functions.append((func_name, func_mlir))

    return functions


# Model IDs (path under mlir_snippets/models without .mlir) to exclude from
# discovery here; they are tested in test_d2m_fusion_with_optimizer.py instead.
SNIPPETS_TO_SKIP = {
    "gpt_oss_20b/gate_up",
    "gpt_oss_20b/rope_embedding",
}


# Discover all MLIR files and extract each function as a separate snippet.
def discover_model_mlir_snippets() -> Dict[str, Dict[str, str]]:
    models_dir = os.path.join(os.path.dirname(__file__), "mlir_snippets/models")
    snippets = {}

    if not os.path.exists(models_dir):
        return snippets

    for root, dirs, files in os.walk(models_dir):
        for filename in files:
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
                    for func_name, func_mlir in functions:
                        snippet_id = f"{model_id}/{func_name}"
                        snippets[snippet_id] = {
                            "path": file_path,
                            "content": func_mlir,
                            "func_name": func_name,
                            "model_id": model_id,
                        }

    return snippets


MODEL_MLIR_SNIPPETS = discover_model_mlir_snippets()


def get_snippet_ids() -> List[str]:
    return list(MODEL_MLIR_SNIPPETS.keys())


@pytest.mark.parametrize("snippet_id", get_snippet_ids())
@pytest.mark.parametrize("target", ["ttmetal"])
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

    print(f"\n{'='*60}")
    print(f"Testing: {snippet_id}")
    print(f"Model: {model_id}, Function: {func_name}")
    print(f"Target: {target}")
    print(f"{'='*60}")
    print(f"MLIR IR:\n{mlir_content}")
    print(f"{'='*60}")

    # Compile
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

    if skip_exec:
        print("Skipping execution (--skip-exec)")
        return

    # Execute using the standard runtime helper
    execute_fb(
        compiled_bin,
        input_output_goldens=input_output_goldens,
        intermediate_goldens=intermediate_goldens,
        device=device,
    )

    print("Execution successful")
    print(f"{'='*60}\n")
