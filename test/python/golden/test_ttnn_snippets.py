# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# PCC regression tests at "transformer layer" granularity for the TTNN
# backend. Sources the canonical TTIR layer files from
# test/ttmlir/models/single_blocks_and_layers/, compiles them through the TTNN
# pipeline, and runs golden PCC via the standard runtime helper.

import os
from typing import Dict

import pytest

from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb
from builder.base.builder_utils import get_artifact_dir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")

# Canonical TTIR transformer-layer snapshots live here.
SINGLE_BLOCKS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "ttmlir",
    "models",
    "single_blocks_and_layers",
)

# Subset of files under SINGLE_BLOCKS_DIR (basename without .mlir) that this
# suite exercises. Curated positively because the directory holds many more
# layers than we want to gate CI on.
ALLOWLIST = [
    "bert_encoder_layer",
    "gemma_2_2b_decode_layer",
    "llama_3_2_1b_decode_layer",
]


def discover_ttnn_model_mlir_snippets() -> Dict[str, str]:
    snippets = {}
    for model_id in ALLOWLIST:
        file_path = os.path.join(SINGLE_BLOCKS_DIR, f"{model_id}.mlir")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"ALLOWLIST entry '{model_id}' has no matching .mlir under "
                f"{SINGLE_BLOCKS_DIR}"
            )
        snippets[model_id] = file_path
    return snippets


TTNN_MODEL_MLIR_SNIPPETS = discover_ttnn_model_mlir_snippets()


@pytest.mark.parametrize("snippet_id", list(TTNN_MODEL_MLIR_SNIPPETS.keys()))
def test_ttnn_model_snippet_compile_execute(
    snippet_id: str,
    request,
    device,
):
    kwargs = get_request_kwargs(request)
    system_desc_path = kwargs["system_desc_path"]
    output_root = kwargs["output_root"]
    save_artifacts = kwargs.get("save_artifacts", False)
    print_ir = kwargs.get("print_ir", False)
    skip_exec = kwargs.get("skip_exec", False)

    with open(TTNN_MODEL_MLIR_SNIPPETS[snippet_id], "r") as f:
        mlir_content = f.read()

    artifact_dir = get_artifact_dir(
        output_root, f"ttnn_model_snippets/{snippet_id}", "ttnn", save_artifacts
    )

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
        target="ttnn",
        save_artifacts=save_artifacts,
        print_ir=print_ir,
    )
    print(f"Compilation successful: {snippet_id}")

    if skip_exec:
        print(f"Skipping execution (--skip-exec): {snippet_id}")
        return

    execute_fb(
        compiled_bin,
        input_output_goldens=input_output_goldens,
        intermediate_goldens=intermediate_goldens,
        device=device,
    )
