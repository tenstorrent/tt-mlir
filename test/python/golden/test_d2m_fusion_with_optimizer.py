# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import _ttmlir_runtime as tt_runtime
from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb
from builder.base.builder_utils import get_artifact_dir
from conftest import clear_device_cache, get_request_kwargs

pytestmark = [pytest.mark.frontend("ttir")]

MLIR_SNIPPETS_DIR = os.path.join(
    os.path.dirname(__file__), "mlir_snippets/models/gpt_oss_20b"
)
GPT_OSS_20B_SNIPPETS = ["gate_up", "rope_embedding"]


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("snippet", GPT_OSS_20B_SNIPPETS)
def test_d2m_fusion_with_optimizer(request, target, snippet):
    """E2E: TTIR with D2M fusing (optimization-level=1, enable-d2m-fusing-pass) -> flatbuffer -> run.

    Compilation runs with no device open so the pipeline can use mock/simulator
    context for opmodel; device is opened only after compile for execute_fb.
    """
    mlir_path = os.path.join(MLIR_SNIPPETS_DIR, f"{snippet}.mlir")
    if not os.path.exists(mlir_path):
        pytest.skip(f"MLIR not found: {mlir_path}")

    kwargs = get_request_kwargs(request)
    system_desc_path = kwargs.get(
        "system_desc_path", "ttrt-artifacts/system_desc.ttsys"
    )
    output_root = kwargs.get("output_root", ".")
    save_artifacts = kwargs.get("save_artifacts", False)
    artifact_dir = get_artifact_dir(
        output_root, f"model_snippets/gpt_oss_20b_{snippet}", "ttnn", save_artifacts
    )

    with open(mlir_path, "r") as f:
        mlir_content = f.read().strip()

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
        pipeline_options=[
            "optimization-level=1",
            "enable-d2m-fusing-pass=true",
        ],
        print_ir=True,
    )
    # Open device only after compile so the pipeline can use mock context for opmodel.
    # If this is not done, we'll get this error: "Cannot switch to real hardware while 1 device(s) are active."
    device = request.getfixturevalue("device")

    try:
        execute_fb(
            compiled_bin,
            input_output_goldens=input_output_goldens,
            intermediate_goldens=intermediate_goldens,
            device=device,
        )
    finally:
        # Use the same close path as conftest session teardown so the runtime
        # fully releases the device; then clear cache so the next parametrized
        # test opens a fresh device (avoids "Cannot switch to real hardware"
        # and teardown crashes).
        tt_runtime.runtime.close_mesh_device(device)
        tt_runtime.runtime.set_fabric_config(tt_runtime.runtime.FabricConfig.DISABLED)
        clear_device_cache()
