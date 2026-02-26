# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb
from builder.base.builder_utils import get_artifact_dir
from conftest import get_request_kwargs

pytestmark = [pytest.mark.frontend("ttir")]

MLIR_PATH = os.path.join(
    os.path.dirname(__file__), "mlir_snippets/models/gpt_oss_20b/gate_up.mlir"
)


@pytest.mark.parametrize("target", ["ttnn"])
def test_d2m_fusion_with_optimizer(request, target):
    """E2E: TTIR with D2M fusing (optimization-level=1, enable-d2m-fusing-pass) -> flatbuffer -> run.

    Compilation runs with no device open so the pipeline can use mock/simulator
    context for opmodel; device is opened only after compile for execute_fb.
    """
    if not os.path.exists(MLIR_PATH):
        pytest.skip(f"MLIR not found: {MLIR_PATH}")

    kwargs = get_request_kwargs(request)
    system_desc_path = kwargs.get(
        "system_desc_path", "ttrt-artifacts/system_desc.ttsys"
    )
    output_root = kwargs.get("output_root", ".")
    save_artifacts = kwargs.get("save_artifacts", False)
    artifact_dir = get_artifact_dir(
        output_root, "model_snippets/gpt_oss_20b_gate_up", "ttnn", save_artifacts
    )

    with open(MLIR_PATH, "r") as f:
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
    )
    # Open device only after compile so the pipeline can use mock context for opmodel.
    # If this is not done, we'll get this error: "Cannot switch to real hardware while 1 device(s) are active."
    device = request.getfixturevalue("device")
    execute_fb(
        compiled_bin,
        input_output_goldens=input_output_goldens,
        intermediate_goldens=intermediate_goldens,
        device=device,
    )
