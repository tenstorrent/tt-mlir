# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import _ttmlir_runtime as tt_runtime
from builder.base.builder_apis import (
    load_mlir_file,
    compile_ttir_module_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb
from builder.base.builder_utils import get_artifact_dir
from conftest import clear_device_cache, get_request_kwargs

pytestmark = [pytest.mark.frontend("ttir")]

SNIPPETS_BASE_DIR = os.path.join(os.path.dirname(__file__), "../mlir_snippets")

# Each entry is a path under SNIPPETS_BASE_DIR (without the .mlir extension).
# Pytest auto-derives test IDs from these strings, so they double as the
# parametrize IDs. Use pytest.param(..., marks=...) to attach xfail/skip marks.
SNIPPETS = [
    "models/gpt_oss_20b/swiglu_prefill",
    "models/gpt_oss_20b/swiglu_decode",
    "models/gpt_oss_20b/attention_mask_prefill",
    "models/gpt_oss_20b/attention_mask_decode",
    "models/gpt_oss_20b/attention_mask_compare_prefill",
    "models/gpt_oss_20b/attention_mask_compare_decode",
    "models/gpt_oss_20b/compare_eq_multiply",
    "models/gpt_oss_20b/rope_sin_prefill",
    "models/gpt_oss_20b/rope_sin_decode",
    "models/gpt_oss_20b/rope_cos_prefill",
    "models/gpt_oss_20b/rope_cos_decode",
    "ttir/d2m_optimizer_two_d2m_subgraphs/unary_matmul_unary",
    "ttir/d2m_optimizer_two_d2m_subgraphs/eltwise_matmul_eltwise",
]

# Snippets that additionally exercise TTNNTraceHoistTransform (enable-trace=true).
# The D2M lowering of these subgraphs materializes scratch ttnn.empty buffers
# that land interleaved between hoistable ops; trace hoist rejects the
# non-hoistable op sitting between hoistable ops (issue #8402).
TRACE_SNIPPETS = [
    "models/gpt_oss_120b/rope_cos_sin_concat_prefill",
]


def _rope_cos_sin_concat_golden_inputs():
    """Realistic RoPE inputs for rope_cos_sin_concat_prefill.

    The snippet derives the rotary frequencies from position_ids via
    typecast -> reshape -> matmul. With the default random goldens,
    position_ids are large integers (randint up to 255) and inv_freq is
    standard-normal, so the matmul produces huge angles and the device
    cos/sin diverge from the f32 golden (PCC ~0.89). Seeding the inputs the
    way a real prefill does -- positions 0..seqlen-1 and the standard
    decaying inverse frequencies -- keeps the angles small and matches the
    >0.99 PCC observed in the full model.
    """
    seqlen = 17
    head_dim = 64  # concat of 32 cos + 32 sin freqs
    n_freq = 32

    position_ids = torch.arange(seqlen, dtype=torch.int32)
    inv_freq = (
        1.0
        / (10000.0 ** (torch.arange(0, n_freq, dtype=torch.float32) * 2.0 / head_dim))
    ).reshape(1, n_freq, 1)
    scale_cos = torch.ones(1, 1, 1, dtype=torch.float32)
    scale_sin = torch.ones(1, 1, 1, dtype=torch.float32)

    return {
        "rope_cos_sin_concat_prefill": [
            {0: position_ids},
            {0: inv_freq},
            {0: scale_cos},
            {0: scale_sin},
        ]
    }


# Snippets whose default random goldens produce numerically unstable values.
# Maps snippet path -> callable returning the golden_inputs dict for
# load_mlir_file (keyed by the MLIR func name).
GOLDEN_INPUTS_BUILDERS = {
    "models/gpt_oss_120b/rope_cos_sin_concat_prefill": _rope_cos_sin_concat_golden_inputs,
}


def _run_d2m_fusion_with_optimizer_test(
    request,
    target,
    snippet,
    *,
    enable_trace,
    artifact_subdir,
):
    mlir_path = os.path.join(SNIPPETS_BASE_DIR, f"{snippet}.mlir")
    if not os.path.exists(mlir_path):
        pytest.skip(f"MLIR not found: {mlir_path}")

    kwargs = get_request_kwargs(request)
    system_desc_path = kwargs.get(
        "system_desc_path", "ttrt-artifacts/system_desc.ttsys"
    )
    output_root = kwargs.get("output_root", ".")
    save_artifacts = kwargs.get("save_artifacts", False)
    artifact_dir = get_artifact_dir(
        output_root,
        f"{artifact_subdir}/{snippet.replace('/', '_')}",
        target,
        save_artifacts,
    )

    with open(mlir_path, "r") as f:
        mlir_content = f.read().strip()

    golden_inputs_builder = GOLDEN_INPUTS_BUILDERS.get(snippet)
    golden_inputs = golden_inputs_builder() if golden_inputs_builder else None
    module, builder = load_mlir_file(
        mlir_content, golden_inputs=golden_inputs, target="ttir"
    )
    pipeline_options = [
        "optimization-level=1",
        "enable-create-d2m-subgraphs=true",
    ]
    if enable_trace:
        pipeline_options.append("enable-trace=true")

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
        pipeline_options=pipeline_options,
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


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("snippet", SNIPPETS)
def test_d2m_fusion_with_optimizer(request, target, snippet):
    """E2E: TTIR with D2M fusion (optimization-level=1, enable-create-d2m-subgraphs) -> flatbuffer -> run.

    Compilation runs with no device open so the pipeline can use mock/simulator
    context for opmodel; device is opened only after compile for execute_fb.
    """
    _run_d2m_fusion_with_optimizer_test(
        request,
        target,
        snippet,
        enable_trace=False,
        artifact_subdir="d2m_fusion",
    )


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("snippet", TRACE_SNIPPETS)
def test_d2m_fusion_with_optimizer_and_trace(request, target, snippet):
    """E2E with enable-trace=true, exercising TTNNTraceHoistTransform.

    D2M subgraphs whose lowering materializes scratch ttnn.empty buffers fail
    trace hoist when a non-hoistable op lands between hoistable ops (issue #8402).
    """
    _run_d2m_fusion_with_optimizer_test(
        request,
        target,
        snippet,
        enable_trace=True,
        artifact_subdir="d2m_fusion_trace",
    )
