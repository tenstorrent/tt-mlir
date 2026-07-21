# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections import OrderedDict

import pytest
import torch

from ttmlir.ir import Context, Location, Module
from ttmlir.passes import ttmetal_to_flatbuffer_bin

from builder.base.builder_apis import create_custom_ttir_pipeline_fn
from builder.base.builder_utils import run_ttir_pipeline
from builder.base.builder_runtime import execute_fb
from golden import GoldenMapTensor

pytestmark = pytest.mark.frontend("ttir")

_D2M_POST_GRID_WITH_VIEW_PIPELINE = (
    "d2m-lower-to-layout,d2m-materialize-view-returns,canonicalize,"
    "ttir-bufferization-pipeline,d2m-insert-scratch-buffers,"
    "d2m-generic-apply-interchange,d2m-generate-outer-loops,d2m-allocate,"
    "d2m-lower-multicast-loads,d2m-generic-lower-to-explicit-form,canonicalize,"
    "d2m-be-pipeline,d2m-to-ttkernel-pipeline,d2m-to-ttmetal-pipeline"
)

_SNIPPETS_DIR = os.path.join(os.path.dirname(__file__), "view_chain_snippets")


def _load_snippet(name):
    with open(os.path.join(_SNIPPETS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _compile_and_run_snippet(mlir_text, golden_input_output, request, device):
    ctx = Context()
    with ctx, Location.unknown(ctx):
        module = Module.parse(mlir_text)

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(_D2M_POST_GRID_WITH_VIEW_PIPELINE),
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", 1), ("y", 1)]),
    )
    _golden_report, output_tensors = execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        input_output_goldens={0: golden_input_output},
        device=device,
        save_artifacts=True,
        check_pcc=True,
        check_atol=True,
        check_rtol=True,
    )


def _im2col_nhwc_flat(t: torch.Tensor, K: int = 2) -> torch.Tensor:
    # Mirrors snippet: per (ki, kj) take slice t[:, ki:ki+H_out, kj:kj+W_out, :],
    # flatten to (N*H_out*W_out, C), then concat K*K on C dim → (N*H_out*W_out,
    # K*K*C). Block order (ki, kj) row-major; channel order preserved within.
    N, H, W, C = t.shape
    H_out, W_out = H - K + 1, W - K + 1
    cols = []
    for ki in range(K):
        for kj in range(K):
            patch = t[:, ki : ki + H_out, kj : kj + W_out, :]
            cols.append(patch.reshape(N * H_out * W_out, C))
    return torch.cat(cols, dim=1)


_DTYPE_MAP = {"f32": torch.float32, "bf16": torch.bfloat16}


def _substitute_dtype(mlir_text: str, dtype: str) -> str:
    # All element-type occurrences in our snippets are `xf32` (in shape decls)
    # or `f32` as a bare type (inside !ttcore.tile<...> attrs etc). Our RM
    # snippets only use the `xf32` form, so a single str.replace is sufficient.
    # Comments containing "f32" are stripped at parse time, so they are safe.
    return mlir_text.replace("xf32", f"x{dtype}")


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_composite_im2col_rm(target, dtype, request, device):
    # Single-core im2col, K=2, parametrized on element type.
    N, H, W, C = 1, 9, 9, 32
    K = 2
    H_out = W_out = H - K + 1  # 8
    in_shape = (N, H, W, C)
    out_shape = (N * H_out * W_out, K * K * C)  # (64, 128)

    torch_dtype = _DTYPE_MAP[dtype]
    arg0 = torch.arange(N * H * W * C, dtype=torch_dtype).reshape(in_shape)
    expected = _im2col_nhwc_flat(arg0, K=K)
    assert expected.shape == out_shape

    goldens = {
        "input_0": GoldenMapTensor({0: arg0}, (1, 1)),
        "output_0": GoldenMapTensor({0: expected}, (1, 1)),
    }

    _compile_and_run_snippet(
        _substitute_dtype(_load_snippet("composite_im2col_rm.mlir"), dtype),
        goldens,
        request,
        device,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_composite_im2col_rm_grid8x8(target, dtype, request, device):
    # im2col K=2 block-sharded on grid<8x8>, parametrized on element type.
    N, H, W, C = 1, 17, 17, 64
    K = 2
    H_out = H - K + 1  # 16
    W_out = W - K + 1  # 16
    out_shape = (N * H_out * W_out, K * K * C)  # (256, 256)

    torch_dtype = _DTYPE_MAP[dtype]
    arg0 = torch.arange(N * H * W * C, dtype=torch_dtype).reshape(N, H, W, C)
    expected = _im2col_nhwc_flat(arg0, K=K)
    assert expected.shape == out_shape

    goldens = {
        "input_0": GoldenMapTensor({0: arg0}, (1, 1)),
        "output_0": GoldenMapTensor({0: expected}, (1, 1)),
    }

    _compile_and_run_snippet(
        _substitute_dtype(_load_snippet("composite_im2col_rm_grid8x8.mlir"), dtype),
        goldens,
        request,
        device,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", ["f32", "bf16"])
def test_composite_im2col_rm_grid64x1(target, dtype, request, device):
    # N=1, H=W=66, C=32,
    # K=3 stride=1 no pad → H_out=W_out=64. NHW_flat=4096, KKC=9*32=288.
    # Per-core L1: input 24 KB + output 72 KB ≈ 96 KB.
    N, H, W, C = 1, 66, 66, 32
    K = 3
    H_out = H - K + 1  # 64
    W_out = W - K + 1  # 64
    out_shape = (N * H_out * W_out, K * K * C)  # (4096, 288)

    torch_dtype = _DTYPE_MAP[dtype]
    arg0 = torch.arange(N * H * W * C, dtype=torch_dtype).reshape(N, H, W, C)
    expected = _im2col_nhwc_flat(arg0, K=K)
    assert expected.shape == out_shape

    goldens = {
        "input_0": GoldenMapTensor({0: arg0}, (1, 1)),
        "output_0": GoldenMapTensor({0: expected}, (1, 1)),
    }

    _compile_and_run_snippet(
        _substitute_dtype(_load_snippet("composite_im2col_rm_grid64x1.mlir"), dtype),
        goldens,
        request,
        device,
    )
