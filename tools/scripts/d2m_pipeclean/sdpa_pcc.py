#!/usr/bin/env python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone PCC harness for the GQA+causal SDPA pipeclean snippet.

Compiles test/python/golden/mlir_snippets/ttir/ttir_sdpa_gqa_causal.mlir through
the ttir-to-ttmetal-pipeline, runs the resulting flatbuffer with deterministic
torch inputs, and compares against torch.nn.functional.scaled_dot_product_attention.

Bypasses tools/builder/* (which currently requires stablehlo bindings the build
doesn't have) by calling _ttmlir_runtime directly.

Usage:
    source env/activate
    ttrt query --save-artifacts   # if ttrt-artifacts/system_desc.ttsys is stale
    python tools/scripts/d2m_pipeclean/sdpa_pcc.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

import _ttmlir_runtime as tt_runtime

# Import golden.metrics by file path so we don't pull in golden.__init__'s
# stablehlo dependency (this build has TTMLIR_ENABLE_STABLEHLO=OFF).
import importlib.util as _ilu

_metrics_path = (
    Path(__file__).resolve().parents[3] / "build/python_packages/golden/metrics.py"
)
_spec = _ilu.spec_from_file_location("_golden_metrics", str(_metrics_path))
_metrics = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
get_atol_rtol_pcc = _metrics.get_atol_rtol_pcc

REPO_ROOT = Path(__file__).resolve().parents[3]
SNIPPET = REPO_ROOT / "test/python/golden/mlir_snippets/ttir/ttir_sdpa_gqa_causal.mlir"
SYSTEM_DESC = REPO_ROOT / "ttrt-artifacts/system_desc.ttsys"

# Shape constants -- must match the snippet.
B, HQ, HKV, S, D = 1, 8, 2, 64, 64
DTYPE = torch.bfloat16


def compile_snippet(out_ttm: Path) -> None:
    """Run ttir-to-ttmetal-pipeline + ttmetal-to-flatbuffer on the snippet."""
    with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as f:
        compiled_mlir = Path(f.name)
    subprocess.run(
        [
            "ttmlir-opt",
            f"--ttir-to-ttmetal-pipeline=system-desc-path={SYSTEM_DESC} "
            f"override-device-shape=1,1",
            "-o",
            str(compiled_mlir),
            str(SNIPPET),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ttmlir-translate",
            "--ttmetal-to-flatbuffer",
            "-o",
            str(out_ttm),
            str(compiled_mlir),
        ],
        check=True,
    )


def torch_sdpa_golden(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Golden using torch's reference SDPA with GQA broadcast + causal mask."""
    # torch SDPA wants K/V broadcast to match Q's head count for GQA. We do
    # this explicitly so the math matches the snippet's decomposition (which
    # uses reshape, not broadcast -- but it's the same algebra when the
    # repeat-interleave is along the head dim).
    groups = HQ // HKV
    k_g = k.repeat_interleave(groups, dim=1)
    v_g = v.repeat_interleave(groups, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q, k_g, v_g, is_causal=True)


def torch_to_runtime(t: torch.Tensor):
    # Match builder_runtime.create_tensor for the single-shard case.
    return tt_runtime.runtime.create_borrowed_host_tensor(
        t.data_ptr(),
        list(t.shape),
        list(t.stride()),
        t.element_size(),
        tt_runtime.runtime.DataType.BFloat16,
    )


def run_on_device(
    ttm: Path, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Submit the flatbuffer and return the device output as a torch tensor."""
    fbb = tt_runtime.binary.load_binary_from_path(str(ttm))

    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    # Use ETH dispatch (the runtime default) so the full 8x8 worker grid is
    # available for compute -- matches `ttrt run` without --disable-eth-dispatch.
    mesh_options.mesh_shape = list(fbb.get_program_mesh_shape(0))
    tt_runtime.runtime.set_current_device_runtime(
        tt_runtime.runtime.DeviceRuntime.TTMetal
    )
    device = tt_runtime.runtime.open_mesh_device(mesh_options)

    try:
        inputs = [torch_to_runtime(t) for t in (q, k, v)]
        converted = []
        for i, t in enumerate(inputs):
            layout = tt_runtime.runtime.get_layout(fbb, 0, i)
            converted.append(tt_runtime.runtime.to_layout(t, device, layout, True))

        runtime_outputs = tt_runtime.runtime.submit(device, fbb, 0, converted)
        tt_runtime.runtime.wait(runtime_outputs)

        output_host = tt_runtime.runtime.to_host(runtime_outputs[0], untilize=True)
        # output_host is a list-of-shards (one per device in the mesh).
        shard = output_host[0]
        buf = bytearray(shard.get_data_buffer())
        out = torch.frombuffer(buf, dtype=torch.bfloat16).reshape(shard.get_shape())
        # Detach from the live buffer before the runtime cleans up.
        out = out.clone()

        tt_runtime.runtime.deallocate_tensor(runtime_outputs[0], force=True)
        return out
    finally:
        tt_runtime.runtime.close_mesh_device(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pcc-threshold", type=float, default=0.95)
    parser.add_argument(
        "--keep-ttm",
        action="store_true",
        help="Don't delete the compiled .ttm; write to ./sdpa_gqa_causal.ttm",
    )
    args = parser.parse_args()

    if not SYSTEM_DESC.exists():
        print(
            f"error: {SYSTEM_DESC} not found; run `ttrt query --save-artifacts` first.",
            file=sys.stderr,
        )
        return 2

    torch.manual_seed(args.seed)
    q = torch.randn(B, HQ, S, D, dtype=DTYPE)
    k = torch.randn(B, HKV, S, D, dtype=DTYPE)
    v = torch.randn(B, HKV, S, D, dtype=DTYPE)

    golden = torch_sdpa_golden(q, k, v)

    if args.keep_ttm:
        ttm_path = Path("sdpa_gqa_causal.ttm")
        compile_snippet(ttm_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".ttm", delete=False) as f:
            ttm_path = Path(f.name)
        compile_snippet(ttm_path)

    out = run_on_device(ttm_path, q, k, v)
    if not args.keep_ttm:
        os.unlink(ttm_path)

    atol, rtol, pcc = get_atol_rtol_pcc(golden, out, atol=1e-3, rtol=1e-2)
    print(f"seed={args.seed}")
    print(f"shape: q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}")
    print(f"PCC  = {pcc:.6f}    (threshold {args.pcc_threshold})")
    print(f"atol = {atol:.6f}")
    print(f"rtol = {rtol:.6f}")

    return 0 if pcc >= args.pcc_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
