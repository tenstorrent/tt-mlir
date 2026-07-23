#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Chisel op-fidelity probe for the Falcon3-7B decoder compute ops that do NOT
require the paged KV cache or blackhole L1 sharding, so they run on n150:
the four projection matmuls (bf16 weights) and rms_norm.

Emits isolated + accumulated PCC per op. Isolated PCC = device op vs torch
golden on the same device inputs -> a direct per-op numerical-fidelity check.
"""
import json
import sys
from typing import List, Optional

import torch

import chisel
import _ttmlir_runtime as tt_runtime
from ttmlir.dialects import ttnn
from builder.base.builder_apis import compile_and_execute_ttnn
from builder.base.builder_utils import Operand
from builder.ttnn.ttnn_builder import TTNNBuilder

_DEV = None


def _device():
    global _DEV
    if _DEV is None:
        mo = tt_runtime.runtime.MeshDeviceOptions()
        mo.mesh_shape = (1, 1)
        tt_runtime.runtime.set_current_device_runtime(
            tt_runtime.runtime.DeviceRuntime.TTNN
        )
        _DEV = tt_runtime.runtime.open_mesh_device(mo)
    return _DEV

H = 3072  # Falcon3-7B hidden size
B = 32    # decode batch (1 token each)

# (name, lhs_shape, rhs_shape, transpose_b) for the 4 projection matmuls.
MATMULS = [
    ("qkv_proj",     (B, H),      (5120, H),   True),   # 32x3072 @ (5120x3072)^T
    ("o_proj",       (B, H),      (H, H),      True),
    ("gate_up_proj", (B, H),      (46080, H),  True),
    ("down_proj",    (B, 23040),  (H, 23040),  True),
]


def run():
    results = {}
    for name, xs, ws, tb in MATMULS:
        def module(builder: TTNNBuilder, xs=xs, ws=ws, tb=tb):
            @builder.func([xs, ws], [torch.bfloat16, torch.bfloat16])
            def fn(x: Operand, w: Operand, builder: TTNNBuilder,
                   unit_attrs: Optional[List[str]] = None):
                return builder.matmul(x, w, transpose_b=tb)

        with chisel.session(
            checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True)
        ) as rep:
            compile_and_execute_ttnn(
                module, test_base=f"probe_{name}", output_root="/tmp/probe_out",
                target="ttnn", device=_device(),
            )
            recs = [r for r in rep.records if r.check == "numerics"
                    and hasattr(r.payload, "pcc")]
        by_mode = {}
        for r in recs:
            by_mode.setdefault(r.payload.mode.value, []).append(r.payload.pcc)
        results[name] = {m: min(v) for m, v in by_mode.items()}
        print(f"[matmul {name}] {results[name]}", flush=True)

    # rms_norm at hidden size
    def rms_module(builder: TTNNBuilder):
        @builder.func([(B, H), (H,)], [torch.bfloat16, torch.bfloat16])
        def fn(x: Operand, w: Operand, builder: TTNNBuilder,
               unit_attrs: Optional[List[str]] = None):
            return builder.rms_norm(x, weight=w, epsilon=1e-6)

    with chisel.session(
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True)
    ) as rep:
        compile_and_execute_ttnn(
            rms_module, test_base="probe_rms_norm", output_root="/tmp/probe_out",
            target="ttnn", device=_device(),
        )
        recs = [r for r in rep.records if r.check == "numerics"
                and hasattr(r.payload, "pcc")]
    by_mode = {}
    for r in recs:
        by_mode.setdefault(r.payload.mode.value, []).append(r.payload.pcc)
    results["rms_norm"] = {m: min(v) for m, v in by_mode.items()}
    print(f"[rms_norm] {results['rms_norm']}", flush=True)

    with open("/localdev/dgolubovic/repos/tt-mlir/falcon3_chisel/probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    tt_runtime.runtime.close_mesh_device(_device())
    print("\nsaved probe_results.json")


if __name__ == "__main__":
    run()
