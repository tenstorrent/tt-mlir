#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Standalone chisel runner for a pre-lowered TTNN-dialect .mlir file.

Pipeline:
  1. Translate <input>.mlir -> <input>.ttnn flatbuffer via ttmlir-translate.
  2. Open a (1,1) mesh device.
  3. Bind a chisel.session (isolated + accumulated numerics).
  4. For each public program: generate inputs (random floats; small valid
     integers for index/token operands), submit through the runtime so chisel's
     per-op debug hooks fire and record PCC per op.
  5. Write the chisel report to JSONL and print a summary.

Isolated-mode PCC compares each device op against a torch golden computed from
the *same* device inputs, so per-op numerical fidelity is meaningful even with
random inputs. Absolute end-to-end accuracy requires real prompts and is not
the goal here.
"""
import argparse
import os
import subprocess
import sys

import torch

# builder runtime helpers (input generation / layout conversion / submit)
from builder.base.builder_runtime import (
    create_tensor,
    convert_input_layouts,
    program_inputs_as_dict,
    runtime_str_dtype_to_torch_dtype,
)

import _ttmlir_runtime as tt_runtime

import chisel


TRANSLATE = os.environ.get(
    "TTMLIR_TRANSLATE",
    os.path.join(os.path.dirname(__file__), "..", "build", "bin", "ttmlir-translate"),
)

# Integer operands are indices (token ids, positions, page tables). Random
# 32-bit garbage causes out-of-bounds embedding lookups / paged-attention
# crashes, so clamp integer inputs to a small safe range.
INT_DTYPES = {
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.uint8, torch.uint16, torch.uint32, torch.uint64,
}
INT_MAX = 8  # safe small index range


def translate_to_flatbuffer(mlir_path: str, out_path: str) -> None:
    cmd = [TRANSLATE, "--ttnn-to-flatbuffer", mlir_path, "-o", out_path]
    print(f"[translate] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def make_input(i_dict, mesh_shape):
    shape = i_dict["desc"]["shape"]
    dtype = runtime_str_dtype_to_torch_dtype(
        i_dict["desc"]["layout"]["memory_desc"]["data_type"]
    )
    if dtype in INT_DTYPES:
        t = torch.randint(0, INT_MAX, shape, dtype=dtype)
    elif dtype == torch.bool:
        t = torch.randint(0, 2, shape).to(torch.bool)
    else:
        t = torch.randn(shape, dtype=dtype)

    unsharded = i_dict["desc"]["shard_status"] == "Unsharded"
    if unsharded:
        shards = {0: t.clone()}
    else:
        n = mesh_shape[0] * mesh_shape[1]
        shards = {j: t.clone() for j in range(n)}
    return create_tensor(shards, mesh_shape)


def run(mlir_path: str, jsonl_path: str, mesh_shape=(1, 1)) -> "chisel.ChiselReport":
    fb_path = os.path.splitext(mlir_path)[0] + ".ttnn"
    translate_to_flatbuffer(mlir_path, fb_path)

    fbb = tt_runtime.binary.load_binary_from_path(fb_path)

    mesh_options = tt_runtime.runtime.MeshDeviceOptions()
    mesh_options.mesh_shape = mesh_shape
    tt_runtime.runtime.set_current_device_runtime(
        tt_runtime.runtime.DeviceRuntime.TTNN
    )
    device = tt_runtime.runtime.open_mesh_device(mesh_options)

    report = None
    try:
        with chisel.session(
            results_path=jsonl_path,
            checks_config=chisel.ChiselChecksConfig(
                isolation=True, accumulation=True
            ),
        ) as rep:
            for program_index in range(fbb.get_num_programs()):
                if fbb.is_program_private(program_index):
                    print(f"[program {program_index}] private, skipping", flush=True)
                    continue
                name = fbb.get_program_name(program_index)
                print(f"[program {program_index}] {name}: running", flush=True)

                input_dict = program_inputs_as_dict(fbb, program_index)
                inputs = [make_input(d, mesh_shape) for d in input_dict]
                converted = convert_input_layouts(
                    device, inputs, fbb=fbb, program_index=program_index
                )
                outs = tt_runtime.runtime.submit(
                    device, fbb, program_index, converted
                )
                tt_runtime.runtime.wait(outs)
                print(f"[program {program_index}] {name}: done", flush=True)
            report = rep
            records = list(rep.records)
    finally:
        tt_runtime.runtime.close_mesh_device(device)

    print(f"\n[chisel] {len(records)} records -> {jsonl_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir", help="TTNN-dialect .mlir file")
    ap.add_argument("-o", "--out", default=None, help="output jsonl path")
    ap.add_argument("--mesh", default="1,1", help="mesh shape, e.g. 1,1")
    args = ap.parse_args()

    out = args.out or (os.path.splitext(args.mlir)[0] + ".chisel.jsonl")
    mesh = tuple(int(x) for x in args.mesh.split(","))
    run(args.mlir, out, mesh_shape=mesh)


if __name__ == "__main__":
    main()
